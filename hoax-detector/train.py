"""
Pipeline Training: Deteksi Hoaks Politik Indonesia
===================================================
Script ini menjalankan pipeline training lengkap:
  1. Memuat dan memvalidasi dataset
  2. Preprocessing teks (via text_cleaner.py)
  3. Stratified Split (80:20)
  4. TF-IDF Vectorization
  5. SMOTE pada data training
  6. GridSearchCV untuk Multinomial Naive Bayes dan SVM
  7. Evaluasi performa (Accuracy, Precision, Recall, F1, ROC-AUC)
  8. Ekspor model terbaik ke folder 'model/'

Cara penggunaan:
    python train.py

Pastikan file 'data/indonesian_hoax_news.csv' tersedia sebelum menjalankan.

Author  : Indonesian Hoax Detection System
Version : 1.0.0
"""

import os
import sys
import time
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif untuk server
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# Tambahkan root direktori ke path agar import utils berjalan
sys.path.insert(0, str(Path(__file__).parent))
from utils.text_cleaner import preprocess_batch, SASTRAWI_AVAILABLE

# Abaikan warning yang tidak penting
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Path direktori
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / 'data' / 'indonesian_hoax_news.csv'
MODEL_DIR  = BASE_DIR / 'model'
REPORT_DIR = BASE_DIR / 'model' / 'reports'

# Parameter TF-IDF
TFIDF_CONFIG = {
    'max_features': 10000,
    'ngram_range'  : (1, 2),
    'sublinear_tf' : True,       # Terapkan log(TF) untuk mengurangi dominasi kata sangat sering
    'min_df'       : 2,          # Abaikan term yang muncul < 2 dokumen
}

# Parameter GridSearch - Multinomial Naive Bayes
MNB_PARAM_GRID = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}

# Parameter GridSearch - SVM Linear
SVM_PARAM_GRID = {
    'estimator__C': [0.01, 0.1, 1.0, 5.0, 10.0]
}

# Kolom yang mungkin ada di dataset
POSSIBLE_TEXT_COLS  = ['text', 'content', 'berita', 'judul', 'artikel', 'narasi', 'title', 'hoax_text']
POSSIBLE_LABEL_COLS = ['label', 'hoax', 'class', 'kategori', 'target', 'status']

# Warna untuk visualisasi
PALETTE = ['#2ecc71', '#e74c3c']  # Hijau = Fakta, Merah = Hoaks


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI HELPER
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directories():
    """Buat direktori yang diperlukan jika belum ada."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Direktori model: {MODEL_DIR}")
    logger.info(f"Direktori laporan: {REPORT_DIR}")


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Muat dataset dari file CSV dengan validasi kolom.

    Args:
        path (Path): Path ke file CSV dataset.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'text' dan 'label'.

    Raises:
        FileNotFoundError : Jika file tidak ditemukan.
        ValueError        : Jika kolom tidak dikenali.
    """
    logger.info(f"Memuat dataset dari: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] File dataset tidak ditemukan: {path}\n"
            f"Pastikan file 'indonesian_hoax_news.csv' ada di folder 'data/'.\n"
        )

    try:
        # Coba beberapa encoding umum
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'utf-8-sig']:
            try:
                df = pd.read_csv(path, encoding=encoding)
                logger.info(f"Dataset berhasil dimuat dengan encoding '{encoding}'.")
                break
            except UnicodeDecodeError:
                continue

        logger.info(f"Shape dataset asli: {df.shape}")
        logger.info(f"Kolom tersedia: {list(df.columns)}")

    except Exception as e:
        raise ValueError(f"Gagal membaca file CSV: {e}")

    # ── Deteksi kolom teks ──
    text_col = None
    for col in POSSIBLE_TEXT_COLS:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        # Coba kolom pertama yang berisi string panjang
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().mean() > 50:
                text_col = col
                logger.warning(f"Kolom teks tidak ditemukan secara eksplisit. Menggunakan: '{col}'")
                break

    if text_col is None:
        raise ValueError(
            f"Kolom teks tidak ditemukan. Kolom tersedia: {list(df.columns)}\n"
            f"Kolom yang dicari: {POSSIBLE_TEXT_COLS}"
        )

    # ── Deteksi kolom label ──
    label_col = None
    for col in POSSIBLE_LABEL_COLS:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(
            f"Kolom label tidak ditemukan. Kolom tersedia: {list(df.columns)}\n"
            f"Kolom yang dicari: {POSSIBLE_LABEL_COLS}"
        )

    logger.info(f"Kolom teks  : '{text_col}'")
    logger.info(f"Kolom label : '{label_col}'")

    # Buat DataFrame bersih
    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']

    # ── Normalisasi label ──
    # Konversi label ke numerik (0 = Fakta, 1 = Hoaks)
    unique_labels = df['label'].unique()
    logger.info(f"Label unik ditemukan: {unique_labels}")

    if df['label'].dtype == object:
        # Map label string ke integer
        label_map = {}
        for lbl in unique_labels:
            lbl_lower = str(lbl).lower().strip()
            if lbl_lower in ['hoax', 'hoaks', '1', 'true', 'yes', 'positif', 'fake']:
                label_map[lbl] = 1
            elif lbl_lower in ['fakta', 'fact', '0', 'false', 'no', 'negatif', 'real']:
                label_map[lbl] = 0
            else:
                label_map[lbl] = int(lbl) if str(lbl).isdigit() else 0

        df['label'] = df['label'].map(label_map)
        logger.info(f"Peta label: {label_map}")

    df['label'] = df['label'].astype(int)

    # ── Hapus baris kosong ──
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != '']
    df = df.reset_index(drop=True)
    after = len(df)
    logger.info(f"Baris setelah pembersihan: {after} (dihapus: {before - after})")

    # ── Tampilkan distribusi label ──
    dist = df['label'].value_counts()
    logger.info(f"Distribusi label:\n  Fakta (0): {dist.get(0, 0)}\n  Hoaks (1): {dist.get(1, 0)}")

    return df


def visualize_data_distribution(df: pd.DataFrame):
    """Buat dan simpan visualisasi distribusi data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Analisis Distribusi Dataset Hoaks Politik Indonesia', fontsize=14, fontweight='bold')

    # ── Plot 1: Distribusi Label ──
    label_counts = df['label'].value_counts()
    label_names  = ['Fakta (0)', 'Hoaks (1)']
    colors = PALETTE

    axes[0].bar(label_names, [label_counts.get(0, 0), label_counts.get(1, 0)],
                color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Distribusi Label', fontweight='bold')
    axes[0].set_ylabel('Jumlah Sampel')
    for i, v in enumerate([label_counts.get(0, 0), label_counts.get(1, 0)]):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # ── Plot 2: Distribusi Panjang Teks ──
    df['text_length'] = df['text'].str.len()
    for idx, (label, color) in enumerate(zip([0, 1], colors)):
        subset = df[df['label'] == label]['text_length']
        axes[1].hist(subset, bins=50, alpha=0.7, color=color,
                     label=f"{'Fakta' if label == 0 else 'Hoaks'} (n={len(subset)})")
    axes[1].set_title('Distribusi Panjang Teks', fontweight='bold')
    axes[1].set_xlabel('Jumlah Karakter')
    axes[1].set_ylabel('Frekuensi')
    axes[1].legend()

    plt.tight_layout()
    save_path = REPORT_DIR / 'data_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualisasi distribusi data disimpan di: {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Buat dan simpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Fakta (0)', 'Hoaks (1)']
    )
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold', fontsize=13)

    save_path = REPORT_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix disimpan di: {save_path}")


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluasi performa model pada test set.

    Args:
        model      : Model yang sudah dilatih (dengan predict_proba).
        X_test     : Fitur test set (sparse matrix TF-IDF).
        y_test     : Label test set.
        model_name : Nama model untuk logging.

    Returns:
        dict: Dictionary berisi semua metrik evaluasi.
    """
    y_pred = model.predict(X_test)

    # Hitung probabilitas untuk ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except AttributeError:
        roc_auc = 0.0
        logger.warning(f"{model_name} tidak mendukung predict_proba.")

    # Hitung semua metrik
    metrics = {
        'model_name': model_name,
        'accuracy'  : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall'    : recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score'  : f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc'   : roc_auc,
    }

    # Tampilkan hasil
    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1-Score   : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"\n  Laporan Klasifikasi Lengkap:")
    print(classification_report(y_test, y_pred, target_names=['Fakta', 'Hoaks'], zero_division=0))

    # Buat confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TRAINING UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def train():
    """
    Fungsi utama training pipeline.
    Menjalankan semua langkah dari memuat data hingga menyimpan model terbaik.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  MULAI TRAINING: DETEKSI HOAKS POLITIK INDONESIA")
    logger.info("=" * 60)
    logger.info(f"PySastrawi tersedia: {SASTRAWI_AVAILABLE}")

    # ── Langkah 0: Siapkan direktori ──
    ensure_directories()

    # ── Langkah 1: Muat dataset ──
    try:
        df = load_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Langkah 2: Visualisasi distribusi data asli ──
    visualize_data_distribution(df)

    # ── Langkah 3: Preprocessing teks ──
    logger.info("\nMemulai preprocessing teks... (ini mungkin memerlukan beberapa menit)")
    df['clean_text'] = preprocess_batch(df['text'].tolist(), apply_stemming=True)

    # Hapus teks kosong setelah preprocessing
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    logger.info(f"Total sampel setelah preprocessing: {len(df)}")

    # Simpan contoh hasil preprocessing
    sample_preview = df[['text', 'clean_text', 'label']].head(5)
    sample_preview.to_csv(REPORT_DIR / 'sample_preprocessing.csv', index=False, encoding='utf-8')

    # ── Langkah 4: Stratified Split (80:20) ──
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # Jaga proporsi label di train dan test
    )

    logger.info(f"\nSplit dataset (Stratified 80:20):")
    logger.info(f"  Training set : {len(X_train)} sampel")
    logger.info(f"  Test set     : {len(X_test)} sampel")
    logger.info(f"  Distribusi train - Fakta: {sum(y_train==0)}, Hoaks: {sum(y_train==1)}")
    logger.info(f"  Distribusi test  - Fakta: {sum(y_test==0)}, Hoaks: {sum(y_test==1)}")

    # ── Langkah 5: TF-IDF Vectorization ──
    logger.info("\nMembangun TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(**TFIDF_CONFIG)

    X_train_tfidf = tfidf.fit_transform(X_train)  # Fit HANYA pada training set
    X_test_tfidf  = tfidf.transform(X_test)        # Transform test set

    logger.info(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    logger.info(f"Shape X_train TF-IDF: {X_train_tfidf.shape}")
    logger.info(f"Shape X_test TF-IDF : {X_test_tfidf.shape}")

    # ── Langkah 6: SMOTE pada Training Data ──
    logger.info("\nMenerapkan SMOTE untuk menangani ketidakseimbangan kelas...")
    logger.info(f"  Sebelum SMOTE - Fakta: {sum(y_train==0)}, Hoaks: {sum(y_train==1)}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
        logger.info(f"  Setelah SMOTE - Fakta: {sum(y_train_smote==0)}, Hoaks: {sum(y_train_smote==1)}")
    except ValueError as e:
        logger.warning(f"SMOTE gagal: {e}. Menggunakan data training tanpa SMOTE.")
        X_train_smote, y_train_smote = X_train_tfidf, y_train

    # ── Langkah 7: GridSearchCV - Multinomial Naive Bayes ──
    logger.info("\n" + "─"*50)
    logger.info("  TAHAP 1: GridSearchCV - Multinomial Naive Bayes")
    logger.info("─"*50)

    # MNB membutuhkan nilai non-negatif; gunakan X_train_tfidf asli jika SMOTE menghasilkan negatif
    # Namun TF-IDF selalu non-negatif, jadi aman
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mnb_grid = GridSearchCV(
        estimator  = MultinomialNB(),
        param_grid = MNB_PARAM_GRID,
        cv         = cv_strategy,
        scoring    = 'f1_weighted',
        n_jobs     = -1,
        verbose    = 1
    )
    mnb_grid.fit(X_train_smote, y_train_smote)

    logger.info(f"  Best params MNB  : {mnb_grid.best_params_}")
    logger.info(f"  Best CV F1 MNB   : {mnb_grid.best_score_:.4f}")

    best_mnb = mnb_grid.best_estimator_

    # ── Langkah 8: GridSearchCV - SVM Linear ──
    logger.info("\n" + "─"*50)
    logger.info("  TAHAP 2: GridSearchCV - SVM (Linear Kernel)")
    logger.info("─"*50)

    # CalibratedClassifierCV membungkus LinearSVC agar bisa menghasilkan probabilitas
    svm_base = CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42))

    svm_grid = GridSearchCV(
        estimator  = svm_base,
        param_grid = SVM_PARAM_GRID,
        cv         = cv_strategy,
        scoring    = 'f1_weighted',
        n_jobs     = -1,
        verbose    = 1
    )
    svm_grid.fit(X_train_smote, y_train_smote)

    logger.info(f"  Best params SVM  : {svm_grid.best_params_}")
    logger.info(f"  Best CV F1 SVM   : {svm_grid.best_score_:.4f}")

    best_svm = svm_grid.best_estimator_

    # ── Langkah 9: Evaluasi pada Test Set ──
    logger.info("\n" + "="*60)
    logger.info("  EVALUASI MODEL PADA TEST SET")
    logger.info("="*60)

    metrics_mnb = evaluate_model(best_mnb, X_test_tfidf, y_test, "Multinomial Naive Bayes")
    metrics_svm = evaluate_model(best_svm, X_test_tfidf, y_test, "SVM Linear Kernel")

    # ── Langkah 10: Tentukan Model Terbaik ──
    logger.info("\n" + "="*60)
    logger.info("  PERBANDINGAN MODEL")
    logger.info("="*60)

    comparison_df = pd.DataFrame([metrics_mnb, metrics_svm])
    comparison_df = comparison_df.set_index('model_name')

    print("\n  Tabel Perbandingan Metrik:")
    print(comparison_df.to_string())

    # Simpan tabel perbandingan
    comparison_df.to_csv(REPORT_DIR / 'model_comparison.csv')

    # Model terbaik berdasarkan F1-Score
    if metrics_svm['f1_score'] >= metrics_mnb['f1_score']:
        best_model      = best_svm
        best_model_name = "SVM Linear Kernel"
        best_metrics    = metrics_svm
    else:
        best_model      = best_mnb
        best_model_name = "Multinomial Naive Bayes"
        best_metrics    = metrics_mnb

    logger.info(f"\n  ✅ MODEL TERPILIH: {best_model_name}")
    logger.info(f"     F1-Score : {best_metrics['f1_score']:.4f}")
    logger.info(f"     Accuracy : {best_metrics['accuracy']:.4f}")
    logger.info(f"     ROC-AUC  : {best_metrics['roc_auc']:.4f}")

    # ── Langkah 11: Simpan Model & Vectorizer ──
    logger.info("\nMenyimpan artefak model...")

    # Simpan vectorizer TF-IDF
    tfidf_path = MODEL_DIR / 'tfidf_vectorizer.pkl'
    joblib.dump(tfidf, tfidf_path)
    logger.info(f"  TF-IDF Vectorizer disimpan: {tfidf_path}")

    # Simpan model terbaik
    best_model_path = MODEL_DIR / 'best_model.pkl'
    joblib.dump(best_model, best_model_path)
    logger.info(f"  Model terbaik disimpan   : {best_model_path}")

    # Simpan kedua model (untuk referensi)
    joblib.dump(best_mnb, MODEL_DIR / 'mnb_model.pkl')
    joblib.dump(best_svm, MODEL_DIR / 'svm_model.pkl')
    logger.info("  Semua model disimpan.")

    # Simpan metadata training
    import json
    metadata = {
        'best_model'        : best_model_name,
        'best_model_params' : str(best_model),
        'tfidf_config'      : TFIDF_CONFIG,
        'training_samples'  : len(X_train),
        'test_samples'      : len(X_test),
        'smote_applied'     : True,
        'metrics'           : {
            'MNB': {k: float(v) for k, v in metrics_mnb.items() if k != 'model_name'},
            'SVM': {k: float(v) for k, v in metrics_svm.items() if k != 'model_name'},
        },
        'label_mapping'     : {'0': 'Fakta', '1': 'Hoaks'},
        'training_duration' : f"{(time.time() - start_time):.2f} detik"
    }

    with open(MODEL_DIR / 'training_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    logger.info("  Metadata training disimpan.")

    # ── Selesai ──
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅ TRAINING SELESAI dalam {elapsed:.2f} detik")
    logger.info(f"  Model & artefak tersimpan di: {MODEL_DIR}")
    logger.info(f"  Laporan tersimpan di: {REPORT_DIR}")
    logger.info(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
