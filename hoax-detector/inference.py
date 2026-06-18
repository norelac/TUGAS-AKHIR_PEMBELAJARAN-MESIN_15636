"""
Pipeline Inferensi: Deteksi Hoaks Politik Indonesia
====================================================
Modul ini memuat model yang telah dilatih dan melakukan prediksi
pada teks baru dengan output berupa label dan skor kepercayaan.

Cara penggunaan:
    from inference import HoaxDetector
    detector = HoaxDetector()
    result   = detector.predict("Teks berita yang ingin diperiksa...")
    print(result)

Author  : Indonesian Hoax Detection System
Version : 1.0.0
"""

import sys
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Union

# Tambahkan root direktori ke path
sys.path.insert(0, str(Path(__file__).parent))
from utils.text_cleaner import preprocess_text

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path direktori
BASE_DIR       = Path(__file__).parent
MODEL_DIR      = BASE_DIR / 'model'
TFIDF_PATH     = MODEL_DIR / 'tfidf_vectorizer.pkl'
BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
MNB_PATH       = MODEL_DIR / 'mnb_model.pkl'
SVM_PATH       = MODEL_DIR / 'svm_model.pkl'

# Label mapping
LABEL_MAP = {
    0: {'label': 'FAKTA', 'emoji': '✅', 'color': 'green'},
    1: {'label': 'HOAKS', 'emoji': '🚨', 'color': 'red'}
}


class HoaxDetector:
    """
    Kelas utama untuk mendeteksi hoaks pada teks berbahasa Indonesia.

    Attributes:
        tfidf     : TF-IDF Vectorizer yang sudah dilatih.
        model     : Model klasifikasi terbaik (MNB atau SVM).
        mnb_model : Model Multinomial Naive Bayes (untuk perbandingan).
        svm_model : Model SVM Linear (untuk perbandingan).
        is_ready  : Status kesiapan model.
    """

    def __init__(self, model_type: str = 'best'):
        """
        Inisialisasi dan muat model dari disk.

        Args:
            model_type (str): Jenis model yang dimuat.
                              'best' = model terbaik dari training,
                              'mnb'  = Multinomial Naive Bayes,
                              'svm'  = SVM Linear.
        """
        self.tfidf     = None
        self.model     = None
        self.mnb_model = None
        self.svm_model = None
        self.is_ready  = False
        self.model_type = model_type

        self._load_models(model_type)

    def _load_models(self, model_type: str = 'best'):
        """
        Muat TF-IDF vectorizer dan model dari file .pkl.

        Args:
            model_type (str): Jenis model yang dimuat.
        """
        try:
            # ── Muat TF-IDF Vectorizer ──
            if not TFIDF_PATH.exists():
                raise FileNotFoundError(
                    f"TF-IDF Vectorizer tidak ditemukan di: {TFIDF_PATH}\n"
                    "Jalankan 'python train.py' terlebih dahulu."
                )
            self.tfidf = joblib.load(TFIDF_PATH)
            logger.info(f"TF-IDF Vectorizer berhasil dimuat. Vocab: {len(self.tfidf.vocabulary_)} term.")

            # ── Muat Model Utama ──
            model_path_map = {
                'best': BEST_MODEL_PATH,
                'mnb' : MNB_PATH,
                'svm' : SVM_PATH,
            }
            selected_path = model_path_map.get(model_type, BEST_MODEL_PATH)

            if not selected_path.exists():
                raise FileNotFoundError(
                    f"Model '{model_type}' tidak ditemukan di: {selected_path}\n"
                    "Jalankan 'python train.py' terlebih dahulu."
                )
            self.model = joblib.load(selected_path)
            logger.info(f"Model '{model_type}' berhasil dimuat: {type(self.model).__name__}")

            # ── Muat Kedua Model untuk Perbandingan (opsional) ──
            if MNB_PATH.exists():
                self.mnb_model = joblib.load(MNB_PATH)
            if SVM_PATH.exists():
                self.svm_model = joblib.load(SVM_PATH)

            self.is_ready = True
            logger.info("✅ HoaxDetector siap digunakan.")

        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            self.is_ready = False
        except Exception as e:
            logger.error(f"❌ Gagal memuat model: {e}")
            self.is_ready = False

    def _validate_input(self, text: str) -> str:
        """
        Validasi dan bersihkan teks input dari pengguna.

        Args:
            text (str): Teks input.

        Returns:
            str: Teks yang sudah divalidasi.

        Raises:
            ValueError: Jika teks kosong atau tidak valid.
        """
        if not isinstance(text, str):
            raise ValueError("Input harus berupa string teks.")
        if not text.strip():
            raise ValueError("Input teks tidak boleh kosong.")
        if len(text.strip()) < 10:
            raise ValueError("Teks terlalu pendek (minimal 10 karakter).")
        return text.strip()

    def predict(self, text: str) -> dict:
        """
        Prediksi apakah teks merupakan hoaks atau fakta.

        Args:
            text (str): Teks berita yang akan diperiksa.

        Returns:
            dict: Hasil prediksi berisi:
                - label        (str)  : 'HOAKS' atau 'FAKTA'
                - label_id     (int)  : 0 (Fakta) atau 1 (Hoaks)
                - confidence   (float): Skor kepercayaan [0.0 - 1.0]
                - prob_fakta   (float): Probabilitas kelas Fakta
                - prob_hoaks   (float): Probabilitas kelas Hoaks
                - clean_text   (str)  : Teks setelah preprocessing
                - emoji        (str)  : Emoji representasi label
                - color        (str)  : Warna representasi label
                - error        (str)  : Pesan error jika ada

        Example:
            >>> detector = HoaxDetector()
            >>> result = detector.predict("Jokowi menjual pulau ke China")
            >>> print(result['label'], result['confidence'])
        """
        # Periksa apakah model siap
        if not self.is_ready:
            return {
                'label'     : 'ERROR',
                'label_id'  : -1,
                'confidence': 0.0,
                'prob_fakta': 0.0,
                'prob_hoaks': 0.0,
                'clean_text': '',
                'emoji'     : '❌',
                'color'     : 'gray',
                'error'     : 'Model belum dimuat. Jalankan train.py terlebih dahulu.'
            }

        try:
            # ── Validasi input ──
            text = self._validate_input(text)

            # ── Preprocessing teks ──
            clean_text = preprocess_text(text, apply_stemming=True)

            if not clean_text.strip():
                return {
                    'label'     : 'ERROR',
                    'label_id'  : -1,
                    'confidence': 0.0,
                    'prob_fakta': 0.0,
                    'prob_hoaks': 0.0,
                    'clean_text': clean_text,
                    'emoji'     : '⚠️',
                    'color'     : 'orange',
                    'error'     : 'Teks tidak mengandung konten yang dapat dianalisis setelah preprocessing.'
                }

            # ── TF-IDF Transformasi ──
            X = self.tfidf.transform([clean_text])

            # ── Prediksi ──
            label_id = int(self.model.predict(X)[0])

            # ── Hitung Probabilitas ──
            try:
                proba = self.model.predict_proba(X)[0]
                prob_fakta = float(proba[0])
                prob_hoaks = float(proba[1])
                confidence = float(max(proba))
            except AttributeError:
                # Fallback jika model tidak mendukung predict_proba
                prob_fakta = 1.0 - label_id
                prob_hoaks = float(label_id)
                confidence = 1.0

            # ── Ambil info label ──
            label_info = LABEL_MAP.get(label_id, LABEL_MAP[0])

            return {
                'label'     : label_info['label'],
                'label_id'  : label_id,
                'confidence': confidence,
                'prob_fakta': prob_fakta,
                'prob_hoaks': prob_hoaks,
                'clean_text': clean_text,
                'emoji'     : label_info['emoji'],
                'color'     : label_info['color'],
                'error'     : None
            }

        except ValueError as ve:
            logger.warning(f"Validasi input gagal: {ve}")
            return {
                'label'     : 'ERROR',
                'label_id'  : -1,
                'confidence': 0.0,
                'prob_fakta': 0.0,
                'prob_hoaks': 0.0,
                'clean_text': '',
                'emoji'     : '⚠️',
                'color'     : 'orange',
                'error'     : str(ve)
            }
        except Exception as e:
            logger.error(f"Error saat prediksi: {e}")
            return {
                'label'     : 'ERROR',
                'label_id'  : -1,
                'confidence': 0.0,
                'prob_fakta': 0.0,
                'prob_hoaks': 0.0,
                'clean_text': '',
                'emoji'     : '❌',
                'color'     : 'gray',
                'error'     : f'Terjadi error internal: {str(e)}'
            }

    def predict_batch(self, texts: list) -> list:
        """
        Prediksi batch untuk beberapa teks sekaligus.

        Args:
            texts (list): List teks yang akan diperiksa.

        Returns:
            list: List dictionary hasil prediksi untuk setiap teks.
        """
        if not self.is_ready:
            return [self.predict("")]  # Kembalikan error

        results = []
        for i, text in enumerate(texts):
            result = self.predict(text)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Prediksi batch: {i+1}/{len(texts)} selesai.")
        return results

    def compare_models(self, text: str) -> dict:
        """
        Bandingkan prediksi dari kedua model (MNB vs SVM).

        Args:
            text (str): Teks yang akan diperiksa.

        Returns:
            dict: Perbandingan prediksi MNB dan SVM.
        """
        if not self.is_ready:
            return {'error': 'Model belum dimuat.'}

        results = {}

        try:
            clean_text = preprocess_text(text, apply_stemming=True)
            X = self.tfidf.transform([clean_text])

            for model_name, model in [('MNB', self.mnb_model), ('SVM', self.svm_model)]:
                if model is None:
                    results[model_name] = {'error': f'Model {model_name} tidak tersedia.'}
                    continue

                label_id = int(model.predict(X)[0])
                try:
                    proba      = model.predict_proba(X)[0]
                    prob_fakta = float(proba[0])
                    prob_hoaks = float(proba[1])
                    confidence = float(max(proba))
                except AttributeError:
                    prob_fakta = 1.0 - label_id
                    prob_hoaks = float(label_id)
                    confidence = 1.0

                label_info = LABEL_MAP.get(label_id, LABEL_MAP[0])
                results[model_name] = {
                    'label'     : label_info['label'],
                    'confidence': confidence,
                    'prob_fakta': prob_fakta,
                    'prob_hoaks': prob_hoaks,
                }

        except Exception as e:
            results['error'] = str(e)

        return results

    def get_model_info(self) -> dict:
        """
        Tampilkan informasi model yang sedang aktif.

        Returns:
            dict: Informasi model, vectorizer, dan status.
        """
        info = {
            'is_ready'      : self.is_ready,
            'model_type'    : self.model_type,
            'model_class'   : type(self.model).__name__ if self.model else 'N/A',
            'tfidf_vocab'   : len(self.tfidf.vocabulary_) if self.tfidf else 0,
            'tfidf_ngrams'  : str(self.tfidf.ngram_range) if self.tfidf else 'N/A',
            'tfidf_features': self.tfidf.max_features if self.tfidf else 0,
        }
        return info


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — Jalankan modul ini langsung untuk pengujian
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  DEMO: DETEKSI HOAKS POLITIK INDONESIA")
    print("=" * 65)

    # Inisialisasi detector
    detector = HoaxDetector()

    if not detector.is_ready:
        print("\n❌ Model belum tersedia. Jalankan 'python train.py' terlebih dahulu.")
        sys.exit(1)

    # Contoh teks untuk diuji
    test_cases = [
        "Presiden Joko Widodo resmi menandatangani Peraturan Presiden tentang digitalisasi layanan publik.",
        "BREAKING: Jokowi diam-diam menjual 3 pulau Indonesia kepada pengusaha asing tanpa sepengetahuan DPR!",
        "DPR RI mengesahkan Rancangan Undang-Undang Pemilihan Umum dengan 280 suara setuju.",
        "VIRAL!!! Calon presiden ketahuan korupsi 100 triliun, namun ditutup-tutupi oleh media mainstream!",
    ]

    print(f"\nInfo Model: {detector.get_model_info()}\n")
    print("-" * 65)

    for i, teks in enumerate(test_cases, 1):
        result = detector.predict(teks)
        print(f"\n[Uji Kasus {i}]")
        print(f"  Teks      : {teks[:70]}...")
        print(f"  Hasil     : {result['emoji']} {result['label']}")
        print(f"  Keyakinan : {result['confidence'] * 100:.2f}%")
        print(f"  P(Fakta)  : {result['prob_fakta'] * 100:.2f}%")
        print(f"  P(Hoaks)  : {result['prob_hoaks'] * 100:.2f}%")
        if result['error']:
            print(f"  Error     : {result['error']}")
        print("-" * 65)
