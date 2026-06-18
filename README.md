# 🛡️ HoaxRadar — Sistem Deteksi Hoaks Politik Indonesia

> **Tugas Akhir Pembelajaran Mesin** — Klasifikasi berita Hoaks vs Fakta berbasis Machine Learning  
> menggunakan TF-IDF, SMOTE, Naive Bayes, dan SVM dengan antarmuka Streamlit.

---

## 📋 Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Direktori](#struktur-direktori)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Pipeline ML](#pipeline-ml)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Gambaran Umum

**HoaxRadar** adalah sistem klasifikasi teks berbahasa Indonesia yang mampu mendeteksi
apakah sebuah berita politik merupakan **HOAKS** (1) atau **FAKTA** (0).

### Teknologi Utama
| Komponen | Teknologi |
|----------|-----------|
| Preprocessing | Case Folding, Regex Cleaning, PySastrawi Stopword + Stemming |
| Vectorization | TF-IDF (max_features=10000, ngram=(1,2)) |
| Imbalance | SMOTE (hanya pada training data) |
| Model 1 | Multinomial Naive Bayes + GridSearchCV |
| Model 2 | SVM Linear Kernel + GridSearchCV |
| Tuning | 5-Fold Stratified Cross Validation |
| Deployment | Streamlit Web App |

---

## 🏗️ Arsitektur Sistem

```
Input Teks Mentah
      │
      ▼
┌─────────────────────────────┐
│   Text Preprocessing        │
│   (utils/text_cleaner.py)   │
│   1. Case Folding           │
│   2. Hapus URL & Simbol     │
│   3. Hapus Stopword         │
│   4. Stemming (PySastrawi)  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   TF-IDF Vectorizer         │
│   max_features = 10,000     │
│   ngram_range  = (1, 2)     │
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌────────┐       ┌─────────┐
│  MNB   │       │   SVM   │
│(alpha) │       │   (C)   │
└────┬───┘       └────┬────┘
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Best Model     │
    │  (F1-Score)     │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Streamlit UI   │
    │  (app.py)       │
    └─────────────────┘
```

---

## 📁 Struktur Direktori

```
hoax-detector/
│
├── data/
│   ├── README.txt                    ← Panduan format dataset
│   └── indonesian_hoax_news.csv      ← Dataset (disediakan user)
│
├── model/
│   ├── best_model.pkl                ← Model terbaik (auto-generated)
│   ├── mnb_model.pkl                 ← Model MNB (auto-generated)
│   ├── svm_model.pkl                 ← Model SVM (auto-generated)
│   ├── tfidf_vectorizer.pkl          ← Vectorizer (auto-generated)
│   ├── training_metadata.json        ← Metrik & info training
│   └── reports/
│       ├── data_distribution.png
│       ├── confusion_matrix_*.png
│       ├── model_comparison.csv
│       └── sample_preprocessing.csv
│
├── utils/
│   ├── __init__.py
│   └── text_cleaner.py               ← Modul preprocessing teks
│
├── train.py                          ← Pipeline training lengkap
├── inference.py                      ← Kelas HoaxDetector
├── app.py                            ← Aplikasi Streamlit
├── requirements.txt                  ← Dependensi Python
├── .gitignore
└── README.md
```

---

## ⚙️ Persyaratan Sistem

- **Python**: 3.9 atau lebih baru
- **RAM**: Minimal 4 GB (direkomendasikan 8 GB untuk stemming batch besar)
- **OS**: Windows / Linux / macOS

---

## 🚀 Instalasi

### 1. Clone / Download project

```bash
cd "d:\SEM 4\TUGAS AKHIR PEMBELAJARAN MESIN\hoax-detector"
```

### 2. (Opsional) Buat Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# atau
source venv/bin/activate # Linux/Mac
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
# atau jika pip tidak dikenali:
python -m pip install -r requirements.txt
```

---

## 💻 Penggunaan

### Langkah 1 — Siapkan Dataset

Letakkan file dataset di folder `data/` dengan nama:
```
data/indonesian_hoax_news.csv
```

Format CSV minimal:
```csv
text,label
"Presiden menandatangani UU baru di Jakarta.",0
"VIRAL!!! Jokowi jual pulau ke China!!!",1
```

> Kolom yang didukung: lihat `data/README.txt`

---

### Langkah 2 — Training Model

```bash
python train.py
```

> 💡 **Windows Store Python**: Jika `streamlit` tidak dikenali langsung, gunakan `python -m streamlit run app.py`

Output yang dihasilkan:
- `model/best_model.pkl` — Model terbaik
- `model/tfidf_vectorizer.pkl` — Vectorizer
- `model/training_metadata.json` — Metrik evaluasi
- `model/reports/` — Grafik confusion matrix & distribusi data

---

### Langkah 3 — Jalankan Aplikasi Web

```bash
# Metode 1 (direkomendasikan untuk Windows):
python -m streamlit run app.py

# Metode 2 (jika streamlit ada di PATH):
streamlit run app.py
```

Buka browser dan akses: **http://localhost:8501**

---

### (Opsional) Uji Inferensi via Terminal

```bash
python inference.py
```

Atau gunakan dalam kode Python:

```python
from inference import HoaxDetector

detector = HoaxDetector()

result = detector.predict("Jokowi diam-diam menjual aset negara kepada asing!")
print(result['label'])      # 'HOAKS' atau 'FAKTA'
print(result['confidence']) # e.g., 0.9342
print(result['prob_hoaks']) # e.g., 0.9342
print(result['prob_fakta']) # e.g., 0.0658
```

---

## 🔬 Pipeline ML

### Preprocessing Teks

```
Input  : "VIRAL!!! Jokowi jual Pulau Natuna ke China! https://berita.com"
   ↓ Case Folding
         "viral!!! jokowi jual pulau natuna ke china! https://berita.com"
   ↓ Hapus URL
         "viral!!! jokowi jual pulau natuna ke china!"
   ↓ Hapus Simbol & Angka
         "viral jokowi jual pulau natuna ke china"
   ↓ Hapus Stopword
         "viral jokowi jual pulau natuna china"
   ↓ Stemming
         "viral jokowi jual pulau natuna china"
Output : "viral jokowi jual pulau natuna china"
```

### Hyperparameter yang Di-tuning

| Model | Parameter | Grid Search Values |
|-------|-----------|-------------------|
| Multinomial NB | `alpha` (Laplace Smoothing) | [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] |
| SVM Linear | `C` (Regularization) | [0.01, 0.1, 1.0, 5.0, 10.0] |

### Metrik Evaluasi

- **Accuracy** — Persentase prediksi benar keseluruhan
- **Precision** — Ketepatan prediksi Hoaks (menghindari false alarm)
- **Recall** — Kemampuan mendeteksi semua Hoaks yang sebenarnya
- **F1-Score** — Harmonik mean Precision dan Recall (metrik utama)
- **ROC-AUC** — Area Under Curve (mengukur kemampuan diskriminasi)

---

## 📊 Hasil Evaluasi

*(Diisi otomatis setelah `python train.py` selesai)*

Lihat file: `model/reports/model_comparison.csv`

---

## 🔧 Troubleshooting

### ❌ `FileNotFoundError: indonesian_hoax_news.csv`
```
Pastikan file dataset ada di: data/indonesian_hoax_news.csv
```

### ❌ `PySastrawi tidak tersedia`
```bash
pip install PySastrawi
```

### ❌ `Model belum dimuat` di Streamlit
```bash
# Jalankan training terlebih dahulu
python train.py
# Kemudian jalankan app
streamlit run app.py
```

### ❌ SMOTE error karena sampel terlalu sedikit
```
Pastikan dataset memiliki minimal 10 sampel per kelas.
Jika dataset sangat kecil, SMOTE akan dinonaktifkan otomatis.
```

### ❌ `UnicodeDecodeError` saat membaca CSV
```
Script secara otomatis mencoba encoding: utf-8, latin-1, iso-8859-1
Jika masih gagal, konversi file CSV ke UTF-8 menggunakan Notepad++
atau Excel (Save As → CSV UTF-8).
```

---

## 👨‍🎓 Informasi Akademik

| | |
|---|---|
| **Mata Kuliah** | Pembelajaran Mesin |
| **Semester** | 4 |
| **Topik TA** | Indonesian Political Hoax Detection |
| **Metode** | TF-IDF + SMOTE + MNB/SVM + GridSearchCV |
| **Framework** | Streamlit (Web UI) |

---

*Dibuat dengan ❤️ menggunakan Python & Streamlit*
