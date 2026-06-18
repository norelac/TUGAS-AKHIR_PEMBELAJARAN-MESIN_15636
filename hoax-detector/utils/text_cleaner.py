"""
Modul Pembersihan Teks untuk Deteksi Hoaks Politik Indonesia
============================================================
Modul ini berisi fungsi-fungsi preprocessing teks berbahasa Indonesia
menggunakan library PySastrawi untuk stemming.

Author  : Indonesian Hoax Detection System
Version : 1.0.0
"""

import re
import string
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STOPWORD & STEMMER — Inisialisasi dengan penanganan error
# ─────────────────────────────────────────────────────────────────────────────

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    # Buat objek factory stopword dan stemmer
    _stop_factory  = StopWordRemoverFactory()
    _stem_factory  = StemmerFactory()

    # Buat remover dan stemmer (berat, buat sekali saja)
    _stopword_remover = _stop_factory.create_stop_word_remover()
    _stemmer          = _stem_factory.create_stemmer()

    SASTRAWI_AVAILABLE = True
    logger.info("PySastrawi berhasil dimuat.")

except ImportError:
    SASTRAWI_AVAILABLE = False
    _stopword_remover  = None
    _stemmer           = None
    logger.warning("PySastrawi tidak tersedia. Stemming dan stopword removal dinonaktifkan.")


# ─────────────────────────────────────────────────────────────────────────────
# STOPWORD TAMBAHAN — Kata-kata umum yang tidak relevan dalam konteks politik
# ─────────────────────────────────────────────────────────────────────────────

ADDITIONAL_STOPWORDS = {
    'yg', 'dgn', 'utk', 'tdk', 'sdh', 'blm', 'jg', 'lg', 'sy', 'krn',
    'kpd', 'pd', 'dg', 'bs', 'tp', 'ttg', 'tsb', 'dr', 'sbg', 'dlm',
    'thd', 'dll', 'dst', 'spt', 'hrs', 'bisa', 'itu', 'ini', 'ada',
    'juga', 'yang', 'untuk', 'dengan', 'tidak', 'dari', 'sudah', 'akan',
    'oleh', 'kepada', 'dalam', 'bahwa', 'pada', 'lebih', 'seperti',
    'http', 'https', 'www', 'com', 'co', 'id', 'net', 'org', 'html',
    'pic', 'twitter', 'facebook', 'instagram', 'youtube', 'wa', 'whatsapp'
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def case_folding(text: str) -> str:
    """
    Case folding: ubah semua teks menjadi huruf kecil (lowercase).

    Args:
        text (str): Teks input mentah.

    Returns:
        str: Teks dalam huruf kecil.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def remove_urls(text: str) -> str:
    """
    Hapus semua URL dari teks (http, https, www).

    Args:
        text (str): Teks input.

    Returns:
        str: Teks tanpa URL.
    """
    # Pola URL lengkap (http/https/www)
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'|www\.[^\s]+'
    )
    return url_pattern.sub(' ', text)


def remove_mentions_hashtags(text: str) -> str:
    """
    Hapus mention (@user) dan hashtag (#tag) dari teks media sosial.

    Args:
        text (str): Teks input.

    Returns:
        str: Teks tanpa mention dan hashtag.
    """
    text = re.sub(r'@\w+', ' ', text)   # hapus @mention
    text = re.sub(r'#\w+', ' ', text)   # hapus #hashtag
    return text


def remove_numbers(text: str) -> str:
    """
    Hapus angka dan karakter numerik dari teks.

    Args:
        text (str): Teks input.

    Returns:
        str: Teks tanpa angka.
    """
    return re.sub(r'\d+', ' ', text)


def remove_punctuation(text: str) -> str:
    """
    Hapus semua tanda baca dan simbol dari teks.

    Args:
        text (str): Teks input.

    Returns:
        str: Teks tanpa tanda baca.
    """
    # Hapus tanda baca standar
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus karakter non-ASCII dan simbol khusus
    text = re.sub(r'[^\w\s]', ' ', text)
    # Hapus underscore yang tersisa
    text = re.sub(r'_', ' ', text)
    return text


def remove_extra_whitespace(text: str) -> str:
    """
    Hapus spasi berlebih (multiple spaces, tab, newline).

    Args:
        text (str): Teks input.

    Returns:
        str: Teks dengan satu spasi antar kata.
    """
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    """
    Hapus stopword bahasa Indonesia menggunakan PySastrawi
    dan tambahan daftar stopword kustom.

    Args:
        text (str): Teks input (sudah lowercase).

    Returns:
        str: Teks tanpa stopword.
    """
    if SASTRAWI_AVAILABLE and _stopword_remover:
        text = _stopword_remover.remove(text)

    # Hapus juga stopword tambahan (kata slang & singkatan)
    words = text.split()
    words = [w for w in words if w not in ADDITIONAL_STOPWORDS and len(w) > 1]
    return ' '.join(words)


def stem_text(text: str) -> str:
    """
    Lakukan stemming pada teks menggunakan PySastrawi.
    Mengubah kata berimbuhan menjadi kata dasar.

    Contoh: "menjalankan" → "jalan", "pemberontakan" → "berontak"

    Args:
        text (str): Teks input.

    Returns:
        str: Teks setelah stemming.
    """
    if SASTRAWI_AVAILABLE and _stemmer:
        return _stemmer.stem(text)
    # Fallback: kembalikan teks apa adanya
    return text


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI UTAMA — Pipeline Lengkap
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str, apply_stemming: bool = True) -> str:
    """
    Pipeline preprocessing teks lengkap untuk deteksi hoaks.

    Urutan langkah:
        1. Validasi input
        2. Case folding (lowercase)
        3. Hapus URL
        4. Hapus mention & hashtag
        5. Hapus angka
        6. Hapus tanda baca & simbol
        7. Hapus spasi berlebih
        8. Hapus stopword
        9. Stemming (opsional)
        10. Hapus spasi berlebih (final cleanup)

    Args:
        text (str)            : Teks input mentah.
        apply_stemming (bool) : Aktifkan stemming (default: True).

    Returns:
        str: Teks bersih hasil preprocessing.
    """
    try:
        # Langkah 1: Validasi input
        if not isinstance(text, str) or not text.strip():
            return ""

        # Langkah 2: Case folding
        text = case_folding(text)

        # Langkah 3: Hapus URL
        text = remove_urls(text)

        # Langkah 4: Hapus mention dan hashtag
        text = remove_mentions_hashtags(text)

        # Langkah 5: Hapus angka
        text = remove_numbers(text)

        # Langkah 6: Hapus tanda baca dan simbol
        text = remove_punctuation(text)

        # Langkah 7: Normalisasi spasi
        text = remove_extra_whitespace(text)

        # Langkah 8: Hapus stopword
        text = remove_stopwords(text)

        # Langkah 9: Stemming (jika diaktifkan)
        if apply_stemming:
            text = stem_text(text)

        # Langkah 10: Normalisasi spasi final
        text = remove_extra_whitespace(text)

        return text

    except Exception as e:
        logger.error(f"Error saat preprocessing teks: {e}")
        return ""


def preprocess_batch(texts: list, apply_stemming: bool = True) -> list:
    """
    Preprocessing batch untuk kolom DataFrame (lebih efisien).

    Args:
        texts (list)          : List teks yang akan diproses.
        apply_stemming (bool) : Aktifkan stemming.

    Returns:
        list: List teks yang sudah bersih.
    """
    processed = []
    total = len(texts)

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"Preprocessing: {i + 1}/{total} teks selesai diproses.")
        processed.append(preprocess_text(text, apply_stemming=apply_stemming))

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TEST SEDERHANA — Jalankan modul ini langsung untuk verifikasi
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Contoh berita hoaks dan fakta untuk uji coba
    contoh_teks = [
        "HOAKS!!! Jokowi menjual Pulau Natuna kepada China. Sumber: https://bohong.com/berita123",
        "Pemerintah telah mengesahkan RUU Pemilu pada sidang paripurna DPR RI yang berlangsung Kamis.",
        "@infobola #pilpres2024 Prabowo mengatakan akan mundur dari pencalonan karena tekanan asing!!!",
        "Virus Covid-19 varian baru menyerang 1000 orang di Jakarta hari ini, pemerintah diam saja!",
    ]

    print("=" * 70)
    print("UJI COBA MODUL PREPROCESSING TEKS")
    print("=" * 70)

    for i, teks in enumerate(contoh_teks):
        hasil = preprocess_text(teks)
        print(f"\n[Teks {i+1}]")
        print(f"  Asli  : {teks[:80]}...")
        print(f"  Bersih: {hasil}")

    print("\n" + "=" * 70)
    print(f"PySastrawi tersedia: {SASTRAWI_AVAILABLE}")
    print("Modul text_cleaner.py berhasil diuji!")
    print("=" * 70)
