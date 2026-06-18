"""
Script Persiapan Dataset: Penggabungan Data Kominfo (Hoaks) & Detik (Fakta)
========================================================================
Script ini digunakan untuk Opsi B:
  1. Memuat dataset Hoaks dari Kominfo (baik format CSV maupun JSON).
  2. Membersihkan penanda seperti "[HOAKS]", "[HOAX]" dari judul agar model tidak bocor.
  3. Mengunduh dataset berita Fakta (judul berita Detik.com) dari repositori publik.
  4. Menggabungkan kedua data secara seimbang (balanced class).
  5. Menyimpan hasilnya ke 'data/indonesian_hoax_news.csv' untuk training.

Cara Penggunaan:
  1. Letakkan file 'komdigi_hoaks.csv' ATAU 'komdigi_hoaks.json' di folder 'data/'.
  2. Jalankan script ini:
     python prepare_dataset.py
"""

import os
import re
import urllib.request
import pandas as pd
from pathlib import Path

# Path Konfigurasi
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FACT_URL = "https://raw.githubusercontent.com/ibamibrahim/dataset-judul-berita-indonesia/master/detik_news_title.csv"

def clean_hoax_title(title):
    """
    Membersihkan penanda seperti [HOAKS], (HOAX) di awal judul
    agar model tidak mengalami kebocoran data (data leakage).
    """
    if not isinstance(title, str):
        return ""
    # Hapus tag di awal seperti [HOAKS], [HOAX], [FITNAH], [DISINFORMASI], [SALAH], dll.
    cleaned = re.sub(r'^\[.*?\]\s*', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\(.*?\)\s*', '', cleaned, flags=re.IGNORECASE)
    # Hapus kata HOAKS/HOAX di awal teks
    cleaned = re.sub(r'^(hoaks|hoax|fitnah|disinformasi|salah)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def prepare():
    print("=" * 60)
    print("  MEMPERSIAPKAN DATASET GABUNGAN (OPSI B)")
    print("=" * 60)

    # 1. Deteksi file Hoaks Kominfo
    hoax_csv_path = DATA_DIR / "komdigi_hoaks.csv"
    hoax_json_path = DATA_DIR / "komdigi_hoaks.json"
    
    df_hoax = None
    
    if hoax_csv_path.exists():
        print(f"[*] Menemukan file hoaks CSV: {hoax_csv_path.name}")
        try:
            df_hoax = pd.read_csv(hoax_csv_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"[!] Gagal membaca CSV dengan utf-8-sig, mencoba latin-1: {e}")
            df_hoax = pd.read_csv(hoax_csv_path, encoding='latin-1')
    elif hoax_json_path.exists():
        print(f"[*] Menemukan file hoaks JSON: {hoax_json_path.name}")
        df_hoax = pd.read_json(hoax_json_path, encoding='utf-8')
    else:
        print("[ERROR] File 'komdigi_hoaks.csv' atau 'komdigi_hoaks.json' TIDAK ditemukan di folder 'data/'.")
        print("\nSilakan unduh dataset tersebut dari Kaggle dan pindahkan ke folder:")
        print(f"  {DATA_DIR.resolve()}\n")
        return

    print(f"[*] Berhasil memuat {len(df_hoax)} data hoaks asli.")

    # Ekstrak kolom teks hoaks (menggunakan 'title' atau 'title' + 'body')
    # Di dataset Kominfo/Komdigi, biasanya ada kolom 'title'
    if 'title' not in df_hoax.columns:
        # Cari kolom alternatif
        title_col = [c for c in df_hoax.columns if 'title' in c or 'judul' in c]
        if title_col:
            df_hoax['text'] = df_hoax[title_col[0]]
        else:
            print("[ERROR] Kolom judul ('title'/'judul') tidak ditemukan di file hoaks Anda.")
            print(f"Kolom yang tersedia: {list(df_hoax.columns)}")
            return
    else:
        df_hoax['text'] = df_hoax['title']

    # Bersihkan penanda hoaks dari teks
    print("[*] Membersihkan tag '[HOAKS]' dari judul berita hoaks...")
    df_hoax['text'] = df_hoax['text'].apply(clean_hoax_title)
    
    # Ambil baris yang tidak kosong
    df_hoax = df_hoax[df_hoax['text'].str.strip() != ''].copy()
    df_hoax['label'] = 1  # Label 1 = Hoaks
    df_hoax = df_hoax[['text', 'label']]
    
    num_hoax = len(df_hoax)
    print(f"[*] Total data hoaks bersih: {num_hoax}")

    # 2. Unduh data berita Fakta (Detik.com)
    fact_csv_temp = DATA_DIR / "detik_news_title.csv"
    if not fact_csv_temp.exists():
        print(f"[*] Mengunduh dataset berita fakta dari GitHub...")
        print(f"    Source: {FACT_URL}")
        try:
            # Gunakan user-agent agar tidak diblokir github
            req = urllib.request.Request(
                FACT_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(fact_csv_temp, 'wb') as out_file:
                out_file.write(response.read())
            print("[*] Unduhan selesai.")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh data fakta dari GitHub: {e}")
            print("Pastikan koneksi internet aktif.")
            return
    else:
        print("[*] Menggunakan data fakta lokal yang sudah ada.")

    # Memuat data fakta
    print("[*] Membaca dataset berita fakta...")
    try:
        df_fact = pd.read_csv(fact_csv_temp, encoding='utf-8')
    except Exception:
        df_fact = pd.read_csv(fact_csv_temp, encoding='latin-1')

    # Cari kolom judul di detik dataset
    # Kolom biasanya adalah 'Title' atau 'title'
    title_col_fact = None
    for col in ['Title', 'title', 'judul', 'Judul']:
        if col in df_fact.columns:
            title_col_fact = col
            break
            
    if title_col_fact is None:
        title_col_fact = df_fact.columns[2]  # fallback ke kolom ketiga
        print(f"[!] Kolom judul fakta tidak pasti. Menggunakan kolom: '{title_col_fact}'")

    df_fact['text'] = df_fact[title_col_fact]
    df_fact = df_fact[df_fact['text'].str.strip() != ''].copy()
    df_fact['label'] = 0  # Label 0 = Fakta
    df_fact = df_fact[['text', 'label']]
    
    # 3. Balancing dataset
    print(f"[*] Jumlah total data fakta yang tersedia: {len(df_fact)}")
    
    # Lakukan sampling agar jumlah data Fakta seimbang dengan Hoaks
    if len(df_fact) > num_hoax:
        print(f"[*] Melakukan random sampling {num_hoax} data fakta agar seimbang.")
        df_fact_sampled = df_fact.sample(n=num_hoax, random_state=42)
    else:
        print(f"[!] Data fakta ({len(df_fact)}) lebih sedikit dibanding data hoaks ({num_hoax}).")
        df_fact_sampled = df_fact
        # Lakukan sampling hoaks agar seimbang dengan data fakta
        df_hoax = df_hoax.sample(n=len(df_fact_sampled), random_state=42)

    # 4. Gabungkan dan Acak (Shuffle)
    print("[*] Menggabungkan dataset hoaks dan fakta...")
    df_combined = pd.concat([df_hoax, df_fact_sampled], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Simpan dataset akhir
    output_path = DATA_DIR / "indonesian_hoax_news.csv"
    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] DATASET GABUNGAN BERHASIL DISIMPAN!")
    print(f"    Lokasi: {output_path.resolve()}")
    print(f"    Ukuran: {len(df_combined)} baris (Fakta: {sum(df_combined['label'] == 0)}, Hoaks: {sum(df_combined['label'] == 1)})")
    print("\nContoh beberapa baris data gabungan:")
    print("-" * 60)
    for idx, row in df_combined.head(5).iterrows():
        label_str = "Hoaks" if row['label'] == 1 else "Fakta"
        print(f"[{label_str}] {row['text'][:80]}...")
    print("-" * 60)
    print("\nSekarang Anda dapat menjalankan program training menggunakan perintah:")
    print("  python train.py")
    print("=" * 60)

if __name__ == "__main__":
    prepare()
