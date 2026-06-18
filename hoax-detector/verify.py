"""Script verifikasi akhir semua komponen proyek."""
import sys
import ast
import io

# Set stdout ke UTF-8 agar emoji tidak error di terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

print("=" * 60)
print("  VERIFIKASI AKHIR - HOAXRADAR PROJECT")
print("=" * 60)
print()

# ── 1. Cek semua dependensi ──
print("[ DEPENDENSI ]")
deps = {
    'streamlit'       : 'streamlit',
    'pandas'          : 'pandas',
    'numpy'           : 'numpy',
    'scikit-learn'    : 'sklearn',
    'imbalanced-learn': 'imblearn',
    'PySastrawi'      : 'Sastrawi',
    'plotly'          : 'plotly',
    'joblib'          : 'joblib',
    'seaborn'         : 'seaborn',
    'matplotlib'      : 'matplotlib',
}

all_ok = True
for display_name, mod_name in deps.items():
    try:
        m   = __import__(mod_name)
        ver = getattr(m, '__version__', 'OK')
        print(f"  [OK]   {display_name:<22} v{ver}")
    except ImportError as e:
        print(f"  [FAIL] {display_name:<22} GAGAL: {e}")
        all_ok = False

print()

# ── 2. Cek modul text_cleaner ──
print("[ MODUL PREPROCESSING ]")
try:
    from utils.text_cleaner import preprocess_text, SASTRAWI_AVAILABLE
    sample  = "Presiden Jokowi menandatangani undang-undang baru di Jakarta"
    cleaned = preprocess_text(sample)
    print(f"  [OK]   utils/text_cleaner.py berhasil dimuat")
    print(f"         PySastrawi aktif : {SASTRAWI_AVAILABLE}")
    print(f"         Input  : {sample[:50]}...")
    print(f"         Output : {cleaned}")
except Exception as e:
    print(f"  [FAIL] utils/text_cleaner.py: {e}")
    all_ok = False

print()

# ── 3. Cek syntax semua file Python ──
print("[ VALIDASI SYNTAX ]")
files = ['train.py', 'inference.py', 'app.py', 'utils/text_cleaner.py']
for fname in files:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        lines = source.count('\n')
        print(f"  [OK]   {fname:<30} ({lines:,} baris - syntax valid)")
    except SyntaxError as e:
        print(f"  [FAIL] {fname:<30} SYNTAX ERROR baris {e.lineno}: {e.msg}")
        all_ok = False
    except FileNotFoundError:
        print(f"  [WARN] {fname:<30} FILE TIDAK DITEMUKAN")
        all_ok = False

print()

# ── 4. Cek struktur direktori ──
print("[ STRUKTUR DIREKTORI ]")
from pathlib import Path

required = [
    'requirements.txt',
    'train.py',
    'inference.py',
    'app.py',
    'utils/text_cleaner.py',
    'utils/__init__.py',
    'data/README.txt',
    'model/README.txt',
    '.gitignore',
    'README.md',
]
for fpath in required:
    exists = Path(fpath).exists()
    icon   = "[OK]  " if exists else "[MISS]"
    size   = Path(fpath).stat().st_size if exists else 0
    suffix = f"({size:,} bytes)" if exists else "TIDAK DITEMUKAN"
    print(f"  {icon} {fpath:<35} {suffix}")
    if not exists:
        all_ok = False

print()
print("=" * 60)
if all_ok:
    print("  SEMUA KOMPONEN SIAP! Project production-ready.")
    print()
    print("  Langkah berikutnya:")
    print("  1. Letakkan CSV di  : data/indonesian_hoax_news.csv")
    print("  2. Training          : python train.py")
    print("  3. Jalankan app      : python -m streamlit run app.py")
else:
    print("  ADA KOMPONEN BERMASALAH - periksa log di atas.")
print("=" * 60)
