# Dataset: Indonesian Political Hoax News
# ========================================
# 
# Tempatkan file dataset di direktori ini dengan nama:
#   indonesian_hoax_news.csv
#
# Format kolom yang didukung:
# ┌──────────────────────────────────────────────────────────┐
# │  Kolom Teks (salah satu):                                │
# │    - text, content, berita, judul, artikel, narasi       │
# │                                                          │
# │  Kolom Label (salah satu):                               │
# │    - label, hoax, class, kategori, target, status        │
# │                                                          │
# │  Nilai Label:                                            │
# │    - 0 atau 'fakta' atau 'fact'  = Berita Fakta          │
# │    - 1 atau 'hoax'  atau 'hoaks' = Berita Hoaks          │
# └──────────────────────────────────────────────────────────┘
#
# Contoh struktur CSV:
#   text,label
#   "Presiden menandatangani UU baru...",0
#   "VIRAL!!! Jokowi menjual pulau...",1
#
# Sumber dataset yang disarankan:
#   - Kaggle: Indonesian Fake News Dataset
#   - KOMINFO Fact Check Dataset
#   - Turnbackhoax.id crawled dataset
