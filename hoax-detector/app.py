"""
Aplikasi Web: Deteksi Hoaks Politik Indonesia
=============================================
Antarmuka pengguna berbasis Streamlit untuk mendeteksi
hoaks pada berita politik Indonesia.

Cara menjalankan:
    streamlit run app.py

Author  : Indonesian Hoax Detection System
Version : 1.0.0
"""

import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Tambahkan direktori root ke path
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN — Harus dipanggil PERTAMA sebelum import Streamlit lain
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HoaxRadar — Deteksi Hoaks Politik Indonesia",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help'    : None,
        'Report a bug': None,
        'About'       : "**HoaxRadar** — Sistem Deteksi Hoaks Politik Indonesia berbasis Machine Learning."
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS KUSTOM — Desain Modern & Premium
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Font Google ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background Utama ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 35%, #1a1035 70%, #0a0e1a 100%);
    min-height: 100vh;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }

/* ── Header Hero ── */
.hero-container {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.15) 0%,
        rgba(139,92,246,0.10) 50%,
        rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 24px;
    padding: 48px 40px 36px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center,
        rgba(99,102,241,0.08) 0%, transparent 60%);
    animation: pulse 6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%       { transform: scale(1.05); opacity: 0.7; }
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    position: relative;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 100px;
    padding: 4px 16px;
    font-size: 0.78rem;
    color: #a5b4fc;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 16px;
    position: relative;
}

/* ── Card Glassmorphism ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.3);
    transform: translateY(-2px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}

/* ── Hasil Deteksi: HOAKS ── */
.result-hoax {
    background: linear-gradient(135deg,
        rgba(239,68,68,0.12) 0%,
        rgba(185,28,28,0.08) 100%);
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: slideInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 40px rgba(239,68,68,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.result-hoax-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #f87171;
    letter-spacing: 0.05em;
    text-shadow: 0 0 30px rgba(248,113,113,0.5);
}

/* ── Hasil Deteksi: FAKTA ── */
.result-fact {
    background: linear-gradient(135deg,
        rgba(34,197,94,0.12) 0%,
        rgba(21,128,61,0.08) 100%);
    border: 2px solid rgba(34,197,94,0.5);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: slideInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 40px rgba(34,197,94,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.result-fact-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #4ade80;
    letter-spacing: 0.05em;
    text-shadow: 0 0 30px rgba(74,222,128,0.5);
}

/* ── Animasi ── */
@keyframes slideInUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 20px;
}
.metric-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e2e8f0;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-weight: 600;
}

/* ── Section Title ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Info Tag ── */
.info-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #a5b4fc;
    font-weight: 500;
    margin: 4px;
}

/* ── Textarea & Tombol ── */
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease;
}
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Sembunyikan elemen Streamlit default ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── History Table ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #94a3b8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_detector():
    """
    Muat HoaxDetector dengan caching agar tidak reload setiap kali.
    Fungsi ini hanya dipanggil sekali selama sesi aktif.
    """
    try:
        from inference import HoaxDetector
        detector = HoaxDetector()
        return detector
    except Exception as e:
        return None


def load_training_metadata() -> dict:
    """Muat metadata training jika tersedia."""
    meta_path = Path(__file__).parent / 'model' / 'training_metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def create_confidence_gauge(confidence: float, label: str) -> go.Figure:
    """
    Buat gauge chart untuk menampilkan tingkat keyakinan model.

    Args:
        confidence (float): Nilai confidence [0.0 - 1.0].
        label (str)       : Label prediksi ('HOAKS' atau 'FAKTA').

    Returns:
        go.Figure: Plotly gauge chart.
    """
    color = "#f87171" if label == "HOAKS" else "#4ade80"
    bg_color = "rgba(248,113,113,0.1)" if label == "HOAKS" else "rgba(74,222,128,0.1)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={
            'suffix': '%',
            'font': {'size': 42, 'color': color, 'family': 'Inter'},
        },
        gauge={
            'axis': {
                'range'    : [0, 100],
                'tickwidth': 1,
                'tickcolor': '#334155',
                'tickfont' : {'color': '#64748b', 'size': 11},
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor'   : 'rgba(255,255,255,0.03)',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  40], 'color': 'rgba(255,255,255,0.04)'},
                {'range': [40, 70], 'color': 'rgba(255,255,255,0.06)'},
                {'range': [70, 100],'color': 'rgba(255,255,255,0.08)'},
            ],
            'threshold': {
                'line' : {'color': color, 'width': 3},
                'thickness': 0.75,
                'value': confidence * 100,
            }
        },
        title={'text': "Tingkat Keyakinan", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'Inter'}},
        domain={'x': [0, 1], 'y': [0, 1]},
    ))

    fig.update_layout(
        height=240,
        margin={'t': 40, 'b': 20, 'l': 20, 'r': 20},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor ='rgba(0,0,0,0)',
        font_family  ='Inter',
    )
    return fig


def create_probability_bar(prob_fakta: float, prob_hoaks: float) -> go.Figure:
    """
    Buat horizontal bar chart untuk membandingkan probabilitas.

    Args:
        prob_fakta (float): Probabilitas kelas Fakta.
        prob_hoaks (float): Probabilitas kelas Hoaks.

    Returns:
        go.Figure: Plotly bar chart.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Fakta',
        x=[prob_fakta * 100],
        y=['Probabilitas'],
        orientation='h',
        marker=dict(color='#4ade80', line=dict(width=0)),
        text=f"{prob_fakta*100:.1f}%",
        textposition='inside',
        textfont=dict(color='white', size=13, family='Inter'),
        insidetextanchor='middle',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Bar(
        name='Hoaks',
        x=[prob_hoaks * 100],
        y=['Probabilitas'],
        orientation='h',
        marker=dict(color='#f87171', line=dict(width=0)),
        text=f"{prob_hoaks*100:.1f}%",
        textposition='inside',
        textfont=dict(color='white', size=13, family='Inter'),
        insidetextanchor='middle',
        hoverinfo='skip',
    ))

    fig.update_layout(
        barmode='stack',
        height=90,
        margin={'t': 10, 'b': 10, 'l': 10, 'r': 10},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor ='rgba(0,0,0,0)',
        showlegend   =True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.0,
            xanchor='right',
            x=1.0,
            font=dict(color='#94a3b8', family='Inter', size=12),
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100], zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def display_result(result: dict):
    """
    Tampilkan hasil prediksi dengan desain yang sesuai label.

    Args:
        result (dict): Hasil dari HoaxDetector.predict().
    """
    label      = result.get('label', 'ERROR')
    confidence = result.get('confidence', 0.0)
    prob_fakta = result.get('prob_fakta', 0.0)
    prob_hoaks = result.get('prob_hoaks', 0.0)
    emoji      = result.get('emoji', '❓')
    clean_text = result.get('clean_text', '')

    if label == 'ERROR':
        st.error(f"⚠️ {result.get('error', 'Terjadi kesalahan.')}")
        return

    # ── Kartu Hasil Utama ──
    if label == 'HOAKS':
        st.markdown(f"""
        <div class="result-hoax">
            <div style="font-size:3.5rem; margin-bottom:8px;">{emoji}</div>
            <div class="result-hoax-title">HOAKS TERDETEKSI</div>
            <div style="color:#fca5a5; font-size:0.9rem; margin-top:8px; font-weight:500;">
                Konten ini memiliki indikasi kuat sebagai informasi palsu
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fact">
            <div style="font-size:3.5rem; margin-bottom:8px;">{emoji}</div>
            <div class="result-fact-title">FAKTA TERVERIFIKASI</div>
            <div style="color:#86efac; font-size:0.9rem; margin-top:8px; font-weight:500;">
                Konten ini terindikasi sebagai informasi yang valid dan faktual
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge dan Probabilitas ──
    col_gauge, col_bar = st.columns([1, 1])

    with col_gauge:
        fig_gauge = create_confidence_gauge(confidence, label)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_bar:
        st.markdown("""
        <div style="padding-top:20px;">
            <div class="section-title">📊 Distribusi Probabilitas</div>
        </div>
        """, unsafe_allow_html=True)
        fig_bar = create_probability_bar(prob_fakta, prob_hoaks)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        # Detail angka
        col_f, col_h = st.columns(2)
        with col_f:
            st.markdown(f"""
            <div class="metric-item" style="border-color:rgba(74,222,128,0.3);">
                <div class="metric-value" style="color:#4ade80;">{prob_fakta*100:.1f}%</div>
                <div class="metric-label">✅ Fakta</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h:
            st.markdown(f"""
            <div class="metric-item" style="border-color:rgba(248,113,113,0.3);">
                <div class="metric-value" style="color:#f87171;">{prob_hoaks*100:.1f}%</div>
                <div class="metric-label">🚨 Hoaks</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Detail Preprocessing ──
    with st.expander("🔍 Lihat Detail Preprocessing Teks", expanded=False):
        st.markdown("""
        <div class="section-title">📝 Hasil Pembersihan Teks</div>
        """, unsafe_allow_html=True)
        st.code(clean_text if clean_text else "(teks kosong setelah preprocessing)", language=None)
        st.caption("Teks di atas adalah hasil setelah case folding, pembersihan URL/simbol, stopword removal, dan stemming.")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(detector, metadata: dict):
    """Render sidebar dengan informasi sistem dan navigasi."""
    with st.sidebar:
        # Logo / Branding
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 24px;">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-size:1.1rem; font-weight:800; color:#e2e8f0; margin-top:8px;">HoaxRadar</div>
            <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">
                v1.0.0 · ML Edition
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Status Sistem ──
        st.markdown('<div class="section-title">⚙️ Status Sistem</div>', unsafe_allow_html=True)

        if detector and detector.is_ready:
            st.success("✅ Model Aktif & Siap")
            model_info = detector.get_model_info()

            st.markdown(f"""
            <div style="margin-top:12px;">
                <div class="info-tag">🤖 {model_info.get('model_class', 'N/A')}</div>
                <div class="info-tag">📚 {model_info.get('tfidf_vocab', 0):,} Fitur</div>
                <div class="info-tag">📏 n-gram {model_info.get('tfidf_ngrams', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Model Belum Dimuat")
            st.markdown("""
            <div style="color:#94a3b8; font-size:0.85rem; margin-top:8px;">
                Jalankan <code>python train.py</code> terlebih dahulu untuk melatih model.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Metadata Training ──
        if metadata:
            st.markdown('<div class="section-title">📈 Info Training</div>', unsafe_allow_html=True)
            best_model = metadata.get('best_model', 'N/A')
            duration   = metadata.get('training_duration', 'N/A')
            n_train    = metadata.get('training_samples', 0)
            n_test     = metadata.get('test_samples', 0)

            st.markdown(f"""
            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.8;">
                🏆 <b style="color:#a5b4fc;">Model Terbaik</b><br>
                &nbsp;&nbsp;&nbsp;{best_model}<br><br>
                📊 <b style="color:#a5b4fc;">Dataset</b><br>
                &nbsp;&nbsp;&nbsp;Train: {n_train:,} | Test: {n_test:,}<br><br>
                ⏱️ <b style="color:#a5b4fc;">Durasi Training</b><br>
                &nbsp;&nbsp;&nbsp;{duration}
            </div>
            """, unsafe_allow_html=True)

            # Metrik evaluasi
            metrics = metadata.get('metrics', {})
            if metrics:
                st.markdown("<br>", unsafe_allow_html=True)
                for model_name, m in metrics.items():
                    with st.expander(f"📊 {model_name}", expanded=False):
                        cols = st.columns(2)
                        metric_items = [
                            ("Accuracy",  m.get('accuracy', 0)),
                            ("Precision", m.get('precision', 0)),
                            ("Recall",    m.get('recall', 0)),
                            ("F1-Score",  m.get('f1_score', 0)),
                            ("ROC-AUC",   m.get('roc_auc', 0)),
                        ]
                        for i, (lbl, val) in enumerate(metric_items):
                            with cols[i % 2]:
                                st.metric(lbl, f"{val*100:.1f}%")

        st.divider()

        # ── Panduan Penggunaan ──
        st.markdown('<div class="section-title">📖 Panduan</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#94a3b8; font-size:0.82rem; line-height:1.9;">
            1️⃣ Masukkan teks berita<br>
            2️⃣ Klik <b style="color:#a5b4fc;">Analisis Sekarang</b><br>
            3️⃣ Lihat hasil dan skor keyakinan<br>
            4️⃣ Gunakan tab <b style="color:#a5b4fc;">Perbandingan</b> untuk melihat hasil kedua model<br>
            5️⃣ Riwayat analisis tersimpan otomatis
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Disclaimer ──
        st.markdown("""
        <div style="color:#475569; font-size:0.72rem; text-align:center; line-height:1.6;">
            ⚠️ Sistem ini adalah alat bantu berbasis ML.<br>
            Selalu verifikasi informasi dari sumber terpercaya.<br><br>
            © 2024 Hoax Detection System<br>
            Tugas Akhir · Pembelajaran Mesin
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Fungsi utama aplikasi Streamlit."""

    # ── Inisialisasi Session State ──
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0

    # ── Muat Model & Metadata ──
    detector = load_detector()
    metadata = load_training_metadata()

    # ── Render Sidebar ──
    render_sidebar(detector, metadata)

    # ─────────────────────────────────────────────
    # HERO SECTION
    # ─────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🤖 Machine Learning · NLP · Bahasa Indonesia</div>
        <h1 class="hero-title">🛡️ HoaxRadar</h1>
        <p class="hero-subtitle">
            Sistem Cerdas Pendeteksi Hoaks Berita Politik Indonesia<br>
            <span style="opacity:0.7;">Berbasis TF-IDF · Naive Bayes · SVM · SMOTE</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Cek Status Model ──
    if not detector or not detector.is_ready:
        st.markdown("""
        <div style="background:rgba(234,179,8,0.1); border:1px solid rgba(234,179,8,0.3);
                    border-radius:16px; padding:28px; text-align:center; margin-bottom:24px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">⚠️</div>
            <div style="color:#fde68a; font-size:1.1rem; font-weight:700; margin-bottom:8px;">
                Model Belum Tersedia
            </div>
            <div style="color:#94a3b8; font-size:0.9rem;">
                Jalankan perintah berikut di terminal untuk memulai training:
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.code("cd hoax-detector\npython train.py", language="bash")
        st.info("💡 Pastikan file `data/indonesian_hoax_news.csv` sudah tersedia di folder `data/`.")
        return

    # ─────────────────────────────────────────────
    # TABS UTAMA
    # ─────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Analisis Teks", "⚖️ Perbandingan Model", "📋 Riwayat"])

    # ═══════════════════════════════════════════════
    # TAB 1: ANALISIS TEKS
    # ═══════════════════════════════════════════════
    with tab1:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("""
            <div class="section-title">✍️ Input Teks Berita</div>
            """, unsafe_allow_html=True)

            # Contoh teks
            with st.expander("💡 Muat Contoh Teks", expanded=False):
                example_texts = {
                    "Contoh Hoaks 🚨": "VIRAL!!! Presiden diam-diam menandatangani perjanjian rahasia yang menjual kedaulatan Indonesia kepada asing tanpa sepengetahuan DPR dan rakyat! Sumber terpercaya mengungkap konspirasi ini!",
                    "Contoh Fakta ✅": "Komisi Pemilihan Umum (KPU) RI resmi menetapkan jadwal tahapan Pemilu 2024. Pendaftaran calon peserta pemilu akan dibuka mulai 29 Juli hingga 13 Agustus 2022.",
                    "Contoh Hoaks 2 🚨": "Hati-hati!!! Chip 5G diam-diam disuntikkan ke vaksin Covid-19 untuk mengontrol pikiran masyarakat Indonesia oleh konglomerat global. Share sebelum dihapus!",
                }
                for btn_label, sample_text in example_texts.items():
                    if st.button(btn_label, key=f"btn_{btn_label}"):
                        st.session_state.input_text = sample_text

            # Area input teks
            user_input = st.text_area(
                label="Masukkan teks berita yang ingin dianalisis:",
                value=st.session_state.get('input_text', ''),
                height=220,
                placeholder="Contoh: Ketua DPR RI mengumumkan bahwa RUU baru telah disahkan dalam sidang paripurna...",
                key="main_text_input",
                label_visibility="collapsed",
            )

            # Counter karakter
            char_count = len(user_input) if user_input else 0
            word_count = len(user_input.split()) if user_input else 0
            st.markdown(f"""
            <div style="color:#475569; font-size:0.78rem; text-align:right; margin-top:4px;">
                {char_count:,} karakter · {word_count:,} kata
            </div>
            """, unsafe_allow_html=True)

            # Tombol analisis
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analisis Sekarang", key="analyze_main", type="primary")

            # Tips
            st.markdown("""
            <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                        border-radius:12px; padding:16px; margin-top:16px;">
                <div style="color:#a5b4fc; font-size:0.8rem; font-weight:600; margin-bottom:8px;">
                    💡 Tips Penggunaan
                </div>
                <ul style="color:#64748b; font-size:0.78rem; line-height:1.8; padding-left:16px;">
                    <li>Masukkan teks lengkap berita (minimal 10 karakter)</li>
                    <li>Hindari memasukkan hanya judul saja</li>
                    <li>Semakin panjang teks, hasil lebih akurat</li>
                    <li>Teks dalam Bahasa Indonesia memberikan hasil terbaik</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ── KOLOM HASIL ──
        with col_result:
            st.markdown("""
            <div class="section-title">📊 Hasil Analisis</div>
            """, unsafe_allow_html=True)

            if analyze_btn:
                if not user_input or not user_input.strip():
                    st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
                else:
                    # Spinner loading
                    with st.spinner("🔄 Menganalisis teks..."):
                        time.sleep(0.4)  # Efek visual
                        result = detector.predict(user_input)

                    # Tampilkan hasil
                    display_result(result)

                    # Simpan ke riwayat
                    if result.get('label') not in ['ERROR']:
                        st.session_state.history.insert(0, {
                            'No'       : len(st.session_state.history) + 1,
                            'Teks'     : user_input[:80] + ('...' if len(user_input) > 80 else ''),
                            'Hasil'    : result.get('label', 'N/A'),
                            'Keyakinan': f"{result.get('confidence', 0)*100:.1f}%",
                            'P(Fakta)' : f"{result.get('prob_fakta', 0)*100:.1f}%",
                            'P(Hoaks)' : f"{result.get('prob_hoaks', 0)*100:.1f}%",
                        })
                        st.session_state.analysis_count += 1

            else:
                # Placeholder ketika belum ada hasil
                st.markdown("""
                <div style="text-align:center; padding: 80px 20px; color:#334155;">
                    <div style="font-size:4rem; margin-bottom:16px; opacity:0.5;">🔍</div>
                    <div style="font-size:1rem; color:#475569; font-weight:500;">
                        Masukkan teks berita di sebelah kiri<br>dan klik <b style="color:#6366f1;">Analisis Sekarang</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # TAB 2: PERBANDINGAN MODEL
    # ═══════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="section-title">⚖️ Perbandingan MNB vs SVM</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#64748b; font-size:0.85rem; margin-bottom:20px;">
            Lihat bagaimana kedua model (Multinomial Naive Bayes dan SVM Linear) 
            memberikan prediksi berbeda pada teks yang sama.
        </div>
        """, unsafe_allow_html=True)

        compare_input = st.text_area(
            "Masukkan teks untuk dibandingkan:",
            height=150,
            placeholder="Masukkan teks berita di sini...",
            key="compare_input",
            label_visibility="visible",
        )

        compare_btn = st.button("⚖️ Bandingkan Kedua Model", key="compare_btn")

        if compare_btn:
            if not compare_input.strip():
                st.warning("⚠️ Masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Menganalisis dengan kedua model..."):
                    comparison = detector.compare_models(compare_input)

                if 'error' in comparison:
                    st.error(f"Error: {comparison['error']}")
                else:
                    col_mnb, col_svm = st.columns(2)

                    for col, model_name in zip([col_mnb, col_svm], ['MNB', 'SVM']):
                        with col:
                            m = comparison.get(model_name, {})
                            if 'error' in m:
                                st.error(f"{model_name}: {m['error']}")
                                continue

                            label      = m.get('label', 'N/A')
                            confidence = m.get('confidence', 0)
                            prob_fakta = m.get('prob_fakta', 0)
                            prob_hoaks = m.get('prob_hoaks', 0)

                            emoji = "🚨" if label == "HOAKS" else "✅"
                            color = "#f87171" if label == "HOAKS" else "#4ade80"
                            bg    = "rgba(248,113,113,0.1)" if label == "HOAKS" else "rgba(74,222,128,0.1)"
                            border = "rgba(248,113,113,0.4)" if label == "HOAKS" else "rgba(74,222,128,0.4)"

                            full_name = "Multinomial Naive Bayes" if model_name == 'MNB' else "SVM Linear Kernel"

                            st.markdown(f"""
                            <div style="background:{bg}; border:2px solid {border};
                                        border-radius:16px; padding:24px; text-align:center;">
                                <div style="color:#64748b; font-size:0.78rem; font-weight:600;
                                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">
                                    {full_name}
                                </div>
                                <div style="font-size:2.5rem;">{emoji}</div>
                                <div style="font-size:1.6rem; font-weight:900; color:{color};
                                            margin:8px 0;">{label}</div>
                                <div style="font-size:1rem; color:{color}; font-weight:600;">
                                    {confidence*100:.1f}% Keyakinan
                                </div>
                                <div style="margin-top:16px; display:flex; gap:8px; justify-content:center;">
                                    <span style="background:rgba(74,222,128,0.15); border:1px solid rgba(74,222,128,0.3);
                                                 border-radius:8px; padding:4px 12px; color:#4ade80; font-size:0.8rem;">
                                        ✅ Fakta: {prob_fakta*100:.1f}%
                                    </span>
                                    <span style="background:rgba(248,113,113,0.15); border:1px solid rgba(248,113,113,0.3);
                                                 border-radius:8px; padding:4px 12px; color:#f87171; font-size:0.8rem;">
                                        🚨 Hoaks: {prob_hoaks*100:.1f}%
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Analisis kesimpulan
                    mnb_label = comparison.get('MNB', {}).get('label', 'N/A')
                    svm_label = comparison.get('SVM', {}).get('label', 'N/A')
                    st.markdown("<br>", unsafe_allow_html=True)

                    if mnb_label == svm_label:
                        st.success(f"✅ **Kedua model sepakat**: Teks ini terindikasi sebagai **{mnb_label}**")
                    else:
                        st.warning("⚠️ **Kedua model memberikan hasil berbeda.** Lakukan verifikasi manual ke sumber terpercaya.")

    # ═══════════════════════════════════════════════
    # TAB 3: RIWAYAT ANALISIS
    # ═══════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div class="section-title">📋 Riwayat Analisis Sesi Ini</div>
        """, unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#334155;">
                <div style="font-size:3.5rem; margin-bottom:16px; opacity:0.4;">📋</div>
                <div style="color:#475569; font-size:0.95rem;">
                    Belum ada riwayat analisis.<br>Mulai analisis di tab <b>Analisis Teks</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Statistik ringkas
            total   = len(st.session_state.history)
            hoax_ct = sum(1 for h in st.session_state.history if h['Hasil'] == 'HOAKS')
            fact_ct = total - hoax_ct

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Total Analisis</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-item" style="border-color:rgba(248,113,113,0.3);">
                    <div class="metric-value" style="color:#f87171;">{hoax_ct}</div>
                    <div class="metric-label">🚨 Hoaks</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-item" style="border-color:rgba(74,222,128,0.3);">
                    <div class="metric-value" style="color:#4ade80;">{fact_ct}</div>
                    <div class="metric-label">✅ Fakta</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tampilkan tabel riwayat
            history_df = pd.DataFrame(st.session_state.history)

            # Styling DataFrame
            def style_hasil(val):
                if val == 'HOAKS':
                    return 'color: #f87171; font-weight: bold;'
                elif val == 'FAKTA':
                    return 'color: #4ade80; font-weight: bold;'
                return ''

            styled_df = history_df.style.applymap(style_hasil, subset=['Hasil'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Tombol hapus riwayat
            if st.button("🗑️ Hapus Semua Riwayat", key="clear_history"):
                st.session_state.history = []
                st.session_state.analysis_count = 0
                st.rerun()

    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:40px 20px 20px; color:#334155; font-size:0.78rem;">
        <div style="margin-bottom:8px;">
            🛡️ <b style="color:#475569;">HoaxRadar</b> · Sistem Deteksi Hoaks Politik Indonesia
        </div>
        <div>
            Dibuat dengan ❤️ untuk Tugas Akhir Pembelajaran Mesin ·
            <span style="color:#6366f1;">TF-IDF + SMOTE + GridSearchCV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
