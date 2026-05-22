"""
Emergency Vehicle Compliance Assessment System (SCAS)
Streamlit Dashboard - Main Application

Run with: streamlit run app.py
"""

import streamlit as st
import sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SCAS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* ── Root Variables ──────────────────────────────────────────────── */
    :root {
        --bg-primary: #0a0c10;
        --bg-card: #161924;
        --bg-hover: #1e2230;
        --border: #252938;
        --text-primary: #e8eaf0;
        --text-muted: #8b90a0;
        --accent-red: #e8272b;
        --accent-green: #10b981;
        --accent-blue: #3b82f6;
        --accent-orange: #f59e0b;
        --font-mono: 'JetBrains Mono', 'Courier New', monospace;
    }

    /* ── Global Overrides ────────────────────────────────────────────── */
    .main {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .stApp {
        background: var(--bg-primary);
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: var(--text-primary);
    }

    /* ── Hero Section ────────────────────────────────────────────────── */
    .hero-section {
        background: linear-gradient(135deg, #161924 0%, #0a0c10 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 48px 40px;
        margin-bottom: 32px;
        text-align: center;
    }

    .hero-tag {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent-blue);
        margin-bottom: 16px;
        font-weight: 600;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 700;
        line-height: 1.2;
        color: var(--text-primary);
        margin-bottom: 16px;
    }

    .hero-title span {
        color: var(--accent-red);
    }

    .hero-desc {
        font-size: 16px;
        line-height: 1.6;
        color: var(--text-muted);
        max-width: 800px;
        margin: 0 auto;
    }

    /* ── Page Headers ────────────────────────────────────────────────── */
    .page-header {
        margin-bottom: 32px;
    }

    .page-tag {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent-blue);
        margin-bottom: 12px;
        font-weight: 600;
    }

    .page-title {
        font-size: 36px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .page-subtitle {
        font-size: 15px;
        color: var(--text-muted);
        line-height: 1.6;
    }

    /* ── Cards ───────────────────────────────────────────────────────── */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .section-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
    }

    .section-card-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary);
    }

    .section-badge {
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        background: var(--bg-hover);
        color: var(--text-muted);
        border: 1px solid var(--border);
    }

    .section-badge.badge-live {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-green);
        border-color: var(--accent-green);
    }

    .section-badge.badge-ok {
        background: rgba(59, 130, 246, 0.1);
        color: var(--accent-blue);
        border-color: var(--accent-blue);
    }

    /* ── KPI Cards ───────────────────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-hover);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted);
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ── Feature Cards ───────────────────────────────────────────────── */
    .feature-card {
        background: var(--bg-hover);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        height: 100%;
        transition: all 0.2s;
    }

    .feature-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }

    .feature-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .feature-desc {
        font-size: 12px;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Timeline ────────────────────────────────────────────────────── */
    .timeline-item {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }

    .timeline-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-top: 4px;
        flex-shrink: 0;
    }

    .timeline-dot.tl-blue {
        background: var(--accent-blue);
    }

    .timeline-body {
        flex: 1;
    }

    .timeline-event {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .timeline-detail {
        font-size: 12px;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Status Badges ───────────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 4px 8px;
        border-radius: 4px;
    }

    .status-badge.status-active {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
    }

    .status-badge.status-violation {
        background: rgba(232, 39, 43, 0.15);
        color: var(--accent-red);
    }

    /* ── Utilities ───────────────────────────────────────────────────── */
    .metric-pill {
        display: inline-block;
        font-size: 11px;
        padding: 6px 12px;
        border-radius: 6px;
        background: var(--bg-hover);
        color: var(--text-muted);
        font-weight: 500;
    }

    /* ── Hide Streamlit Branding ─────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Force sidebar toggle button to always be visible above everything */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 0.4rem !important;
        left: 0.4rem !important;
        z-index: 99999 !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

_DB = Path(__file__).parent.parent / "outputs" / "database" / "violations_demo.db"
_DEMO_TOTAL = 14
_DEMO_MAJOR = 11

def _sidebar_stats():
    if not _DB.exists():
        return _DEMO_TOTAL, _DEMO_MAJOR
    try:
        conn = sqlite3.connect(str(_DB))
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Violation"); total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM Violation WHERE severity='Major'"); major = cur.fetchone()[0]
        conn.close()
        if total == 0:
            return _DEMO_TOTAL, _DEMO_MAJOR
        return total, major or _DEMO_MAJOR
    except Exception:
        return _DEMO_TOTAL, _DEMO_MAJOR

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand" style="display:flex;align-items:center;gap:12px;padding:16px 0;">
        <div style="width:40px;height:40px;background:var(--accent-red);border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:14px;color:#fff;flex-shrink:0;">EV</div>
        <div>
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);">EV-SCAN</div>
            <div style="font-size:11px;color:var(--text-muted);">Emergency Vehicle Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:6px 0 14px 0;">
        <div style="width:8px;height:8px;border-radius:50%;background:var(--accent-green);"></div>
        <span style="font-size:12px;color:var(--accent-green);font-weight:500;">System Online</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:var(--text-muted);letter-spacing:1px;'
                'text-transform:uppercase;margin-bottom:8px;">NAVIGATION</div>',
                unsafe_allow_html=True)

    selected = st.radio("Navigation",
                        ["Home", "Video Analysis", "Model Dashboard", "Compliance Reports"],
                        label_visibility="collapsed")

    st.markdown("---")

    st.markdown('<div style="font-size:10px;color:var(--text-muted);letter-spacing:1px;'
                'text-transform:uppercase;margin-bottom:10px;">QUICK STATS</div>',
                unsafe_allow_html=True)

    _total, _major = _sidebar_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:var(--bg-hover);border:1px solid var(--border);border-radius:6px;
                    padding:10px;text-align:center;">
            <div style="font-size:20px;font-weight:700;color:var(--text-primary);">{_total}</div>
            <div style="font-size:10px;color:var(--text-muted);">Violations</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:var(--bg-hover);border:1px solid var(--border);border-radius:6px;
                    padding:10px;text-align:center;">
            <div style="font-size:20px;font-weight:700;color:var(--accent-red);">{_major}</div>
            <div style="font-size:10px;color:var(--text-muted);">Major</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:var(--text-muted);">
        v2.4.1 · AI-Powered · © 2025 EV-SCAN
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE ROUTING
# ============================================================================

from pages_modules import home, video_analysis, dashboard, compliance_report

PAGES = {
    "Home":               home,
    "Video Analysis":     video_analysis,
    "Model Dashboard":    dashboard,
    "Compliance Reports": compliance_report,
}

PAGES[selected].show()
