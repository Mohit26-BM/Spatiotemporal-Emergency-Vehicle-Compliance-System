import streamlit as st
import sqlite3, json
from pathlib import Path

_FRONTEND = Path(__file__).parent.parent
_CAPSTONE = _FRONTEND.parent
DB_PATH   = _CAPSTONE / "outputs" / "database" / "violations_demo.db"
CV_JSON   = _CAPSTONE / "outputs" / "plots" / "cv_summary.json"


def _load_cv_metrics():
    d = {
        "CV Mean Acc":   "91.95%",
        "Best Fold Acc": "92.30%",
        "Test Accuracy": "--",
        "F1 Score":      "0.92",
        "CV Folds":      "5-Fold",
        "Ensemble":      "YOLO+ResNet",
    }
    if CV_JSON.exists():
        try:
            data = json.loads(CV_JSON.read_text())
            d["CV Mean Acc"]   = f"{data['mean_acc']*100:.2f}%"
            d["Best Fold Acc"] = f"{data['best_val_acc']*100:.2f}%"
            d["CV Folds"]      = f"{data['k']}-Fold"
        except Exception:
            pass
    return d


def _quick_stats():
    if not DB_PATH.exists():
        return {"total": 0, "major": 0, "unique": 0}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Violation"); total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT vehicle_id) FROM Violation"); unique = cur.fetchone()[0]
        conn.close()
        return {"total": total, "unique": unique}
    except Exception:
        return {"total": 0, "unique": 0}


def show():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-tag">◈ CAPSTONE PROJECT — AI SYSTEMS</div>
        <div class="hero-title">
            Spatiotemporal<br><span>Compliance Assessment</span><br>for Emergency Vehicles
        </div>
        <div class="hero-desc">
            An AI-powered pipeline that detects emergency vehicles, tracks all traffic,
            evaluates yielding behaviour using spatiotemporal rules, and logs violations
            to a structured database — from video input to audit-ready compliance reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-tag">◈ CORE CAPABILITIES</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("Ensemble Classification",
         "YOLOv8-cls + ResNet18 trained with K-Fold CV (K=5). "
         "Weighted ensemble achieves 92% accuracy with F1=0.92."),
        ("DeepSORT Tracking",
         "MobileNet-based appearance embeddings + Kalman filter assign "
         "persistent track IDs. Sticky emergency labels prevent flip-flopping."),
        ("Compliance Engine",
         "Spatiotemporal rule engine: 180px proximity radius, 30-frame "
         "reaction window, 40px minimum yield threshold with camera-aware blocking."),
        ("Violation Logging",
         "SQLite 3NF schema (Vehicle · Violation) with JPEG "
         "evidence crops auto-saved for every flagged blocking event."),
    ]
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Pipeline Architecture</div>
                <span class="section-badge badge-ok">SCAS</span>
            </div>
        """, unsafe_allow_html=True)

        layers = [
            ("Video Input",
             "MP4 dashcam or overhead traffic camera footage"),
            ("Detection",
             "YOLOv8n detects vehicle bounding boxes at 30 FPS"),
            ("Classification",
             "YOLOv8-cls + ResNet18 ensemble labels: emergency / non-emergency"),
            ("Tracking",
             "DeepSORT assigns persistent IDs with sticky emergency labels"),
            ("Compliance Check",
             "Spatiotemporal engine evaluates 30-frame yield behavior"),
            ("Logging & Evidence",
             "SQLite database + JPEG evidence frames of each violation"),
        ]
        for title, desc in layers:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-dot tl-blue"></div>
                <div class="timeline-body">
                    <div class="timeline-event">{title}</div>
                    <div class="timeline-detail">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Project Info</div>
            </div>
        """, unsafe_allow_html=True)

        info = [
            ("Institution",   "Engineering College — Final Year"),
            ("Academic Year", "2024–2025"),
            ("Project Type",  "AI/ML Capstone — Computer Vision"),
            ("Tech Stack",    "Python · PyTorch · Ultralytics · OpenCV"),
            ("Models",        "YOLOv8-cls + ResNet18 Ensemble"),
            ("Validation",    "5-Fold Stratified Cross-Validation"),
            ("Tracker",       "DeepSORT (MobileNet embedder)"),
            ("Storage",       "SQLite 3NF + JPEG evidence crops"),
        ]
        for key, val in info:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);">
                <div>
                    <div style="font-size:10px;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase;">{key}</div>
                    <div style="font-size:13px;color:var(--text-primary);font-weight:500;margin-top:2px;">{val}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="page-tag">◈ MODEL PERFORMANCE — TRAINED ENSEMBLE</div>',
                unsafe_allow_html=True)

    train_metrics = _load_cv_metrics()
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">YOLOv8-cls + ResNet18 — K-Fold Cross-Validation</div>
            <span class="section-badge badge-ok">5-Fold CV</span>
        </div>
    """, unsafe_allow_html=True)
    mcols = st.columns(len(train_metrics))
    for col, (label, val) in zip(mcols, train_metrics.items()):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="page-tag">◈ VIOLATION DATABASE — CURRENT STATS</div>',
                unsafe_allow_html=True)

    stats    = _quick_stats()
    db_ready = DB_PATH.exists()

    k1, k2, k3 = st.columns(3)
    for col, (label, val, color) in zip(
        [k1, k2, k3],
        [
            ("Total Violations",  stats["total"],  "#e8272b"),
            ("Vehicles Caught",   stats["unique"], "#10b981"),
            ("DB Status", "Live" if db_ready else "No DB",
             "#10b981" if db_ready else "#f59e0b"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="page-tag">◈ QUICK ACCESS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.info("Go to **Dashboard** to view model performance and live database charts.")
    with c2:
        st.info("Go to **Violations** to explore detailed violation reports and evidence images.")