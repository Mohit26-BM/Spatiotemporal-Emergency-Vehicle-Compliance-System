import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os, sqlite3
from pathlib import Path

_FRONTEND = Path(__file__).parent.parent
_CAPSTONE = _FRONTEND.parent
_SRC      = _CAPSTONE / "src"
sys.path.insert(0, str(_SRC))

DB_PATH      = _CAPSTONE / "outputs" / "database" / "violations_demo.db"
EVIDENCE_DIR = _CAPSTONE / "outputs" / "evidence_frames"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,25,32,0.8)",
    font_color="#8b90a0",
    font_family="Inter",
    xaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", tickfont_color="#4b5060"),
    yaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", tickfont_color="#4b5060"),
)

def _load_violations():
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            "SELECT v.violation_id, v.timestamp, vh.track_id, "
            "v.distance_px, v.severity, v.frame_id, v.evidence_path "
            "FROM Violation v JOIN Vehicle vh ON v.vehicle_id = vh.vehicle_id "
            "ORDER BY v.frame_id ASC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">◈ COMPLIANCE REPORTING</div>
        <h1 class="page-title">COMPLIANCE REPORTS</h1>
        <p class="page-subtitle">
            Audit-ready reports generated directly from the violations SQLite database
            produced by the SCAS compliance pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df      = _load_violations()
    db_live = DB_PATH.exists() and not df.empty

    if not db_live:
        st.warning("No violation data found. Run the pipeline on the **Video Analysis** page first.")
        return

    total  = len(df)
    major  = int((df["severity"] == "Major").sum())
    minor  = int((df["severity"] == "Minor").sum())
    uveh   = int(df["track_id"].nunique())
    frames = int(df["frame_id"].max() - df["frame_id"].min()) if total > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, (label, val, color) in zip(
        [k1, k2, k3, k4, k5],
        [
            ("Total Violations", total,  "#e8272b"),
            ("Major",            major,  "#e8272b"),
            ("Minor",            minor,  "#f59e0b"),
            ("Unique Offenders", uveh,   "#10b981"),
            ("Frame Span",       frames, "#8b90a0"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Filters</div>
        </div>
    """, unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        sev_filter = st.selectbox("Severity", ["All", "Major", "Minor"])
    with f2:
        track_opts   = ["All"] + [str(t) for t in sorted(df["track_id"].unique().tolist())]
        track_filter = st.selectbox("Track ID", track_opts)
    with f3:
        sort_by = st.selectbox("Sort By", ["Frame # ↑", "Distance ↓", "Severity"])
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = df.copy()
    if sev_filter != "All":
        filtered = filtered[filtered["severity"] == sev_filter]
    if track_filter != "All":
        filtered = filtered[filtered["track_id"] == int(track_filter)]
    if sort_by == "Distance ↓":
        filtered = filtered.sort_values("distance_px", ascending=False)
    elif sort_by == "Severity":
        filtered = filtered.sort_values("severity")

    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Incident Log — violations_demo.db</div>
            <span class="section-badge badge-live">LIVE DATA</span>
        </div>
    """, unsafe_allow_html=True)
    display = filtered[["violation_id", "timestamp", "track_id",
                         "distance_px", "severity", "frame_id"]].copy()
    display.columns = ["ID", "Timestamp", "Track ID", "Distance (px)", "Severity", "Frame #"]

    def _sev_color(val):
        if val == "Major": return "color:#e8272b;font-weight:600;"
        if val == "Minor": return "color:#f59e0b;font-weight:600;"
        return ""

    st.dataframe(
        display.style.map(_sev_color, subset=["Severity"]),
        use_container_width=True,
        height=340,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Severity Breakdown</div>
            </div>
        """, unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts()
        fig = go.Figure(go.Pie(
            labels=sev_counts.index.tolist(),
            values=sev_counts.values.tolist(),
            hole=0.6,
            marker_colors=["#e8272b" if l == "Major" else "#f59e0b" for l in sev_counts.index],
            textinfo="label+percent",
            textfont=dict(color="#8b90a0", size=11),
        ))
        fig.update_layout(**PLOTLY_THEME, height=260,
                          margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Violations per Track ID</div>
            </div>
        """, unsafe_allow_html=True)
        per_track = df.groupby("track_id").size().reset_index(name="count")
        per_track["label"] = "Track " + per_track["track_id"].astype(str)
        bar_colors = ["#e8272b" if c >= 3 else "#f59e0b" for c in per_track["count"]]
        fig2 = go.Figure(go.Bar(x=per_track["label"], y=per_track["count"],
                                marker_color=bar_colors))
        fig2.update_layout(**PLOTLY_THEME, height=260,
                           yaxis_title="Violations",
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Distance at Violation (pixels from EV)</div>
        </div>
    """, unsafe_allow_html=True)
    fig3 = go.Figure(go.Histogram(
        x=df["distance_px"], nbinsx=12, marker_color="#3b82f6", opacity=0.85,
    ))
    fig3.update_layout(**PLOTLY_THEME, height=220,
                       xaxis_title="Pixel distance from EV at time of violation",
                       yaxis_title="Count",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    evidence_imgs = sorted(EVIDENCE_DIR.glob("*.jpg")) if EVIDENCE_DIR.exists() else []
    if evidence_imgs:
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Evidence Images — Cropped from Video</div>
                <span class="section-badge">JPEG · AUTO-SAVED BY PIPELINE</span>
            </div>
        """, unsafe_allow_html=True)
        cols = st.columns(4)
        for i, img in enumerate(evidence_imgs):
            with cols[i % 4]:
                st.image(str(img), caption=img.stem, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)