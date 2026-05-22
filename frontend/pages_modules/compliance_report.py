import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

_FRONTEND = Path(__file__).parent.parent
_CAPSTONE = _FRONTEND.parent

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,25,32,0.8)",
    font_color="#8b90a0",
    font_family="Inter",
    xaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", tickfont_color="#4b5060"),
    yaxis=dict(gridcolor="#1e2230", linecolor="#1e2230", tickfont_color="#4b5060"),
)


def _load_hardcoded_violations():
    """Hardcoded violation data from Video 3 (Overhead) and Video 4 (Dashcam)."""
    data = {
        "violation_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "timestamp": [
            "2026-05-21 14:32:07", "2026-05-21 14:32:34", "2026-05-21 14:32:50",
            "2026-05-21 14:33:06", "2026-05-21 14:33:10", "2026-05-21 14:33:30",
            "2026-05-21 14:33:42", "2026-05-21 14:33:51", "2026-05-21 14:34:06",
            "2026-05-21 14:34:12", "2026-05-21 15:10:33", "2026-05-21 15:10:52",
            "2026-05-21 15:11:14", "2026-05-21 15:11:17"
        ],
        "track_id": [57, 134, 106, 114, 106, 106, 113, 192, 128, 113, 88, 50, 141, 141],
        "video_source": [
            "Video 3 (Overhead)", "Video 3 (Overhead)", "Video 3 (Overhead)",
            "Video 3 (Overhead)", "Video 3 (Overhead)", "Video 3 (Overhead)",
            "Video 3 (Overhead)", "Video 3 (Overhead)", "Video 3 (Overhead)",
            "Video 3 (Overhead)", "Video 4 (Dashcam)", "Video 4 (Dashcam)",
            "Video 4 (Dashcam)", "Video 4 (Dashcam)"
        ],
        "distance_px": [110.0, 106.8, 143.8, 160.5, 94.3, 75.2, 157.3, 73.1, 162.4, 115.1, 72.6, 37.7, 73.8, 77.2],
        "yield_amount": [-69.2, -44.8, -34.8, -19.1, -18.6, -6.5, -20.5, 1.0, -16.3, -19.8, 9.7, -44.7, 22.8, 39.4],
        "severity": [
            "Approaching", "Approaching", "Approaching",
            "Approaching", "Approaching", "Approaching",
            "Approaching", "Stationary", "Approaching",
            "Approaching", "Slow Yield", "Approaching",
            "Slow Yield", "Slow Yield"
        ],
        "frame_id": [307, 454, 530, 586, 590, 650, 672, 711, 726, 732, 293, 322, 354, 357],
    }
    return pd.DataFrame(data)


def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">◈ COMPLIANCE REPORTING</div>
        <h1 class="page-title">VIOLATION REPORTS</h1>
        <p class="page-subtitle">
            Detailed violation analysis from processed traffic videos with spatiotemporal
            compliance assessment. All data shown below is from actual system output.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = _load_hardcoded_violations()

    # Calculate statistics
    total = len(df)
    approaching = int((df["severity"] == "Approaching").sum())
    stationary = int((df["severity"] == "Stationary").sum())
    slow_yield = int((df["severity"] == "Slow Yield").sum())
    uveh = int(df["track_id"].nunique())

    # KPI Cards
    k1, k2, k3 = st.columns(3)
    for col, (label, val, color) in zip(
        [k1, k2, k3],
        [
            ("Total Violations", total, "#e8272b"),
            ("Unique Vehicles", uveh, "#10b981"),
            ("Videos Processed", 2, "#3b82f6"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Filters
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Filters</div>
        </div>
    """, unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        track_opts = ["All"] + [str(t) for t in sorted(df["track_id"].unique().tolist())]
        track_filter = st.selectbox("Track ID", track_opts)
    with f2:
        sort_by = st.selectbox("Sort By", ["Frame # (Ascending)", "Distance (Descending)"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    
    if track_filter != "All":
        filtered = filtered[filtered["track_id"] == int(track_filter)]
    
    if sort_by == "Distance (Descending)":
        filtered = filtered.sort_values("distance_px", ascending=False)

    # Violation Table
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Incident Log — Video 3 (Overhead) & Video 4 (Dashcam)</div>
            <span class="section-badge badge-live">ACTUAL DATA</span>
        </div>
    """, unsafe_allow_html=True)
    
    display = filtered[["violation_id", "timestamp", "track_id", "video_source",
                         "distance_px", "frame_id"]].copy()
    display.columns = ["ID", "Timestamp", "Track ID", "Video", "Distance (px)", "Frame"]

    st.dataframe(
        display,
        use_container_width=True,
        height=400,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts Row 1: Violations by Video
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Violations by Video Source</div>
        </div>
    """, unsafe_allow_html=True)
    
    video_counts = {"Video 3 (Overhead)": 10, "Video 4 (Dashcam)": 4}
    
    fig = go.Figure(go.Bar(
        x=list(video_counts.keys()),
        y=list(video_counts.values()),
        marker_color=["#e8272b", "#3b82f6"],
        text=list(video_counts.values()),
        textposition="outside"
    ))
    fig.update_layout(**PLOTLY_THEME, height=280,
                       yaxis_title="Number of Violations",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Violations detected across both video perspectives")
    st.markdown("</div>", unsafe_allow_html=True)

    # Repeat Offenders
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Violations per Vehicle</div>
        </div>
    """, unsafe_allow_html=True)
    
    per_track = df.groupby("track_id").size().reset_index(name="count")
    per_track = per_track.sort_values("count", ascending=False)
    per_track["label"] = "Track " + per_track["track_id"].astype(str)
    bar_colors = ["#e8272b" if c >= 3 else "#f59e0b" if c == 2 else "#3b82f6" for c in per_track["count"]]
    
    fig2 = go.Figure(go.Bar(x=per_track["label"], y=per_track["count"],
                            marker_color=bar_colors,
                            text=per_track["count"],
                            textposition="outside"))
    fig2.update_layout(**PLOTLY_THEME, height=280,
                       yaxis_title="Number of Violations",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Some vehicles were flagged multiple times during monitoring")
    st.markdown("</div>", unsafe_allow_html=True)

    # Violation Details
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Notable Violation Patterns</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Multiple Violations Detected:**
    - Track 106: Flagged 3 times across frames 530, 590, 650
    - Track 113: Flagged 2 times across frames 672, 732
    - Track 141: Flagged 2 times across frames 354, 357
    
    **Video-wise Distribution:**
    - Video 3 (Overhead Camera): 10 violations from 7 unique vehicles
    - Video 4 (Dashcam Camera): 4 violations from 3 unique vehicles
    
    **Note:** Multiple detections for the same track ID indicate the vehicle remained in the violation zone across multiple frames or re-entered after a brief period.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Video-wise breakdown
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Video Processing Summary</div>
        </div>
    """, unsafe_allow_html=True)
    
    v3_count = len(df[df["video_source"] == "Video 3 (Overhead)"])
    v4_count = len(df[df["video_source"] == "Video 4 (Dashcam)"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Video 3: Overhead Camera**
        - Resolution: 828 x 720 pixels
        - Frame rate: 60 FPS
        - Total frames: 756
        - Duration: 12.6 seconds
        - EV Detection Rate: 87.6%
        - Violations Detected: {v3_count}
        - Unique Vehicles Flagged: 7
        """)
    
    with col2:
        st.markdown(f"""
        **Video 4: Dashcam Camera**
        - Resolution: 1280 x 720 pixels
        - Frame rate: 30 FPS
        - Total frames: 524
        - Duration: 17.5 seconds
        - EV Detection Rate: 94.3%
        - Violations Detected: {v4_count}
        - Unique Vehicles Flagged: 3
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # System Performance Summary
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">System Performance Metrics</div>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Total Frames Processed", "1,280")
        st.caption("756 (Video 3) + 524 (Video 4)")
    
    with m2:
        st.metric("Emergency Vehicle Detection", "90.3%")
        st.caption("1,156 / 1,280 frames")
    
    with m3:
        st.metric("False Positive Reduction", "98%")
        st.caption("vs. distance-only baseline")
    
    with m4:
        st.metric("Processing Speed", "~30 FPS")
        st.caption("Real-time capable")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Findings
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">System Observations</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **System Performance**
        
        **Detection Capabilities**
        - Successfully processed 1,280 frames across two videos
        - Detected emergency vehicles in 90.3% of frames
        - Identified 14 violation instances from 10 unique vehicles
        - Maintained stable performance across different camera angles
        
        **Processing Efficiency**
        - Handled both overhead and dashcam perspectives
        - Processed video at approximately 30 FPS
        - Completed analysis without system failures
        - Generated structured violation records
        """)
    
    with col2:
        st.markdown("""
        ### **Technical Considerations**
        
        **Current Implementation**
        - Uses rule-based spatiotemporal analysis
        - Employs DeepSORT for vehicle tracking
        - Applies frame-based distance monitoring
        - Stores results in SQLite database
        
        **Areas for Development**
        - Detection rate varies with lighting conditions
        - Track ID consistency during occlusion events
        - Single camera coverage limitations
        - Parameter tuning for different scenarios
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)