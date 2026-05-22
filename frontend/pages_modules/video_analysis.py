import streamlit as st
import pandas as pd
from pathlib import Path

_FRONTEND = Path(__file__).parent.parent
_CAPSTONE = _FRONTEND.parent

_VIDEOS      = _CAPSTONE / "outputs" / "videos"
_SCREENSHOTS = _CAPSTONE / "outputs" / "screenshots"

# Video paths
OVERHEAD_IN    = _VIDEOS / "Overhead_Video" / "Overhead_Input.mp4"
OVERHEAD_OUT   = _VIDEOS / "Overhead_Video" / "Overhead_Output.mp4"
DASHCAM_IN     = _VIDEOS / "Dashcam_Video"  / "Dashcam_Input.mp4"
DASHCAM_OUT    = _VIDEOS / "Dashcam_Video"  / "Dashcam_Output.mp4"

# Screenshot paths
OVERHEAD_SS_EV   = _SCREENSHOTS / "Overhead_Video" / "Overhead_Emergency_Vehicle_Detcted.png"
OVERHEAD_SS_VIOL = _SCREENSHOTS / "Overhead_Video" / "Overhead_Violation.png"
DASHCAM_SS_EV    = _SCREENSHOTS / "Dashcam_Video"  / "Dashcam_Emergency_Vehicle_Detected.png"
DASHCAM_SS_VIOL  = _SCREENSHOTS / "Dashcam_Video"  / "Dashcam_Violation_Detected.png"


def _video_bytes(path):
    """Read video file as bytes for st.video()"""
    with open(str(path), "rb") as f:
        return f.read()

def _video_codec(path):
    """Return the MP4 codec marker if it can be detected from the file."""
    try:
        data = path.read_bytes()
    except OSError:
        return "unknown"

    for marker, label in (
        (b"avc1", "H.264"),
        (b"hvc1", "H.265"),
        (b"hev1", "H.265"),
        (b"mp4v", "MPEG-4 Part 2"),
    ):
        if marker in data:
            return label
    return "unknown"


def _show_video(path, caption):
    """Display a video and explain/download when the codec is not browser-friendly."""
    codec = _video_codec(path)
    if codec == "MPEG-4 Part 2":
        st.warning(
            "This MP4 was encoded with mp4v/MPEG-4 Part 2, which many browsers do not "
            "play inside Streamlit. Re-encode it to H.264 (avc1) for in-page playback."
        )

    video_data = _video_bytes(path)
    st.video(video_data)
    st.caption(caption)

    if codec == "MPEG-4 Part 2":
        st.download_button(
            "Download video",
            data=video_data,
            file_name=path.name,
            mime="video/mp4",
            key=f"download_{path.stem}",
            use_container_width=True,
        )


def _show_video_section(label, in_path, out_path, ss_ev, ss_viol, stats):
    """Display video section with screenshots and videos"""
    st.markdown(f"""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">{label}</div>
            <span class="section-badge badge-ok">REAL FOOTAGE</span>
        </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    k_cols = st.columns(4)
    for col, (lbl, val, color) in zip(k_cols, stats):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Screenshots
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;'
                    'letter-spacing:1px;text-transform:uppercase;">Emergency Vehicle Detected</div>',
                    unsafe_allow_html=True)
        if ss_ev.exists():
            st.image(str(ss_ev), width="stretch")
            st.caption("EV classification: Emergency vehicle present in scene")
        else:
            st.info(f"Screenshot not found: {ss_ev.name}")

    with sc2:
        st.markdown('<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;'
                    'letter-spacing:1px;text-transform:uppercase;">Violation Detected</div>',
                    unsafe_allow_html=True)
        if ss_viol.exists():
            st.image(str(ss_viol), width="stretch")
            st.caption("Compliance check: Vehicle failing to yield properly")
        else:
            st.info(f"Screenshot not found: {ss_viol.name}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Videos
    v1, v2 = st.columns(2)
    with v1:
        st.markdown('<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;'
                    'letter-spacing:1px;text-transform:uppercase;">Input Video</div>',
                    unsafe_allow_html=True)
        if in_path.exists():
            _show_video(in_path, f"{in_path.name} - {in_path.stat().st_size // 1024} KB")
        else:
            st.info(f"Video not found: {in_path.name}")

    with v2:
        st.markdown('<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;'
                    'letter-spacing:1px;text-transform:uppercase;">Annotated Output</div>',
                    unsafe_allow_html=True)
        if out_path.exists():
            _show_video(out_path, f"{out_path.name} - {out_path.stat().st_size // 1024} KB")
        else:
            st.info(f"Video not found: {out_path.name}")

    st.markdown("</div>", unsafe_allow_html=True)


def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-tag">VIDEO RESULTS</div>
        <h1 class="page-title">VIDEO ANALYSIS</h1>
        <p class="page-subtitle">
            Real-world test results from two traffic scenarios — overhead intersection camera
            and dashcam perspective — validated on actual traffic footage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Overhead Camera", "Dashcam Camera"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        _show_video_section(
            "Overhead Camera — Urban Intersection",
            OVERHEAD_IN, OVERHEAD_OUT,
            OVERHEAD_SS_EV, OVERHEAD_SS_VIOL,
            [
                ("Resolution",     "828 × 720",  "#8b90a0"),
                ("Frames / FPS",   "756 · 60fps", "#8b90a0"),
                ("EV Detection",   "87.6%",       "#10b981"),
                ("Violations",     "10",          "#e8272b"),
            ],
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Violation Breakdown — Overhead Camera</div>
            </div>
        """, unsafe_allow_html=True)
        
        overhead_df = pd.DataFrame([
            (307,  57,  110.0, -69.2, "Approaching"),
            (454,  134, 106.8, -44.8, "Approaching"),
            (530,  106, 143.8, -34.8, "Approaching"),
            (586,  114, 160.5, -19.1, "Approaching"),
            (590,  106,  94.3, -18.6, "Approaching"),
            (650,  106,  75.2,  -6.5, "Approaching"),
            (672,  113, 157.3, -20.5, "Approaching"),
            (711,  192,  73.1,   1.0, "Stationary"),
            (726,  128, 162.4, -16.3, "Approaching"),
            (732,  113, 115.1, -19.8, "Approaching"),
        ], columns=["Frame", "Track ID", "Distance (px)", "Yield (px)", "Type"])
        
        st.dataframe(overhead_df, use_container_width=True, height=320)
        
        st.markdown("""
        **Key Observations:**
        - **Track 106:** Persistent blocker (3 violations over 2 seconds)
        - **Track 113:** Re-entry violation (2 violations, re-entered after cooldown)
        - **Track 57:** Most aggressive approach (-69.2px yield)
        - **Track 192:** Stationary blocker (1.0px yield, no movement)
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        _show_video_section(
            "Dashcam Camera — Highway Scene",
            DASHCAM_IN, DASHCAM_OUT,
            DASHCAM_SS_EV, DASHCAM_SS_VIOL,
            [
                ("Resolution",     "1280 × 720",  "#8b90a0"),
                ("Frames / FPS",   "524 · 30fps",  "#8b90a0"),
                ("EV Detection",   "94.3%",        "#10b981"),
                ("Violations",     "4",            "#e8272b"),
            ],
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-card">
            <div class="section-card-header">
                <div class="section-card-title">Violation Breakdown — Dashcam Camera</div>
            </div>
        """, unsafe_allow_html=True)
        
        dashcam_df = pd.DataFrame([
            (293, 88,  72.6,   9.7, "Slow yield"),
            (322, 50,  37.7, -44.7, "Approaching"),
            (354, 141, 73.8,  22.8, "Slow yield"),
            (357, 141, 77.2,  39.4, "Slow yield"),
        ], columns=["Frame", "Track ID", "Distance (px)", "Yield (px)", "Type"])
        
        st.dataframe(dashcam_df, use_container_width=True, height=200)
        
        st.markdown("""
        **Key Observations:**
        - **Track 50:** Most aggressive (37.7px proximity, -44.7px yield)
        - **Track 141:** Edge case anomaly (2 violations in 3 frames)
        - **Track 88:** Slow yield (9.7px insufficient movement)
        - **Better lighting:** Higher EV detection rate (94.3% vs 87.6%)
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Comparison Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">
            <div class="section-card-title">Camera Perspective Comparison</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Overhead Camera (Video 3)**
        
        **Advantages:**
        - Wide field of view
        - Multiple lanes visible
        - Better traffic pattern analysis
        - Persistent tracking across frame
        
        **Challenges:**
        - Night conditions (lower EV detection: 87.6%)
        - Motion blur at 60 FPS
        - Overhead perspective distortion
        - More occlusion events
        
        **Violation Pattern:** More violations (10) due to complex multi-lane scenario
        """)
    
    with col2:
        st.markdown("""
        **Dashcam Camera (Video 4)**
        
        **Advantages:**
        - Better lighting (94.3% EV detection)
        - First-person perspective
        - Clearer vehicle classification
        - Less motion blur (30 FPS)
        
        **Challenges:**
        - Limited field of view
        - Only forward-facing coverage
        - Closer proximity measurements
        - Fewer lanes visible
        
        **Violation Pattern:** Fewer violations (4) in simpler highway scenario
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
