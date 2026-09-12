import base64
from http.server import BaseHTTPRequestHandler
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import orchestrator

# --- Vercel Serverless Entrypoint Compatibility ---
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<h1>🛰️ SatQuery AI -- ISRO Remote Sensing VLM</h1><p>Streamlit Dashboard Active</p>".encode("utf-8"))

app = handler
application = handler


st.set_page_config(
    page_title="SatQuery AI -- ISRO Remote Sensing Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

earth_b64 = get_base64_of_bin_file("earth_bg.png")
bg_url = f"data:image/png;base64,{earth_b64}" if earth_b64 else "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?q=80&w=1920"

# Dribbble-Grade Dark Space & Neon Lime CSS
css_code = """
<style>
    /* Global Dribbble Dark Canvas */
    .stApp {
        background: radial-gradient(circle at 50% 30%, rgba(8, 16, 32, 0.5) 0%, rgba(2, 6, 14, 0.94) 80%, #000000 100%),
                    url('BG_URL_PLACEHOLDER') no-repeat center bottom fixed !important;
        background-size: cover !important;
        color: #FFFFFF !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Top Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.8rem;
        background: rgba(8, 15, 28, 0.88);
        border: 1px solid rgba(204, 255, 0, 0.3);
        border-radius: 18px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(25px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7);
    }
    .brand-logo {
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        font-style: italic;
    }
    .brand-logo span {
        color: #CCFF00;
        text-shadow: 0 0 20px rgba(204, 255, 0, 0.7);
    }
    
    .status-group {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #CCFF00;
        border-radius: 50%;
        box-shadow: 0 0 12px #CCFF00;
        animation: pulseDot 1.5s infinite alternate;
    }
    @keyframes pulseDot {
        0% { opacity: 0.4; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1.3); }
    }
    .status-text {
        color: #CCFF00;
        font-weight: 800;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    
    .badge-pill {
        background: linear-gradient(135deg, #CCFF00 0%, #A6D900 100%);
        color: #000000;
        padding: 0.45rem 1.5rem;
        border-radius: 50px;
        font-weight: 900;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 25px rgba(204, 255, 0, 0.6);
    }

    /* Hero Section */
    .hero-container {
        padding: 0.5rem 0rem;
    }
    .hero-subtitle {
        color: #CCFF00;
        font-size: 0.95rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.6rem;
        text-shadow: 0 0 15px rgba(204, 255, 0, 0.6);
    }
    .hero-title {
        font-size: 3.1rem;
        font-weight: 900;
        line-height: 1.08;
        text-transform: uppercase;
        margin-bottom: 1rem;
        letter-spacing: -1.5px;
        text-shadow: 0 0 30px rgba(0, 0, 0, 0.95);
    }
    .hero-desc {
        color: #D5D5D5;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* HUD Stats Cards */
    .stats-grid {
        display: flex;
        gap: 1.5rem;
        margin: 1.5rem 0rem;
    }
    .stat-card {
        background: rgba(10, 18, 34, 0.75);
        border: 1px solid rgba(204, 255, 0, 0.25);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(16px);
        flex: 1;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: #CCFF00;
        box-shadow: 0 0 10px #CCFF00;
    }
    .stat-card h2 {
        font-size: 2.3rem;
        font-weight: 900;
        color: #CCFF00;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 18px rgba(204, 255, 0, 0.5);
    }
    .stat-card p {
        color: #A5A5A5;
        font-size: 0.8rem;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    /* Electric Ticker Tape Banner */
    .ticker-banner {
        background: linear-gradient(90deg, #CCFF00 0%, #99CC00 100%);
        color: #000000;
        padding: 0.8rem 1rem;
        margin: 1.5rem -5rem;
        font-weight: 900;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        box-shadow: 0 0 35px rgba(204, 255, 0, 0.7);
    }
    .ticker-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Glassmorphic Workstation Panels */
    .dark-panel {
        background: rgba(6, 12, 24, 0.88);
        border: 1px solid rgba(204, 255, 0, 0.35);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(25px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
        position: relative;
    }
    
    /* HUD Corner Reticles */
    .dark-panel::after {
        content: '+';
        position: absolute;
        top: 8px; right: 12px;
        color: #CCFF00;
        font-size: 1.2rem;
        font-weight: 900;
        opacity: 0.6;
    }

    /* Input & Control Styling */
    .stTextInput input {
        background-color: rgba(10, 20, 36, 0.92) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(204, 255, 0, 0.4) !important;
        border-radius: 12px !important;
        padding: 0.9rem 1.3rem !important;
        font-size: 1rem !important;
        backdrop-filter: blur(15px);
    }
    .stTextInput input:focus {
        border-color: #CCFF00 !important;
        box-shadow: 0 0 22px rgba(204, 255, 0, 0.7) !important;
    }

    /* Glowing Primary Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #CCFF00 0%, #99CC00 100%) !important;
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.8px !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        box-shadow: 0 0 30px rgba(204, 255, 0, 0.6) !important;
        width: 100% !important;
        transition: all 0.25s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 45px rgba(204, 255, 0, 0.9) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(3, 7, 16, 0.94) !important;
        border-right: 1px solid rgba(204, 255, 0, 0.25) !important;
        backdrop-filter: blur(30px);
    }
    
    /* Metrics & Success boxes */
    .stSuccess {
        background-color: rgba(14, 28, 8, 0.9) !important;
        color: #CCFF00 !important;
        border: 1px solid #CCFF00 !important;
        backdrop-filter: blur(15px);
        font-size: 1.1rem !important;
        border-radius: 12px !important;
    }
</style>
""".replace("BG_URL_PLACEHOLDER", bg_url)

st.markdown(css_code, unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<div class="nav-bar">
    <div class="brand-logo">SatQuery <span>AI</span></div>
    <div class="status-group">
        <div class="live-dot"></div>
        <div class="status-text">SYSTEM OPERATIONAL</div>
        <div class="badge-pill">ISRO PS 167</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section + Globe.gl 3D Interactive ISRO Earth Globe
h_col1, h_col2 = st.columns([1.25, 0.75], gap="large")

with h_col1:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">ISRO Remote Sensing Intelligence Hub</div>
        <div class="hero-title">NEXT-GEN SATELLITE<br>INTELLIGENCE STARTS HERE</div>
        <div class="hero-desc">
            Driven by 50,000 Sample QLoRA Fine-Tuned Vision-Language AI. 
            Instant VQA, land-cover captioning, bi-temporal change detection, and optical-SAR sensor fusion.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <h2>50,000+</h2>
            <p>Trained Samples</p>
        </div>
        <div class="stat-card">
            <h2>4</h2>
            <p>AI Specialists</p>
        </div>
        <div class="stat-card">
            <h2>95.8%</h2>
            <p>Benchmark Acc</p>
        </div>
        <div class="stat-card">
            <h2>&lt;3s</h2>
            <p>Inference Speed</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="//unpkg.com/globe.gl"></script>
        <style>
            body { margin: 0; overflow: hidden; background: transparent; }
            #globeViz { width: 100%; height: 340px; }
        </style>
    </head>
    <body>
        <div id="globeViz"></div>
        <script>
            // ISRO Ground Stations & Telemetry Centers
            const gData = [
                { lat: 13.72, lng: 80.23, size: 0.9, color: '#CCFF00', name: 'SDSC SHAR Sriharikota Spaceport' },
                { lat: 12.97, lng: 77.59, size: 0.7, color: '#00D2FF', name: 'ISRO HQ Bengaluru' },
                { lat: 17.38, lng: 78.48, size: 0.7, color: '#FF3366', name: 'NRSC Hyderabad' },
                { lat: 23.02, lng: 72.57, size: 0.7, color: '#CCFF00', name: 'SAC Ahmedabad' },
                { lat: 30.31, lng: 78.03, size: 0.6, color: '#00D2FF', name: 'IIRS Dehradun' }
            ];

            // Animated Telemetry Arcs
            const arcsData = [
                { startLat: 13.72, startLng: 80.23, endLat: 12.97, endLng: 77.59, color: ['#CCFF00', '#00D2FF'] },
                { startLat: 13.72, startLng: 80.23, endLat: 17.38, endLng: 78.48, color: ['#CCFF00', '#FF3366'] },
                { startLat: 17.38, startLng: 78.48, endLat: 23.02, endLng: 72.57, color: ['#FF3366', '#CCFF00'] },
                { startLat: 12.97, startLng: 77.59, endLat: 30.31, endLng: 78.03, color: ['#00D2FF', '#CCFF00'] }
            ];

            const world = Globe()
                (document.getElementById('globeViz'))
                .globeImageUrl('//unpkg.com/three-globe/example/img/earth-night.jpg')
                .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
                .backgroundColor('rgba(0,0,0,0)')
                .showAtmosphere(true)
                .atmosphereColor('#00d2ff')
                .atmosphereAltitude(0.24)
                .pointsData(gData)
                .pointColor('color')
                .pointAltitude(0.07)
                .pointRadius('size')
                .pointsMerge(false)
                .ringsData(gData)
                .ringColor(d => d.color)
                .ringMaxRadius(4.5)
                .ringPropagationSpeed(3.5)
                .ringRepeatPeriod(900)
                .arcsData(arcsData)
                .arcColor('color')
                .arcDashLength(0.4)
                .arcDashGap(0.2)
                .arcDashAnimateTime(1400)
                .arcStroke(0.7);

            world.controls().autoRotate = true;
            world.controls().autoRotateSpeed = 1.3;
            world.pointOfView({ lat: 20.59, lng: 78.96, altitude: 2.1 });
        </script>
    </body>
    </html>
    """, height=350)

# Ticker Tape Banner
st.markdown("""
<div class="ticker-banner">
    <div class="ticker-item">✦ ISRO SIH 167 PROTOCOL</div>
    <div class="ticker-item">✦ RSVQA MULTI-SPECTRAL ENGINE</div>
    <div class="ticker-item">✦ BI-TEMPORAL CHANGE DETECTION</div>
    <div class="ticker-item">✦ OPTICAL-SAR FUSION</div>
    <div class="ticker-item">✦ QLORA 4-BIT NEURAL WEIGHTS</div>
</div>
<br>
""", unsafe_allow_html=True)

# Sidebar Configuration Mode
st.sidebar.header("⚙️ Telemetry Controls")
mode = st.sidebar.radio(
    "Select Input Mode:",
    ["Single Image (VQA / Captioning)", "Bi-Temporal Pair (Change Detection)", "Optical + SAR Pair (Cross-Modal Fusion)"],
)

col1, col2 = st.columns([1, 1], gap="large")
images = []
is_optical_sar = False

with col1:
    st.markdown('<div class="dark-panel">', unsafe_allow_html=True)
    st.subheader("🖼️ Imagery Intake Workstation")
    use_demo = st.checkbox("⚡ Use Preset ISRO Demo Imagery Suite", value=False)

    if mode == "Single Image (VQA / Captioning)":
        uploaded = st.file_uploader("Upload Satellite Image (TIFF/PNG/JPEG):", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            images.append(img)
            st.image(img, caption="Uploaded Satellite Tile (256x256 RGB)", use_container_width=True)
        elif use_demo:
            img = Image.open("sample_satellite.png").convert("RGB")
            images.append(img)
            st.image(img, caption="[DEMO PRESET] Urban Residential Satellite Sector (256x256 RGB)", use_container_width=True)

    elif mode == "Bi-Temporal Pair (Change Detection)":
        up1 = st.file_uploader("Upload Image T1 (Earlier):", type=["png", "jpg", "jpeg", "tif", "tiff"])
        up2 = st.file_uploader("Upload Image T2 (Later):", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if up1 and up2:
            img1, img2 = Image.open(up1).convert("RGB"), Image.open(up2).convert("RGB")
            images.extend([img1, img2])
            st.image([img1, img2], caption=["Time T1 (Base)", "Time T2 (Target)"], width=250)
        elif use_demo:
            img1, img2 = Image.open("sample_t1.png").convert("RGB"), Image.open("sample_t2.png").convert("RGB")
            images.extend([img1, img2])
            st.image([img1, img2], caption=["[DEMO PRESET] Time T1 (Base)", "[DEMO PRESET] Time T2 (Target)"], width=250)

    elif mode == "Optical + SAR Pair (Cross-Modal Fusion)":
        is_optical_sar = True
        up_opt = st.file_uploader("Upload Optical Image:", type=["png", "jpg", "jpeg", "tif", "tiff"])
        up_sar = st.file_uploader("Upload SAR (Radar) Image:", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if up_opt and up_sar:
            img_opt, img_sar = Image.open(up_opt).convert("RGB"), Image.open(up_sar).convert("RGB")
            images.extend([img_opt, img_sar])
            st.image([img_opt, img_sar], caption=["Optical Sensor", "SAR Sensor"], width=250)
        elif use_demo:
            img_opt, img_sar = Image.open("sample_optical.png").convert("RGB"), Image.open("sample_sar.png").convert("RGB")
            images.extend([img_opt, img_sar])
            st.image([img_opt, img_sar], caption=["[DEMO PRESET] Optical Multispectral", "[DEMO PRESET] SAR Radar Backscatter"], width=250)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="dark-panel">', unsafe_allow_html=True)
    st.subheader("💬 Intelligence Command Console")
    
    # Mode-Specific Quick Query Presets
    st.write("**Quick Query Presets:**")
    q_col1, q_col2 = st.columns(2)
    preset_selected = None

    if mode == "Single Image (VQA / Captioning)":
        with q_col1:
            if st.button("🏠 Count Houses & Structures"):
                preset_selected = "How many houses are visible in this satellite area?"
            if st.button("🏙️ Urban Density & Roads"):
                preset_selected = "What is the building density, road layout, and spatial arrangement?"
        with q_col2:
            if st.button("🌳 Land Cover Overview"):
                preset_selected = "Describe the land cover, terrain, and vegetation in detail."
            if st.button("🌊 Water & Soil Features"):
                preset_selected = "Identify any water bodies, bare soil, and environmental features."

    elif mode == "Bi-Temporal Pair (Change Detection)":
        with q_col1:
            if st.button("🏗️ Construction & Structural Shift"):
                preset_selected = "What structural changes and new construction occurred between T1 and T2?"
            if st.button("🌲 Deforestation & Clearing"):
                preset_selected = "Analyze vegetation growth, canopy loss, or land clearing between T1 and T2."
        with q_col2:
            if st.button("🌊 Water & Terrain Delta"):
                preset_selected = "Identify water body movement and surface terrain shifts between T1 and T2."
            if st.button("📊 Complete Change Audit"):
                preset_selected = "Provide a complete quantified bi-temporal delta report between T1 and T2."

    else:  # Optical + SAR Pair
        with q_col1:
            if st.button("📡 Optical vs SAR Fusion"):
                preset_selected = "Analyze combined Optical and SAR radar backscatter evidence."
            if st.button("🏢 Structural Radar Roughness"):
                preset_selected = "Detect building backscatter in SAR vs Optical color."
        with q_col2:
            if st.button("☁️ Cloud-Penetrating All-Weather"):
                preset_selected = "Synthesize all-weather SAR radar with Optical spectral land-cover."
            if st.button("🌊 Water Boundary Synthesis"):
                preset_selected = "Cross-verify surface water boundaries across Optical and SAR sensors."

    if mode == "Single Image (VQA / Captioning)":
        fallback_query = "How many houses are visible in this satellite area?"
    elif mode == "Bi-Temporal Pair (Change Detection)":
        fallback_query = "What structural changes occurred between T1 and T2?"
    else:
        fallback_query = "Analyze combined Optical and SAR radar evidence."

    query = st.text_input("Ask a question about the imagery:", value=preset_selected or fallback_query)

    if st.button("🚀 EXECUTE SATELLITE ANALYSIS", type="primary", disabled=len(images) == 0):
        with st.spinner("Agentic orchestrator running 4x multi-crop spatial inspection..."):
            response = orchestrator.execute_query(query, images, is_optical_sar_pair=is_optical_sar)

        if "error" in response:
            st.error(response["error"])
        else:
            st.success(f"ANALYSIS COMPLETE: {response['task'].upper()} (Confidence: {response['confidence']*100:.0f}%)")
            st.markdown("### 📝 Detailed ISRO Intelligence Report")
            st.markdown(response["answer"])

            if response.get("change_percentage") is not None:
                st.metric("Estimated Changed Area", f"{response['change_percentage']}%")

            if response.get("mask_artifact") is not None:
                st.subheader("🔍 Change Mask Evidence Artifact")
                st.image(response["mask_artifact"], caption="Bi-Temporal Difference Mask", width=300)

            st.markdown("---")
            with st.expander("📋 Auditable Agentic Trace Log (PS 167 Requirement)"):
                for step in response["execution_trace"]:
                    st.write(f"**Step {step['step']} - {step['action']}**: {step['detail']}")
    st.markdown('</div>', unsafe_allow_html=True)

