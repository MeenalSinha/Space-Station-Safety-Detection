import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Space Station Safety Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
SAFETY_CLASSES = [
    "OxygenTank",
    "NitrogenTank", 
    "FirstAidBox",
    "FireAlarm",
    "SafetySwitchPanel",
    "EmergencyPhone",
    "FireExtinguisher"
]

CLASS_COLORS = {
    0: (255, 0, 0),      # OxygenTank - Red
    1: (0, 255, 0),      # NitrogenTank - Green
    2: (0, 0, 255),      # FirstAidBox - Blue
    3: (255, 255, 0),    # FireAlarm - Yellow
    4: (255, 0, 255),    # SafetySwitchPanel - Magenta
    5: (0, 255, 255),    # EmergencyPhone - Cyan
    6: (255, 165, 0)     # FireExtinguisher - Orange
}

CLASS_EMOJIS = {
    "OxygenTank": "🫧",
    "NitrogenTank": "❄️",
    "FirstAidBox": "🩹",
    "FireAlarm": "🚨",
    "SafetySwitchPanel": "⚡",
    "EmergencyPhone": "📞",
    "FireExtinguisher": "🧯"
}

# ============================================================================
# GLASSMORPHISM + PASTEL UI THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Pastel gradient background */
    .main {
        background: linear-gradient(135deg, #e8f5e9 0%, #bbdefb 100%);
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(232, 245, 233, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Headers with gradient */
    h1, h2, h3, h4, h5, h6 {
        color: #1565c0 !important;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hero section with glassmorphism */
    .hero-section {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.6), rgba(129, 212, 250, 0.6));
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        animation: heroFadeIn 1s ease-out;
    }
    
    @keyframes heroFadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .hero-logo {
        font-size: 4rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        color: white !important;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .hero-subtitle {
        color: white;
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Pastel buttons */
    .stButton>button {
        background: linear-gradient(135deg, #64b5f6 0%, #42a5f5 100%);
        color: white;
        border-radius: 15px;
        height: 3.5em;
        width: 100%;
        font-size: 1.1em;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(66, 165, 245, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(66, 165, 245, 0.5);
    }
    
    /* Metric cards with glassmorphism */
    .metric-glass-card {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.7), rgba(129, 212, 250, 0.7));
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-glass-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 40px rgba(100, 181, 246, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert boxes with glassmorphism */
    .glass-alert-success {
        background: rgba(200, 230, 201, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #66bb6a;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 187, 106, 0.2);
    }
    
    .glass-alert-warning {
        background: rgba(255, 245, 157, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #ffa726;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 167, 38, 0.2);
    }
    
    .glass-alert-danger {
        background: rgba(255, 205, 210, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #ef5350;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(239, 83, 80, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .glass-alert-info {
        background: rgba(187, 222, 251, 0.7);
        backdrop-filter: blur(10px);
        border-left: 5px solid #42a5f5;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(66, 165, 245, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #64b5f6 0%, #81d4fa 100%);
        animation: progressGlow 2s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 10px rgba(100, 181, 246, 0.5); }
        50% { box-shadow: 0 0 20px rgba(129, 212, 250, 0.7); }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(227, 242, 253, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 15px 15px 0 0;
        color: #1565c0;
        padding: 12px 24px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(227, 242, 253, 0.8);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(100, 181, 246, 0.8), rgba(129, 212, 250, 0.8));
        color: white;
        box-shadow: 0 4px 15px rgba(100, 181, 246, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(227, 242, 253, 0.5);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(100, 181, 246, 0.6);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #42a5f5;
        background: rgba(227, 242, 253, 0.7);
    }
    
    /* Footer */
    .footer {
        background: rgba(232, 245, 233, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Badge styling */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #64b5f6, #81d4fa);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLO model with caching - uses relative path"""
    script_dir = Path(__file__).parent
    model_path = script_dir / "weights" / "best.pt"
    
    fallback_paths = [
        script_dir / "runs" / "detect" / "train" / "weights" / "best.pt",
        script_dir / "best.pt",
        Path("weights") / "best.pt",
        Path("best.pt")
    ]
    
    if model_path.exists():
        try:
            model = YOLO(str(model_path))
            st.success(f"✅ Model loaded from: {model_path}")
            return model
        except Exception as e:
            st.warning(f"⚠️ Error loading from {model_path}: {e}")
    
    for fallback in fallback_paths:
        if fallback.exists():
            try:
                model = YOLO(str(fallback))
                st.success(f"✅ Model loaded from: {fallback}")
                return model
            except Exception as e:
                continue
    
    st.error("❌ Model file 'best.pt' not found!")
    st.info("""
    **Please ensure the model file is in one of these locations:**
    - `weights/best.pt`
    - `runs/detect/train/weights/best.pt`
    - `best.pt` (same directory as app)
    """)
    return None

def draw_predictions(image, results, visible_classes):
    """Draw bounding boxes and labels on image"""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    detections = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = SAFETY_CLASSES[cls]
            
            if class_name not in visible_classes:
                continue
            
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            
            label = f"{class_name}: {conf:.2%}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            cv2.putText(img_bgr, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, detections

def analyze_safety(detections):
    """Analyze detected objects and provide safety insights"""
    detected_classes = set([d['class'] for d in detections])
    critical_equipment = {'FireExtinguisher', 'FireAlarm', 'FirstAidBox', 'EmergencyPhone'}
    missing_critical = critical_equipment - detected_classes
    
    return {
        'detected_classes': detected_classes,
        'missing_critical': missing_critical,
        'total_objects': len(detections)
    }

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # ========================================================================
    # HERO SECTION
    # ========================================================================
    st.markdown("""
    <div class="hero-section">
        <div class="hero-logo">🛰️</div>
        <h1 class="hero-title">Space Station Safety Detection</h1>
        <p class="hero-subtitle">
            AI-Powered Safety Equipment Recognition for Space Operations
        </p>
        <div style="margin-top: 1.5rem;">
            <span class="tech-badge">🤖 YOLOv8</span>
            <span class="tech-badge">⚡ CUDA Accelerated</span>
            <span class="tech-badge">🎯 88% mAP</span>
            <span class="tech-badge">🚀 ~20ms Inference</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance Banner
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-label">mAP@0.5</div>
            <div class="metric-value">0.88</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col2:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">0.92</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col3:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">0.82</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col4:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-label">Inference</div>
            <div class="metric-value">~20ms</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics source note
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.85rem; font-style: italic; margin-top: 0.5rem;">
    *Metrics reported on validation set (YOLOv8, imgsz=800, conf=0.25)*
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # NAVIGATION TABS
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home",
        "🔍 Detect",
        "💡 Why This Matters",
        "ℹ️ About"
    ])
    
    # ========================================================================
    # TAB 1: HOME
    # ========================================================================
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #42a5f5 !important;">🎯</h2>
                <h4 style="text-align: center;">The Challenge</h4>
                <p style="text-align: center; color: #666;">
                    Space station safety requires constant monitoring of 7 critical equipment types across varying lighting conditions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #66bb6a !important;">🤖</h2>
                <h4 style="text-align: center;">Our Solution</h4>
                <p style="text-align: center; color: #666;">
                    Real-time AI detection using synthetic training data from Falcon digital twin simulation platform.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: #ffa726 !important;">🚀</h2>
                <h4 style="text-align: center;">Impact</h4>
                <p style="text-align: center; color: #666;">
                    88% mAP accuracy with 20ms inference enables autonomous inspection and crew safety assurance.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # How It Works
        with st.expander("ℹ️ How It Works", expanded=True):
            st.markdown("""
            <div class="glass-alert-info">
            <h4>🔬 Detection Pipeline</h4>
            <ol>
                <li><strong>Upload Image:</strong> Select an image from the space station containing safety equipment</li>
                <li><strong>AI Processing:</strong> YOLOv8 model analyzes the image in real-time (~20ms)</li>
                <li><strong>Object Detection:</strong> Model identifies and localizes 7 types of safety equipment</li>
                <li><strong>Results Display:</strong> Bounding boxes with confidence scores are overlaid on the image</li>
                <li><strong>Safety Analysis:</strong> System checks for missing critical equipment</li>
                <li><strong>Export:</strong> Download annotated images for audit or reporting</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: DETECT
    # ========================================================================
    with tab2:
        # Sidebar controls
        with st.sidebar:
            st.markdown("### ⚙️ Detection Settings")
            
            conf_threshold = st.slider(
                "🎚️ Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Minimum confidence score for detections"
            )
            
            st.info("""
            **📊 Optimal Threshold Selection**
            
            Default: **0.25** (data-driven)
            
            Selected from F1-confidence curve analysis to maximize balance between recall and precision in safety-critical scenarios where missing equipment is more costly than false positives.
            """)
            
            st.markdown("---")
            st.markdown("#### 🎨 Visibility Controls")
            st.markdown("Toggle classes to show/hide:")
            
            visible_classes = []
            for cls_name in SAFETY_CLASSES:
                emoji = CLASS_EMOJIS.get(cls_name, "📦")
                if st.checkbox(f"{emoji} {cls_name}", value=True, key=f"vis_{cls_name}"):
                    visible_classes.append(cls_name)
            
            st.markdown("---")
            upload_mode = st.radio(
                "📊 Upload Mode:",
                ["Single Image", "Batch Images"]
            )
        
        # Load model
        model = load_model()
        
        if model is None:
            st.stop()
        
        st.markdown("### 📤 Upload Images for Detection")
        
        uploaded_files = st.file_uploader(
            "Choose space station images (JPG, PNG)" if upload_mode == "Single Image" else "Upload multiple images for batch processing",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images to detect safety equipment"
        )
        
        if uploaded_files:
            if upload_mode == "Batch Images":
                st.markdown(f"""
                <div class="glass-alert-info">
                    <strong>📦 Batch Mode:</strong> Processing {len(uploaded_files)} images
                </div>
                """, unsafe_allow_html=True)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown("---")
                st.markdown(f"#### 🖼️ Image {idx + 1}: {uploaded_file.name}")
                
                try:
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                    
                    if len(img_array.shape) == 2:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                except Exception as e:
                    st.error(f"❌ Image reading failed: {e}")
                    continue
                
                # Run detection with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("🔍 Analyzing safety equipment...")
                
                start_time = time.time()
                
                for i in range(100):
                    time.sleep(0.005)
                    progress_bar.progress(i + 1)
                
                try:
                    results = model.predict(
                        img_array,
                        conf=conf_threshold,
                        iou=0.6,
                        max_det=20,
                        imgsz=800,
                        device=0,
                        verbose=False
                    )
                    
                    inference_time = time.time() - start_time
                    progress_bar.empty()
                    status_text.empty()
                    
                    result_image, detections = draw_predictions(image, results, visible_classes)
                    
                except Exception as e:
                    st.error(f"❌ Detection failed: {e}")
                    continue
                
                # Display results
                if detections:
                    st.success(f"✅ Found {len(detections)} equipment item(s)")
                    if len(detections) > 3:
                        st.balloons()
                else:
                    st.info("ℹ️ No equipment detected in this image")
                
                # Side-by-side comparison
                col_orig, col_detect = st.columns(2)
                
                with col_orig:
                    st.markdown("##### 📷 Original Image")
                    st.image(img_array, use_container_width=True)
                
                with col_detect:
                    st.markdown("##### ✨ Detection Results")
                    st.image(result_image, use_container_width=True)
                
                # Metrics dashboard
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-glass-card">
                        <div class="metric-label">Detections</div>
                        <div class="metric-value">{len(detections)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                    st.markdown(f"""
                    <div class="metric-glass-card">
                        <div class="metric-label">Avg Confidence</div>
                        <div class="metric-value">{avg_conf:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-glass-card">
                        <div class="metric-label">Inference Time</div>
                        <div class="metric-value">{inference_time*1000:.0f}ms</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Safety analysis
                safety_info = analyze_safety(detections)
                
                if safety_info['missing_critical']:
                    st.markdown(f"""
                    <div class="glass-alert-danger">
                        <strong>⚠️ Safety Alert:</strong> Missing critical equipment detected!<br>
                        <strong>Not found:</strong> {', '.join(safety_info['missing_critical'])}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="glass-alert-success">
                        <strong>✅ All critical safety equipment detected!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence distribution
                if detections:
                    st.markdown("##### 📊 Confidence Distribution")
                    
                    conf_values = [d['confidence'] for d in detections]
                    fig = px.histogram(
                        x=conf_values,
                        nbins=10,
                        labels={'x': 'Confidence', 'y': 'Count'},
                        color_discrete_sequence=['#64b5f6']
                    )
                    fig.add_vline(
                        x=conf_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {conf_threshold}"
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed detection breakdown
                if detections:
                    st.markdown("##### 📋 Detection Breakdown")
                    
                    for i, det in enumerate(detections, 1):
                        emoji = CLASS_EMOJIS.get(det['class'], "📦")
                        
                        col_det1, col_det2 = st.columns([2, 1])
                        with col_det1:
                            st.markdown(f"**{emoji} {det['class']} #{i}**")
                        with col_det2:
                            st.progress(det['confidence'], text=f"{det['confidence']:.1%}")
                
                # Download section
                st.markdown("##### 💾 Export Results")
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    result_pil = Image.fromarray(result_image)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_img_{idx}"
                    )
                
                with col_d2:
                    if detections:
                        df = pd.DataFrame([{
                            'Detection_ID': i+1,
                            'Class': d['class'],
                            'Confidence': f"{d['confidence']:.2%}",
                            'BBox': f"({d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]})"
                        } for i, d in enumerate(detections)])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv,
                            file_name=f"report_{uploaded_file.name}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"download_csv_{idx}"
                        )
    
    # ========================================================================
    # TAB 3: WHY THIS MATTERS
    # ========================================================================
    with tab3:
        st.markdown("### 💡 Why This Matters")
        
        with st.expander("🚀 Real-World Impact", expanded=True):
            st.markdown("""
            <div class="glass-alert-info">
            <h4>Real-World Applications</h4>
            <ul>
                <li><strong>🛡️ Safety Assurance:</strong> Prevents missed safety equipment in emergency scenarios, reducing risk to crew members in life-threatening situations</li>
                <li><strong>🌓 Robust Detection:</strong> Works reliably in cluttered environments and varying lighting conditions (shadows, glare, darkness) common in space stations</li>
                <li><strong>🤖 Integration Ready:</strong> Can be deployed on autonomous inspection drones, CCTV monitoring systems, or crew handheld devices for continuous monitoring</li>
                <li><strong>📈 Continuous Improvement:</strong> Falcon synthetic data enables rapid model updates as new equipment or configurations are introduced without costly real-world data collection</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🔗 Potential System Integrations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1565c0 !important;">🛰️ Autonomous Inspection</h4>
                <ul style="color: #666;">
                    <li>Real-time monitoring via station cameras</li>
                    <li>Automated safety compliance checks</li>
                    <li>Alert generation for missing equipment</li>
                    <li>Integration with station management systems</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1565c0 !important;">🤖 Robotic Inspection</h4>
                <ul style="color: #666;">
                    <li>Drone-based automated surveys</li>
                    <li>GPS-tagged equipment mapping</li>
                    <li>Hard-to-reach area monitoring</li>
                    <li>Emergency damage assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1565c0 !important;">👨‍🚀 Crew Assistance</h4>
                <ul style="color: #666;">
                    <li>Real-time crew safety alerts</li>
                    <li>AR-assisted equipment location</li>
                    <li>Training and compliance validation</li>
                    <li>Emergency response optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1565c0 !important;">📊 Mission Control</h4>
                <ul style="color: #666;">
                    <li>Centralized safety dashboard</li>
                    <li>Historical equipment tracking</li>
                    <li>Predictive maintenance scheduling</li>
                    <li>Compliance reporting automation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Performance Impact")
        
        col1, col2, col3 = st.columns(3)
        
        impact_metrics = [
            ("88%", "Accuracy", "mAP@0.5 validated"),
            ("20ms", "Inference", "Real-time capable"),
            ("7", "Classes", "Equipment types")
        ]
        
        for col, (value, label, desc) in zip([col1, col2, col3], impact_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-glass-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <p style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    with tab4:
        st.markdown("### ℹ️ About the Project")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #1565c0 !important;">🤖 System Overview</h4>
                <p style="color: #666; line-height: 1.8;">
                This Space Station Safety Detection system leverages state-of-the-art YOLOv8 
                architecture to detect and classify 7 critical safety equipment types in real-time. 
                The model was trained using synthetic data from Duality AI's Falcon digital twin 
                simulation platform, enabling robust performance across varying lighting conditions 
                and camera angles.
                </p>
                <p style="color: #666; line-height: 1.8;">
                The system achieves 88% mAP@0.5 with 92% precision and 82% recall, processing 
                images at ~20ms per frame. It's designed for deployment on edge devices, monitoring 
                dashboards, and autonomous inspection systems.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-glass-card">
                <h4 style="color: white !important;">📊 Key Stats</h4>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">88%</div>
                    <div class="metric-label">mAP@0.5</div>
                </div>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">92%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div style="margin: 1rem 0;">
                    <div class="metric-value">20ms</div>
                    <div class="metric-label">Inference</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🛠️ Technology Stack")
        
        tech_categories = {
            "AI/ML": ["YOLOv8", "PyTorch", "CUDA", "OpenCV"],
            "Frontend": ["Streamlit", "Plotly", "Pandas", "NumPy"],
            "Training": ["Falcon Platform", "Synthetic Data", "Augmentation"],
            "Deployment": ["Edge Devices", "Cloud Ready", "API Integration"]
        }
        
        cols = st.columns(4)
        for col, (category, techs) in zip(cols, tech_categories.items()):
            with col:
                tech_list = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
                st.markdown(f"""
                <div class="glass-card">
                    <h5 style="color: #1565c0 !important; text-align: center;">{category}</h5>
                    <div style="text-align: center; margin-top: 1rem;">
                        {tech_list}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Equipment Classes")
        
        st.markdown("##### Detected Safety Equipment:")
        
        cols = st.columns(7)
        for col, (cls_name, emoji) in zip(cols, CLASS_EMOJIS.items()):
            with col:
                color_idx = SAFETY_CLASSES.index(cls_name)
                color = CLASS_COLORS[color_idx]
                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(f"""
                <div style="background: {color_hex}; padding: 15px; border-radius: 15px; text-align: center; color: white; font-weight: bold; margin: 5px;">
                    <div style="font-size: 2rem;">{emoji}</div>
                    <div style="font-size: 0.75rem; margin-top: 5px;">{cls_name}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Model Performance Details")
        
        metrics_data = {
            'Metric': ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'F1-Score'],
            'Score': [0.92, 0.82, 0.88, 0.65, 0.87]
        }
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_metrics['Metric'],
            y=df_metrics['Score'],
            marker_color=['#64b5f6', '#81d4fa', '#42a5f5', '#1e88e5', '#1565c0'],
            text=df_metrics['Score'].round(2),
            textposition='auto',
        ))
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_range=[0, 1],
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="glass-alert-success">
            <strong>✅ Production Ready:</strong> The model is optimized for deployment with 
            efficient inference times and high accuracy across diverse conditions.
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <div class="hero-logo" style="font-size: 2.5rem;">🛰️</div>
        <h3 style="color: #1565c0 !important; margin: 1rem 0;">Space Station Safety Detection</h3>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 1.5rem;">
            AI-Powered Safety Equipment Recognition for Space Operations
        </p>
        <div style="margin: 1.5rem 0;">
            <span class="tech-badge">🤖 YOLOv8</span>
            <span class="tech-badge">⚡ CUDA Accelerated</span>
            <span class="tech-badge">🎯 88% mAP</span>
            <span class="tech-badge">🚀 20ms Inference</span>
        </div>
        <p style="opacity: 0.7; font-size: 0.95rem; margin-top: 2rem;">
            Duality AI Space Station Challenge: Safety Object Detection #2<br>
            Powered by YOLOv8 + Falcon Synthetic Data | Trained on 18,717+ Images
        </p>
        <p style="opacity: 0.8; font-size: 1rem; margin-top: 1rem; color: #1565c0; font-weight: bold;">
            🚀 Designed for deployment on edge devices, monitoring dashboards, and autonomous inspection systems
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    if 'processed_images' not in st.session_state:
        st.session_state['processed_images'] = []
    
    main()