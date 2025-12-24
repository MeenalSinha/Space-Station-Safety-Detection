# 🛰️ Space Station Safety Detection System
### Duality AI Space Station Challenge: Safety Object Detection #2

An end-to-end **real-time AI-powered safety equipment detection system** built using **YOLOv8** to identify and monitor critical safety equipment in space station environments. The system leverages synthetic training data from Duality AI's Falcon digital twin simulation platform to achieve robust performance across varying lighting conditions and camera angles.

---

## 🚀 Project Overview

Space station safety requires constant monitoring of critical emergency equipment across multiple modules and varying environmental conditions. Manual inspection is resource-intensive and prone to human error, especially during emergency situations or crew fatigue.

This project automates safety compliance monitoring by detecting key safety equipment in real-time from images captured by station cameras, autonomous inspection drones, or crew handheld devices.

### 🎯 Detected Safety Equipment Classes
- **OxygenTank** 🫧 - Breathable air supply reserves
- **NitrogenTank** ❄️ - Pressurization and cooling systems
- **FirstAidBox** 🩹 - Medical emergency supplies
- **FireAlarm** 🚨 - Fire detection and warning systems
- **SafetySwitchPanel** ⚡ - Emergency power and control systems
- **EmergencyPhone** 📞 - Direct communication systems
- **FireExtinguisher** 🧯 - Fire suppression equipment

---

## 🧠 Model Architecture & Approach

- **Architecture:** YOLOv8 (Ultralytics) - State-of-the-art object detection
- **Task:** Multi-class real-time object detection
- **Training Data:** Synthetic images from Falcon digital twin platform (1767 training images, 336 validation images)
- **Training Strategy:** Transfer learning with pretrained weights
- **Input Resolution:** 800px (optimized for accuracy-speed balance)
- **Inference Speed:** ~20ms per frame (real-time capable)
- **Deployment Target:** Edge devices, monitoring dashboards, autonomous systems

### 🔬 Technical Specifications
- **Loss Functions:** Box regression, Classification, Distribution Focal Loss (DFL)
- **Optimizer:** AdamW with learning rate scheduling
- **Confidence Threshold:** 0.25 (F1-optimized, data-driven selection)
- **IOU Threshold:** 0.6 (NMS post-processing)
- **Mixed Precision:** Enabled (AMP for faster training)

---

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | **92.4%** | Percentage of correct positive predictions |
| **Recall** | **82.4%** | Percentage of actual positives detected |
| **mAP@0.5 (Val)** | **88.1%** | Mean Average Precision at 50% IOU (validation set) |
| **mAP@0.5 (Test)** | **~73%** | Mean Average Precision on held-out test set |
| **mAP@0.5:0.95** | **65.0%** | Mean Average Precision across IOU thresholds |
| **F1-Score** | **87.0%** | Harmonic mean of precision and recall |
| **Inference Time** | **~20ms** | Processing time per frame |

*Primary metrics reported on validation set. Final test-set performance achieved ~73% mAP@0.5, demonstrating reasonable generalization with expected domain gap between validation and test distributions.*

### 📈 Key Strengths
- ✅ **High Precision:** 92% - Minimizes false alarms
- ✅ **Robust Recall:** 82% - Catches most safety equipment
- ✅ **Real-time Performance:** 20ms inference enables live monitoring
- ✅ **Lighting Invariance:** Trained on synthetic data with varied lighting
- ✅ **Cluttered Environment Performance:** Handles complex station interiors

---

## 🗂️ Repository Structure

```
space-station-safety-detection/
│
├── app.py                      # Streamlit web application (main demo)
├── requirements.txt            # Python dependencies
├── packages.txt               # System dependencies for Streamlit Cloud
│
├── weights/
│   └── best.pt                # Trained YOLOv8 model weights
│
├── train/                     # Training dataset
│   ├── images/               # Training images
│   └── labels/               # YOLO-format annotations
│
├── val/                      # Validation dataset
│   ├── images/              # Validation images
│   └── labels/              # YOLO-format annotations
│
├── test/                     # Test dataset
│   └── images/              # Test images (unlabeled)
│
├── predictions/              # Inference outputs
│   ├── images/              # Annotated images with bounding boxes
│   └── labels/              # YOLO-format prediction files
│
├── runs/                     # Training artifacts
│   └── detect/
│       ├── train/           # Training logs, plots, metrics
│       │   ├── weights/    # Model checkpoints
│       │   ├── confusion_matrix.png
│       │   ├── F1_curve.png
│       │   ├── P_curve.png
│       │   ├── R_curve.png
│       │   └── results.csv
│       └── val/            # Validation results
│
└── README.md               # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Git

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/space-station-safety-detection.git
cd space-station-safety-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **For system-level dependencies (Linux)**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Streamlit Cloud Deployment

The `packages.txt` file is included for automatic system dependency installation on Streamlit Cloud:
```
libgl1-mesa-glx
libglib2.0-0
```

---

## 🚀 Usage

### 1. Run Web Application (Recommended)

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

**Features:**
- 📤 Upload single or batch images
- 🎚️ Adjustable confidence threshold
- 🎨 Toggle visibility for specific equipment classes
- 📊 Real-time confidence distribution visualization
- 💾 Download annotated images and CSV reports
- ⚠️ Automated safety compliance alerts

### 2. Batch Inference (Command Line)

For programmatic batch processing:

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('weights/best.pt')

# Run inference
results = model.predict(
    source='test/images/',
    conf=0.25,
    iou=0.6,
    imgsz=800,
    save=True,
    project='predictions'
)
```

### 3. Model Training (Advanced)

To retrain or fine-tune the model:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8s.pt')

# Train
results = model.train(
    data='data.yaml',      # Dataset configuration
    epochs=50,
    imgsz=800,
    batch=16,
    device=0,              # GPU device
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    conf=0.25
)
```

---

## 🎯 Application Scenarios

### 🛰️ Autonomous Inspection Systems
- **Real-time monitoring** via station CCTV networks
- **Automated compliance checks** during shift changes
- **Alert generation** for missing or damaged equipment
- **Integration** with station management systems

### 🤖 Robotic Inspection Drones
- **Autonomous patrol routes** through station modules
- **GPS-tagged equipment mapping** for inventory management
- **Hard-to-reach area monitoring** (ventilation shafts, external modules)
- **Emergency damage assessment** post-incident

### 👨‍🚀 Crew Assistance Tools
- **AR-guided equipment location** on handheld devices
- **Real-time safety alerts** during operations
- **Training validation** for new crew members
- **Emergency response optimization** with equipment localization

### 📊 Mission Control Dashboards
- **Centralized safety monitoring** across all modules
- **Historical equipment tracking** and analytics
- **Predictive maintenance scheduling** based on equipment status
- **Compliance reporting automation** for safety audits

---

## 📈 Model Performance Analysis

### Training Insights

The model was trained for up to 50 epochs with validation-based monitoring. Key observations:

- **Convergence:** Model stabilized around epoch 35-40
- **Optimal threshold:** 0.25 confidence selected via F1-curve analysis
- **Class balance:** Most classes achieve strong (>75%) individual mAP@0.5 on validation
- **Robustness:** Performance maintained across lighting variations (synthetic data advantage)
- **Generalization:** Test performance (~73% mAP@0.5) shows expected domain gap from validation (88.1%)

### Validation Results

Per-class performance breakdown:

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| OxygenTank | 0.94 | 0.86 | 0.91 |
| NitrogenTank | 0.93 | 0.81 | 0.88 |
| FirstAidBox | 0.91 | 0.84 | 0.89 |
| FireAlarm | 0.95 | 0.83 | 0.90 |
| SafetySwitchPanel | 0.89 | 0.78 | 0.84 |
| EmergencyPhone | 0.92 | 0.82 | 0.88 |
| FireExtinguisher | 0.93 | 0.83 | 0.87 |

*All metrics measured on validation set (imgsz=800, conf=0.25)*

### Confusion Matrix Insights

- **Low false positive rate:** Minimal confusion between classes
- **Challenging classes:** SafetySwitchPanel shows slightly lower recall (78%) due to visual similarity with other panels
- **Strong performers:** FireAlarm and OxygenTank achieve highest precision (95%, 94%)

---

## 🔬 Technical Deep Dive

### Dataset Characteristics

- **Training Images:** 1,767 synthetic images
- **Validation Images:** 336 synthetic images
- **Test Images:** 1,408 images
- **Source:** Falcon digital twin simulation platform (Duality AI)
- **Lighting Conditions:** Day, night, shadow, glare, emergency lighting
- **Camera Angles:** Multiple perspectives per scene
- **Clutter Levels:** Varied from clean to highly cluttered environments
- **Annotation Quality:** Pixel-perfect bounding boxes from simulation

### Model Optimization

**Confidence Threshold Selection:**
The default threshold of 0.25 was selected through F1-confidence curve analysis to maximize the balance between precision and recall. In safety-critical applications, this threshold favors **recall** (catching all equipment) over precision (avoiding false alarms), as missing equipment is more costly than false positives.

**Image Resolution Trade-off:**
- **640px:** Faster inference (~15ms) but 3-5% lower mAP
- **800px:** Optimal balance at ~20ms with ~88% validation mAP ✅
- **1024px:** Marginal gains (+1-2% mAP) but 2x slower inference

**Anchor Box Optimization:**
YOLOv8 uses anchor-free detection, eliminating the need for manual anchor tuning and improving generalization to varied object sizes.

---

## 🚨 Safety Compliance Features

The application includes intelligent safety analysis:

### Critical Equipment Detection
Automatically identifies missing critical equipment:
- FireExtinguisher
- FireAlarm
- FirstAidBox
- EmergencyPhone

### Alert System
- ✅ **Green Alert:** All critical equipment detected
- ⚠️ **Red Alert:** Missing critical equipment with specific items listed

### Export & Reporting
- **Annotated Images:** Download visual evidence with bounding boxes
- **CSV Reports:** Export detection data with confidence scores and coordinates
- **Audit Trail:** Timestamp and metadata for compliance documentation

---

## 🎨 User Interface Features

### Glassmorphism + Pastel Design
- Modern, accessible UI with soft gradients and blur effects
- High contrast for critical alerts (red/yellow/green)
- Animated transitions and hover effects for engagement

### Interactive Controls
- **Confidence Slider:** Adjust detection sensitivity in real-time
- **Class Toggle:** Show/hide specific equipment types
- **Batch Processing:** Upload and process multiple images
- **Real-time Metrics:** Live confidence distribution charts

### Data Visualization
- **Plotly Charts:** Interactive confidence histograms
- **Progress Indicators:** Real-time processing feedback
- **Metric Cards:** Key performance indicators (detections, confidence, time)

---

## 🔄 Future Enhancements

### Phase 2 Development
- [ ] **Video stream processing** for live CCTV monitoring
- [ ] **Multi-camera fusion** for 3D equipment mapping
- [ ] **Temporal tracking** to detect equipment movement/tampering
- [ ] **Integration APIs** for station management systems

### Model Improvements
- [ ] **YOLOv9/v10 migration** for improved accuracy
- [ ] **Quantization** for edge device deployment (INT8)
- [ ] **Active learning pipeline** for continuous improvement
- [ ] **Anomaly detection** for damaged equipment

### Advanced Features
- [ ] **AR overlay system** for crew handheld devices
- [ ] **Predictive maintenance** using equipment status trends
- [ ] **Natural language alerts** ("Missing fire extinguisher in Module A")
- [ ] **Multi-station deployment** with centralized dashboard

---

## 📄 Technical Documentation

### Model Card

**Model Name:** SpaceStation-Safety-YOLOv8  
**Version:** 1.0  
**Architecture:** YOLOv8s (small variant)  
**Parameters:** ~11M  
**Training Data:** Falcon synthetic dataset  
**Intended Use:** Safety equipment detection in space station environments  
**Limitations:** Performance may degrade on real-world images if significantly different from Falcon simulation  

### Ethical Considerations

- **Privacy:** No human detection or identification
- **Transparency:** All metrics and limitations documented
- **Fairness:** Equipment detection is class-agnostic (no bias)
- **Safety:** High recall prioritized to minimize missed equipment

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black app.py
isort app.py
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Duality AI** for providing the Falcon digital twin platform and synthetic training data
- **Ultralytics** for the exceptional YOLOv8 framework
- **Streamlit** for the intuitive web app framework
- **Duality AI Space Station Challenge** organizers and community

---

## 📧 Contact

**Author:** Meenal Sinha  
**Project Link:** [https://github.com/yourusername/space-station-safety-detection](https://github.com/yourusername/space-station-safety-detection)  
**Demo App:** [https://space-station-safety.streamlit.app](https://space-station-safety.streamlit.app)

---

## 🌟 Citation

If you use this project in your research or application, please cite:

```bibtex
@software{space_station_safety_2025,
  author = {Sinha, Meenal},
  title = {Space Station Safety Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/space-station-safety-detection}
}
```

---

### 🛰️ Built for the future of space safety

**Real-time AI • 88% Validation mAP@50 • 20ms Inference**

*Powered by YOLOv8 + Falcon Synthetic Data*
