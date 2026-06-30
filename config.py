# config.py

# ==============================
# MODEL CONFIGURATION
# ==============================

# Use YOLO26 (change to n/s/m/l/x based on need)
YOLO_MODEL_PATH = "models/yolo26n.pt"

# ==============================
# VIDEO SOURCE
# ==============================

# Use 0 for USB/Webcam or replace with your DroidCam/IP Webcam URL
VIDEO_PATH = 0

# ==============================
# CAMERA INTRINSICS (IMPORTANT)
# Replace with real calibrated values later
# ==============================

FX = 800.0
FY = 800.0
CX = 320.0
CY = 240.0

# ==============================
# OUTPUT
# ==============================

OUTPUT_PICKLE = "outputs/output.pkl"
OUTPUT_JSON = "outputs/output.json"
