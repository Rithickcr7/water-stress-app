import streamlit as st
import cv2
import numpy as np
from reportlab.pdfgen import canvas
import tempfile
import os

# ---------------- UI CONFIG ----------------

logo_path = "logo.png"

st.set_page_config(
    page_title="Crop Water Stress Detector",
    page_icon="🌱",
    layout="wide"
)

# Center logo
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(logo_path, width=150)

st.markdown("""
# 🌱 Crop Water Stress Detection System
AI-based Leaf Analysis for Smart Irrigation
""")

# ---------------- INPUT ----------------

colA, colB = st.columns(2)

with colA:
    image = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

with colB:
    camera_image = st.camera_input("Capture Leaf Image")

if camera_image is not None:
    image = camera_image


# ---------------- LEAF DETECTION ----------------

def detect_leaf(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20,30,30])
    upper_green = np.array([100,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    leaf = cv2.bitwise_and(img,img,mask=mask)

    return leaf, mask


# ---------------- GREENNESS INDEX ----------------

def calculate_exg(img):

    img = img.astype(np.float32)

    b,g,r = cv2.split(img)

    exg = 2*g - r - b

    return exg


# ---------------- CHLOROPHYLL ESTIMATION ----------------

def chlorophyll_value(exg):

    return 0.45 * exg + 25


# ---------------- TEMPERATURE ESTIMATION ----------------

def estimate_leaf_temperature(exg):

    temp = 35 - (exg * 0.05)

    return np.clip(temp,24,38)


# ---------------- HEATMAP ----------------

def create_heatmap(exg_matrix, mask):

    exg_norm = cv2.normalize(exg_matrix,None,0,255,cv2.NORM_MINMAX)

    exg_norm = exg_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(exg_norm, cv2.COLORMAP_TURBO)

    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    return heatmap


# ---------------- LEGEND ----------------

def create_color_legend():

    gradient = np.linspace(0,255,256).astype(np.uint8)

    gradient = np.tile(gradient,(40,1))

    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)

    return legend


# ---------------- STRESS LOGIC ----------------

def stress_logic(temp, chl):

    if chl > 45 and temp < 30:
        return "Healthy Plant", "No irrigation needed"

    elif 30 <= chl <= 45 or 30 <= temp <= 35:
        return "Moderate Stress", "Irrigation recommended"

    else:
        return "High Stress", "Immediate irrigation required"


# ---------------- PDF ----------------

def generate_pdf(status, chl, temp):

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    c = canvas.Canvas(temp_file.name)

    c.drawString(100,750,"Crop Water Stress Report")
    c.drawString(100,720,f"Stress Status: {status}")
    c.drawString(100,700,f"Estimated Chlorophyll: {chl}")
    c.drawString(100,680,f"Estimated Leaf Temperature: {temp} °C")

    c.drawString(100,640,"Recommendation:")

    if "Healthy" in status:
        c.drawString(100,620,"No irrigation needed")

    elif "Moderate" in status:
        c.drawString(100,620,"Irrigation recommended")

    else:
        c.drawString(100,620,"Immediate irrigation required")

    c.save()

    return temp_file.name


# ---------------- MAIN ----------------

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()),dtype=np.uint8)
    img = cv2.imdecode(file_bytes,1)

    img = cv2.GaussianBlur(img,(5,5),0)

    leaf, mask = detect_leaf(img)

    exg_matrix = calculate_exg(leaf)

    leaf_pixels = exg_matrix[mask > 0]

    exg_value = np.mean(leaf_pixels)

    chl_value = chlorophyll_value(exg_value)

    leaf_temp = estimate_leaf_temperature(exg_value)

    heatmap = create_heatmap(exg_matrix, mask)

    legend = create_color_legend()

    # -------- DISPLAY --------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Leaf")
        st.image(leaf, channels="BGR")

    with col2:
        st.subheader("Stress Heatmap")
        st.image(heatmap, channels="BGR")

        st.markdown("**Stress Scale**")
        st.image(legend, channels="BGR")

        st.info("Blue/Green = Healthy | Yellow/Red = High Stress")

    st.markdown("### 📊 Analysis Results")

    c1,c2,c3 = st.columns(3)

    c1.metric("Greenness Index", round(exg_value,2))
    c2.metric("Chlorophyll Estimate", round(chl_value,2))
    c3.metric("Estimated Leaf Temp (°C)", round(leaf_temp,2))

    if st.button("Analyze Plant Stress"):

        result, suggestion = stress_logic(leaf_temp, chl_value)

        st.success(result)

        st.info("Recommendation: " + suggestion)

        pdf_file = generate_pdf(
            result,
            round(chl_value,2),
            round(leaf_temp,2)
        )

        with open(pdf_file,"rb") as f:

            st.download_button(
                "Download Farmer PDF Report",
                f,
                file_name="crop_stress_report.pdf"
            )
