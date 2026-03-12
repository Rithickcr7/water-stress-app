import streamlit as st
import cv2
import numpy as np
from reportlab.pdfgen import canvas
import tempfile
import os

# ---------------- UI CONFIG ----------------

logo_path = "logo.png"

st.set_page_config(
    page_title="Banana Leaf Water Stress Detector",
    page_icon="🌱",
    layout="wide"
)

# Center logo
if os.path.exists(logo_path):
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.image(logo_path,width=150)

st.markdown("""
# 🍌 Banana Leaf Water Stress Detection System
AI-based image analysis for irrigation decision support
""")

# ---------------- IMAGE INPUT ----------------

colA,colB = st.columns(2)

with colA:
    image = st.file_uploader("Upload Banana Leaf Image",type=["jpg","png","jpeg"])

with colB:
    camera_image = st.camera_input("Capture Banana Leaf Image")

if camera_image is not None:
    image = camera_image


# ---------------- BANANA LEAF SEGMENTATION ----------------

def detect_leaf(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_green = np.array([15,25,25])
    upper_green = np.array([110,255,255])

    mask = cv2.inRange(hsv,lower_green,upper_green)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    leaf_mask = np.zeros(mask.shape,dtype=np.uint8)

    if len(contours) > 0:

        largest_contour = max(contours,key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 2000:

            cv2.drawContours(leaf_mask,[largest_contour],-1,255,-1)

    leaf = cv2.bitwise_and(img,img,mask=leaf_mask)

    return leaf,leaf_mask


# ---------------- GREENNESS INDEX ----------------

def calculate_exg(img):

    img = img.astype(np.float32)

    b,g,r = cv2.split(img)

    exg = 2*g - r - b

    return exg


# ---------------- CHLOROPHYLL ESTIMATION ----------------

def chlorophyll_value(exg):

    return 0.5 * exg + 20


# ---------------- TEMPERATURE ESTIMATION ----------------

def estimate_leaf_temperature(exg):

    temp = 35 - (exg * 0.05)

    return np.clip(temp,24,38)


# ---------------- HEATMAP ----------------

def create_heatmap(exg_matrix):

    exg_norm = cv2.normalize(
        exg_matrix,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    exg_norm = exg_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(
        exg_norm,
        cv2.COLORMAP_JET
    )

    return heatmap


# ---------------- STRESS CLASSIFICATION ----------------

def stress_logic(temp,chl):

    if 24 <= temp <= 30 and chl > 60:
        return "Healthy Plant","No irrigation needed"

    elif (30 < temp <= 34) or (40 <= chl <= 60):
        return "Moderate Stress","Irrigation recommended"

    else:
        return "High Stress","Immediate irrigation required"


# ---------------- PDF REPORT ----------------

def generate_pdf(status,stress,chl,temp):

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    c = canvas.Canvas(temp_file.name)

    c.drawString(100,750,"Banana Crop Water Stress Report")

    c.drawString(100,720,f"Stress Status: {status}")
    c.drawString(100,700,f"Stress Area: {stress}%")
    c.drawString(100,680,f"Estimated Chlorophyll: {chl}")
    c.drawString(100,660,f"Estimated Leaf Temperature: {temp} °C")

    c.drawString(100,620,"Recommendation:")

    if "Healthy" in status:
        c.drawString(100,600,"No irrigation required")

    elif "Moderate" in status:
        c.drawString(100,600,"Provide irrigation soon")

    else:
        c.drawString(100,600,"Immediate irrigation required")

    c.save()

    return temp_file.name


# ---------------- MAIN PROCESS ----------------

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()),dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    img = cv2.GaussianBlur(img,(5,5),0)

    leaf,leaf_mask = detect_leaf(img)

    exg_matrix = calculate_exg(leaf)

    leaf_pixels = exg_matrix[leaf_mask>0]

    exg_value = np.mean(leaf_pixels)

    chl_value = chlorophyll_value(exg_value)

    leaf_temp = estimate_leaf_temperature(exg_value)

    heatmap = create_heatmap(exg_matrix)

    # ---------------- STRESS AREA DETECTION ----------------

    stress_threshold = 55

    stress_mask = (exg_matrix < stress_threshold) & (leaf_mask>0)

    highlight = leaf.copy()

    highlight[stress_mask] = [0,0,255]

    stress_pixels = np.sum(stress_mask)

    leaf_pixels_count = np.sum(leaf_mask>0)

    stress_percent = (stress_pixels / leaf_pixels_count) * 100


    # ---------------- DISPLAY ----------------

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Detected Banana Leaf")
        st.image(leaf,channels="BGR")

    with col2:
        st.subheader("Stress Heatmap")
        st.image(heatmap,channels="BGR")
        st.info("Blue/Green = Healthy | Yellow/Red = Stress")


    st.subheader("Stress Highlight Map")
    st.image(highlight,channels="BGR")


    st.markdown("### 📊 Analysis Results")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Greenness Index",round(exg_value,2))
    c2.metric("Chlorophyll Estimate",round(chl_value,2))
    c3.metric("Leaf Temp Estimate (°C)",round(leaf_temp,2))
    c4.metric("Stress Area (%)",round(stress_percent,2))


    if st.button("Analyze Banana Plant Stress"):

        result,suggestion = stress_logic(leaf_temp,chl_value)

        st.success(result)

        st.info("Recommendation: "+suggestion)

        pdf_file = generate_pdf(
            result,
            round(stress_percent,2),
            round(chl_value,2),
            round(leaf_temp,2)
        )

        with open(pdf_file,"rb") as f:

            st.download_button(
                "Download Farmer PDF Report",
                f,
                file_name="banana_crop_stress_report.pdf"
            )
