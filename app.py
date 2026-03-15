import streamlit as st
import cv2
import numpy as np
from reportlab.pdfgen import canvas
import tempfile

# ---------------- UI ----------------

st.set_page_config(
    page_title="Leaf Stress Detection System",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Crop Water Stress Detection System")
st.write("AI-based RGB leaf analysis for smart irrigation support")

# ---------------- INPUT ----------------

col1, col2 = st.columns(2)

with col1:
    image = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

with col2:
    camera = st.camera_input("Capture Leaf Image")

if camera is not None:
    image = camera

# ---------------- LIGHTING CORRECTION ----------------

def white_balance(img):

    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(result[:,:,1])
    avg_b = np.average(result[:,:,2])

    result[:,:,1] -= ((avg_a-128)*(result[:,:,0]/255.0)*1.1)
    result[:,:,2] -= ((avg_b-128)*(result[:,:,0]/255.0)*1.1)

    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result

# ---------------- LEAF SEGMENTATION ----------------

def segment_leaf(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([20,30,30])
    upper = np.array([95,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    leaf = cv2.bitwise_and(img, img, mask=mask)

    return leaf, mask

# ---------------- GREENNESS INDEX ----------------

def greenness_index(img):

    img = img.astype(np.float32)

    b,g,r = cv2.split(img)

    exg = 2*g - r - b

    return exg

# ---------------- TEMPERATURE MAP ----------------

def temperature_map(exg):

    temp_map = 32.5 - (0.035 * exg)

    temp_map = np.clip(temp_map, 25, 38)

    return temp_map

# ---------------- CHLOROPHYLL ----------------

def chlorophyll_estimate(exg):

    chl = 0.38 * exg + 28

    return np.clip(chl, 20, 60)

# ---------------- HEATMAP ----------------

def temperature_heatmap(img, temp_map, mask):

    temp_norm = cv2.normalize(temp_map, None, 0, 255, cv2.NORM_MINMAX)

    temp_norm = temp_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(temp_norm, cv2.COLORMAP_TURBO)

    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay

# ---------------- COLOR LEGEND ----------------

def color_legend():

    gradient = np.linspace(0,255,256).astype(np.uint8)

    gradient = np.tile(gradient,(40,1))

    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)

    return legend

# ---------------- STRESS SCORE ----------------

def stress_score(chl,temp):

    c_score = (chl - 20) / (60 - 20)

    t_score = 1 - ((temp - 25) / (38 - 25))

    score = 100 * (0.6*c_score + 0.4*t_score)

    return np.clip(score,0,100)

# ---------------- STRESS CLASSIFICATION ----------------

def stress_logic(score):

    if score >= 80:
        return "Healthy Plant", "No irrigation required"

    elif score >= 50:
        return "Moderate Stress", "Irrigation recommended"

    else:
        return "High Stress", "Immediate irrigation required"

# ---------------- PDF REPORT ----------------

def generate_pdf(status, chl, temp, score):

    file = tempfile.NamedTemporaryFile(delete=False)

    c = canvas.Canvas(file.name)

    c.drawString(100,750,"Crop Water Stress Report")

    c.drawString(100,720,f"Stress Status : {status}")
    c.drawString(100,700,f"Chlorophyll Estimate : {chl}")
    c.drawString(100,680,f"Estimated Leaf Temperature : {temp} °C")
    c.drawString(100,660,f"Plant Stress Score : {score}")

    c.save()

    return file.name

# ---------------- MAIN PROCESS ----------------

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    img = white_balance(img)

    img = cv2.GaussianBlur(img,(5,5),0)

    leaf, mask = segment_leaf(img)

    exg = greenness_index(leaf)

# -------- Temperature Matrix --------

    temp_map = temperature_map(exg)

    leaf_pixels = temp_map[mask>0]

    avg_temp = np.mean(leaf_pixels)

# -------- Chlorophyll --------

    exg_leaf = exg[mask>0]

    avg_exg = np.mean(exg_leaf)

    chl_value = chlorophyll_estimate(avg_exg)

# -------- Heatmap --------

    heatmap_overlay = temperature_heatmap(img, temp_map, mask)

    legend = color_legend()

# -------- Stress Score --------

    score = stress_score(chl_value, avg_temp)

# ---------------- DISPLAY ----------------

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Detected Leaf")
        st.image(leaf, channels="BGR")

    with col2:
        st.subheader("Leaf Temperature Heatmap")
        st.image(heatmap_overlay, channels="BGR")

    st.markdown("Temperature Scale")
    st.image(legend, channels="BGR")

    st.info("Blue = Cool | Yellow = Moderate | Red = Hot (Water Stress)")

# ---------------- METRICS ----------------

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Greenness Index", round(avg_exg,2))
    c2.metric("Chlorophyll Estimate", round(chl_value,2))
    c3.metric("Estimated Leaf Temp (°C)", round(avg_temp,2))
    c4.metric("Plant Stress Score", round(score,1))

# ---------------- HEALTH BAR ----------------

    st.subheader("Plant Health Score")

    st.progress(int(score))

# ---------------- FINAL RESULT ----------------

    if st.button("Analyze Stress"):

        status, recommendation = stress_logic(score)

        st.success(status)

        st.info("Recommendation: " + recommendation)

        pdf = generate_pdf(status, round(chl_value,2), round(avg_temp,2), round(score,1))

        with open(pdf,"rb") as f:

            st.download_button(
                "Download Farmer Report",
                f,
                file_name="crop_stress_report.pdf"
            )
