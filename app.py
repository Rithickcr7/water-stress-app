import streamlit as st
import cv2
import numpy as np
from reportlab.pdfgen import canvas
import tempfile

# ---------------- UI ----------------

st.set_page_config(
    page_title="Hibiscus Leaf Stress Detector",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Hibiscus Leaf Water Stress Detection")
st.write("AI based plant stress monitoring system")

# ---------------- INPUT ----------------

col1,col2 = st.columns(2)

with col1:
    image = st.file_uploader("Upload Leaf Image",type=["jpg","png","jpeg"])

with col2:
    camera = st.camera_input("Capture Leaf")

if camera is not None:
    image = camera

# ---------------- LEAF SEGMENTATION ----------------

def segment_leaf(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower = np.array([20,30,30])
    upper = np.array([95,255,255])

    mask = cv2.inRange(hsv,lower,upper)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    leaf = cv2.bitwise_and(img,img,mask=mask)

    return leaf,mask

# ---------------- GREENNESS INDEX ----------------

def greenness_index(img):

    img = img.astype(np.float32)

    b,g,r = cv2.split(img)

    exg = 2*g - r - b

    return exg

# ---------------- CHLOROPHYLL ----------------

def chlorophyll_estimate(exg):

    chl = 0.38 * exg + 28

    return np.clip(chl,20,60)

# ---------------- TEMPERATURE ----------------

def estimate_temperature(exg):

    temp = 34 - (0.04 * exg)

    return np.clip(temp,25,38)

# ---------------- STRESS SCORE ----------------

def stress_score(chl,temp):

    c_score = (chl - 20) / (60 - 20)

    t_score = 1 - ((temp - 25) / (38 - 25))

    score = 100 * (0.6 * c_score + 0.4 * t_score)

    return np.clip(score,0,100)

# ---------------- HEATMAP ----------------

def stress_heatmap_overlay(img,exg,mask):

    exg_norm = cv2.normalize(exg,None,0,255,cv2.NORM_MINMAX)

    exg_norm = exg_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(exg_norm,cv2.COLORMAP_TURBO)

    heatmap = cv2.bitwise_and(heatmap,heatmap,mask=mask)

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    return overlay

# ---------------- STRESS SPOTS ----------------

def stress_spots(img,exg,mask):

    threshold = 50

    stress_mask = (exg < threshold) & (mask > 0)

    highlight = img.copy()

    highlight[stress_mask] = [0,0,255]

    return highlight

# ---------------- COLOR LEGEND ----------------

def color_legend():

    gradient = np.linspace(0,255,256).astype(np.uint8)

    gradient = np.tile(gradient,(40,1))

    legend = cv2.applyColorMap(gradient,cv2.COLORMAP_TURBO)

    return legend

# ---------------- STRESS LOGIC ----------------

def stress_logic(score):

    if score >= 80:
        return "Healthy Plant","No irrigation required"

    elif 50 <= score < 80:
        return "Moderate Stress","Irrigation recommended"

    else:
        return "High Stress","Immediate irrigation needed"

# ---------------- PDF ----------------

def generate_pdf(status,chl,temp,score):

    file = tempfile.NamedTemporaryFile(delete=False)

    c = canvas.Canvas(file.name)

    c.drawString(100,750,"Hibiscus Plant Stress Report")

    c.drawString(100,720,f"Stress Status : {status}")
    c.drawString(100,700,f"Chlorophyll Estimate : {chl}")
    c.drawString(100,680,f"Estimated Leaf Temperature : {temp} C")
    c.drawString(100,660,f"Plant Stress Score : {score}")

    c.save()

    return file.name

# ---------------- MAIN PROCESS ----------------

if image is not None:

    file_bytes = np.asarray(bytearray(image.read()),dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    img = cv2.GaussianBlur(img,(5,5),0)

    leaf,mask = segment_leaf(img)

    exg = greenness_index(leaf)

    leaf_pixels = exg[mask>0]

    exg_value = np.mean(leaf_pixels)

    chl_value = chlorophyll_estimate(exg_value)

    temp_value = estimate_temperature(exg_value)

    score = stress_score(chl_value,temp_value)

    heatmap_overlay = stress_heatmap_overlay(img,exg,mask)

    stress_highlight = stress_spots(img,exg,mask)

    legend = color_legend()

# ---------------- DISPLAY ----------------

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Detected Leaf")
        st.image(leaf,channels="BGR")

    with col2:
        st.subheader("Stress Heatmap Overlay")
        st.image(heatmap_overlay,channels="BGR")

    st.subheader("Stress Spot Detection")
    st.image(stress_highlight,channels="BGR")

    st.markdown("Stress Color Scale")
    st.image(legend,channels="BGR")

    st.info("Blue = Healthy | Yellow = Moderate | Red = High Stress")

# ---------------- METRICS ----------------

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Greenness Index",round(exg_value,2))
    c2.metric("Chlorophyll Estimate",round(chl_value,2))
    c3.metric("Estimated Leaf Temp (°C)",round(temp_value,2))
    c4.metric("Plant Stress Score",round(score,1))

# ---------------- SCORE BAR ----------------

    st.subheader("Plant Health Score")

    st.progress(int(score))

# ---------------- FINAL RESULT ----------------

    if st.button("Analyze Stress"):

        status,recommend = stress_logic(score)

        st.success(status)

        st.info(recommend)

        pdf = generate_pdf(status,round(chl_value,2),round(temp_value,2),round(score,1))

        with open(pdf,"rb") as f:

            st.download_button(
                "Download Farmer Report",
                f,
                file_name="hibiscus_stress_report.pdf"
            )
