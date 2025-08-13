import streamlit as st
st.set_page_config(page_title="Cheating Detection", page_icon="📸", layout="wide")

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
from math import atan2
from PIL import Image
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# ----------------------------
# Load models with caching to save memory
@st.cache(allow_output_mutation=True)
def load_cnn_model():
    return tf.keras.models.load_model("cheating_detection_cnn.h5")

@st.cache(allow_output_mutation=True)
def load_yolo_model():
    return YOLO("yolov8n")

model = load_cnn_model()
yolo_model = load_yolo_model()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ----------------------------
# Email sender config
from_email = "sahilmadan0508@gmail.com"
from_password = ""  # Move this to Streamlit secrets for security

# ----------------------------
# Helper functions
def normalize_pose(keypoints):
    left_shoulder = keypoints[11 * 3: (11 * 3) + 3]
    right_shoulder = keypoints[12 * 3: (12 * 3) + 3]

    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = -atan2(dy, dx)

    keypoints_rotated = []
    for i in range(0, len(keypoints), 3):
        x, y, z = keypoints[i], keypoints[i+1], keypoints[i+2]
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        keypoints_rotated.extend([x_new, y_new, z])
    
    return np.array(keypoints_rotated)

def detect_persons(image):
    results = yolo_model(image)
    detections = results[0].boxes
    persons = []

    for box, conf, cls in zip(detections.xyxy.cpu().numpy(),
                              detections.conf.cpu().numpy(),
                              detections.cls.cpu().numpy()):
        if int(cls) == 0 and conf > 0.35:
            xmin, ymin, xmax, ymax = map(int, box)
            persons.append((xmin, ymin, xmax, ymax))
    
    return persons

def analyze_frame(frame, detection_threshold=0.5):
    cheating = False
    for xmin, ymin, xmax, ymax in detect_persons(frame):
        crop = frame[ymin:ymax, xmin:xmax]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            continue
        kpts = []
        for lm in res.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.z])
        kpts = normalize_pose(kpts)
        kpts = kpts / np.max(np.abs(kpts))
        X = np.array([kpts])[:, :, np.newaxis]
        prob = model.predict(X)[0][0]
        pred = int(prob > detection_threshold)
        color = (0, 0, 255) if pred else (0, 255, 0)
        label = "Cheating" if pred else "Not Cheating"
        if pred:
            cheating = True
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"{label} {prob*100:.2f}%",
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame, cheating

def send_email_alert(to_email, subject="Cheating Alert", body="Cheating detected.", attachment=None):
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = from_email, to_email, subject
        msg.attach(MIMEText(body, 'plain'))
        if attachment:
            part = MIMEBase('application', 'octet-stream')
            with open(attachment, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            f'attachment; filename={os.path.basename(attachment)}')
            msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email error:", e)
        return False

# ----------------------------
# Streamlit UI
st.set_page_config(page_title="Cheating Detection", page_icon="📸", layout="wide")

# Sidebar inputs
emails = st.sidebar.text_input(
    "📧 Enter email(s) to receive alerts",
    placeholder="example1@gmail.com, example2@gmail.com"
)
email_list = [e.strip() for e in emails.split(",") if e.strip()]

theme_mode = st.sidebar.selectbox("🌃 Theme", ("Dark", "Light"))
detection_threshold = st.sidebar.slider("Cheating Detection Threshold", 0.0, 1.0, 0.5, 0.05)

mode = st.sidebar.radio("Select Mode", ("Upload File", "Real-Time Webcam Feed"))

st.title("🎥 **Cheating Detection from Image**")

# ----------------------------
# Mode: File Upload
if mode == "Upload File":
    st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], disabled=True)
    st.error("🚫 Video upload unavailable on Streamlit Cloud.")
    st.markdown(
        "[🎥 Watch full video demo here](https://drive.google.com/file/d/1AKrS8LUVNul4EWxKOFTmc30uRhF0sukM/view?usp=sharing)",
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("Or upload an image (Max 1MB)",
                                      type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed")
    if uploaded_image is not None:
        if uploaded_image.size > 1_000_000:
            st.error("❌ Image exceeds 1MB limit. Please upload a smaller image.")
        else:
            img_pil = Image.open(uploaded_image).convert("RGB")
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            processed_img, cheating = analyze_frame(img_bgr, detection_threshold)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                    caption="Processed Image", use_column_width=True)

            processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            from io import BytesIO
            buf = BytesIO()
            processed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

            if cheating:
                st.success("⚠️ **Cheating detected in the image!**")
                if email_list:
                    for e in email_list:
                        success = send_email_alert(to_email=e, attachment=None)
                        if success:
                            st.success(f"📨 Email alert sent to {e}")
                        else:
                            st.error(f"⚠️ Failed to send email to {e}")
            else:
                st.info("✅ **No cheating detected in the image.**")

# ----------------------------
# Mode: Webcam
elif mode == "Real-Time Webcam Feed":
    col1, col2 = st.columns(2)
    with col1:
        st.button("📷 Start Webcam", disabled=True)
    with col2:
        st.button("🛑 Stop Webcam", disabled=True)

    st.error("🚫 Webcam feed unavailable on Streamlit Cloud.")
    st.markdown(
        "[📺 Watch full webcam demo here](https://drive.google.com/file/d/1AKrS8LUVNul4EWxKOFTmc30uRhF0sukM/view?usp=sharing)",
        unsafe_allow_html=True
    )
