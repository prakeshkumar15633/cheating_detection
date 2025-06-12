import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
from math import atan2
import tempfile
import os
import zipfile
from PIL import Image
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tqdm import tqdm  # Progress bar library
import time

# ----------------------------
# Load models
MODEL_PATH = "cheating_detection_cnn.h5"  # Replace with your model path
model = tf.keras.models.load_model(MODEL_PATH)
yolo_model = YOLO("yolov8n")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ----------------------------
# Email sender config
from_email = "sahilmadan0508@gmail.com"   # 🔁 Replace with your email
from_password = "pmnu rqdj sfns mvdl"     # 🔁 Replace with your app password

# ----------------------------
# Helper functions
def normalize_pose(keypoints):
    """Rotates keypoints so shoulders are aligned horizontally."""
    left_shoulder = keypoints[11 * 3: (11 * 3) + 3]
    right_shoulder = keypoints[12 * 3: (12 * 3) + 3]

    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = -atan2(dy, dx)  # Compute rotation angle

    keypoints_rotated = []
    for i in range(0, len(keypoints), 3):
        x, y, z = keypoints[i], keypoints[i+1], keypoints[i+2]
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        keypoints_rotated.extend([x_new, y_new, z])
    
    return np.array(keypoints_rotated)

def detect_persons(image):
    """Detects all persons in the image using YOLOv8 with confidence filtering."""
    results = yolo_model(image)
    detections = results[0].boxes  # Bounding boxes
    persons = []

    for box, conf, cls in zip(detections.xyxy.cpu().numpy(), detections.conf.cpu().numpy(), detections.cls.cpu().numpy()):
        if int(cls) == 0 and conf > 0.35:  # Class 0 = 'person', Confidence > 35%
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
# Streamlit UI configuration
st.set_page_config(page_title="Cheating Detection", page_icon="📸", layout="wide")

st.markdown("""
    <style>
        body { background-color: #121212; color: #fff; font-family: "Helvetica Neue", sans-serif; }
        .stButton > button { background-color: #0078d4; color: white; font-weight: bold; padding: 15px 32px; border-radius: 8px; }
        .stButton > button:hover { background-color: #005a8d; }
        .stFileUploader { background-color: #1a1a1a; padding: 10px; border-radius: 8px; }
        .stTextInput, .stFileUploader { border-radius: 8px; }
        .stMarkdown, .stWrite { color: #bbb; }
    </style>
""", unsafe_allow_html=True)

# Sidebar inputs
emails = st.sidebar.text_input(
    "📧 Enter email(s) to receive alerts",
    placeholder="example1@gmail.com, example2@gmail.com"
)
email_list = [e.strip() for e in emails.split(",") if e.strip()]

theme_mode = st.sidebar.selectbox("🌃 Theme", ("Dark", "Light"))

# Apply Theme CSS
if theme_mode == "Dark":
    st.markdown("""
        <style>
            .block-container {
                background-color: #121212;
                color: #ffffff;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff;
            }
            .stTextInput > div > input,
            .stFileUploader > div > div {
                background-color: #1e1e1e;
                color: #ffffff;
                border-radius: 8px;
            }
            .stButton > button {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 15px 32px;
                border-radius: 8px;
            }
            .stButton > button:hover {
                background-color: #005a8d;
            }
            .stMarkdown, .stWrite, .stText {
                color: #bbbbbb;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .block-container {
                background-color: #ffffff;
                color: #000000;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #000000;
            }
            .stTextInput > div > input,
            .stFileUploader > div > div {
                background-color: #ffffff;
                color: #000000;
                border-radius: 8px;
            }
            .stButton > button {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 15px 32px;
                border-radius: 8px;
            }
            .stButton > button:hover {
                background-color: #005a8d;
            }
            .stMarkdown, .stWrite, .stText {
                color: #222222;
            }
        </style>
    """, unsafe_allow_html=True)


mode = st.sidebar.radio("Select Mode", ("Upload File", "Real-Time Webcam Feed"))
frame_gap = st.sidebar.slider("Frame Sampling Gap", 1, 100, 30, 1)


detection_threshold = st.sidebar.slider("Cheating Detection Threshold", 0.0, 1.0, 0.5, 0.05)

st.title("🎥 **Cheating Detection from Video, Image or Live Webcam**")

# ----------------------------
# Mode: File Upload (Video/Image Processing)
if mode == "Upload File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
    uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_video is not None:
        temp_dir = tempfile.mkdtemp()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        saved_frames = []
        stframe = st.empty()
        st.write("⏳ **Processing video...**")
        with st.spinner("Processing... Please wait."):
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Frames Processed", position=0, dynamic_ncols=True)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_gap == 0:
                    processed_frame, cheating = analyze_frame(frame, detection_threshold)
                    timestamp = str(timedelta(seconds=int(frame_count / fps)))
                    if cheating:
                        preview_path = os.path.join(temp_dir, f"cheat_frame_{frame_count}.jpg")
                        cv2.imwrite(preview_path, processed_frame)
                        saved_frames.append((preview_path, timestamp))
                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", channels="RGB")
                frame_count += 1
                pbar.update(1)
            cap.release()
            pbar.close()
        if saved_frames:
            st.success(f"⚠️ Detected cheating in {len(saved_frames)} frame(s).")
            st.write("### 🖼️ **Preview of Detected Cheating Frames**:")
            for path, timestamp in saved_frames:
                st.image(path, caption=f"Cheating detected at {timestamp}", use_column_width=True)
            zip_path = os.path.join(temp_dir, "cheating_frames.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for img_path, _ in saved_frames:
                    zipf.write(img_path, os.path.basename(img_path))
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ **Download Cheating Frames (ZIP)**",
                    data=f,
                    file_name="cheating_frames.zip",
                    mime="application/zip"
                )
            if email_list:
                for e in email_list:
                    success = send_email_alert(
                        to_email=e,
                        subject="Cheating Detection Completed",
                        body=f"Cheating detected in {len(saved_frames)} frames.",
                        attachment=zip_path
                    )
                    if success:
                        st.success(f"📨 Email alert sent to {e}")
                    else:
                        st.error(f"⚠️ Failed to send email to {e}")

            
        else:
            st.info("✅ **No cheating detected in sampled frames.**")

    elif uploaded_image is not None:
        img_pil = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        processed_img, cheating = analyze_frame(img_bgr, detection_threshold)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

        if cheating:
            st.success("⚠️ **Cheating detected in the image!**")
        
        # Handle email list
            email_list = [e.strip() for e in emails.split(",") if e.strip()]
            if email_list:
                for e in email_list:
                    success = send_email_alert(to_email=e, attachment=None)
                    if success:
                        st.success(f"📨 Email alert sent to {e}")
                    else:
                        st.error(f"⚠️ Failed to send email to {e}")
        
        # Save image and offer download
            preview_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(preview_path.name, processed_img)
            with open(preview_path.name, "rb") as f:
                st.download_button(
                    label="⬇️ **Download Image with Detected Cheating**",
                    data=f,
                    file_name="cheating_detected_image.jpg",
                    mime="image/jpeg"
                )
        else:
            st.info("✅ **No cheating detected in the image.**")

# ----------------------------
# Mode: Real-Time Webcam Feed with 5s gap & instance image
elif mode == "Real-Time Webcam Feed":
    st.write("### **Webcam Real-Time Cheating Detection**")

    # Initialize session state
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "saved_frames" not in st.session_state:
        st.session_state.saved_frames = []
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0.0

    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam"):
            st.session_state.webcam_running = True
    with col2:
        if st.button("Stop Webcam"):
            st.session_state.webcam_running = False

    placeholder = st.empty()
    temp_dir = tempfile.mkdtemp()
    frame_count = 0

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        st.info("Webcam feed started. Click 'Stop Webcam' to end.")
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing webcam!")
                break

            frame_count += 1
            proc_frame, cheat = analyze_frame(frame, detection_threshold)

            if cheat:
                now = time.time()
                if now - st.session_state.last_alert_time > 5.0:
                    st.session_state.last_alert_time = now

                    # Show the exact cheating instance
                    st.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB),
                             caption=f"Instance at {timedelta(seconds=frame_count//30)}",
                             use_column_width=True)

                    # Blinking red banner
                    st.markdown("""
                      <div style="background:#ff0033;padding:10px;border-radius:8px;
                                  animation:blink 1s infinite;">
                        <h3 style="color:#fff;text-align:center;">🚨 Cheating Detected!</h3>
                      </div>
                      <style>
                        @keyframes blink {
                          0%   {opacity:1;}
                          50%  {opacity:0.3;}
                          100% {opacity:1;}
                        }
                      </style>
                    """, unsafe_allow_html=True)

                    # Toast notification
                    st.toast("🚨 Cheating detected on webcam!")
                    # Voice Alert (Play Sound)
                    st.components.v1.html("""
                    <audio autoplay>
                    <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                        Your browser does not support the audio element.
                    </audio>
                    """, height=0)


                    # Save frame for report
                    path = os.path.join(temp_dir, f"cheat_frame_{frame_count}.jpg")
                    cv2.imwrite(path, proc_frame)
                    st.session_state.saved_frames.append((path, str(timedelta(seconds=frame_count//30))))

            # Always display current feed
            placeholder.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.03)

        cap.release()
        st.success("Webcam feed stopped.")

        # Post-session summary
        if st.session_state.saved_frames:
            st.success(f"⚠️ Detected cheating in {len(st.session_state.saved_frames)} frame(s).")
            for p, ts in st.session_state.saved_frames:
                st.image(p, caption=f"Detected at {ts}", use_column_width=True)
            zip_path = os.path.join(temp_dir, "cheating_frames_webcam.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                for p, _ in st.session_state.saved_frames:
                    zf.write(p, os.path.basename(p))
            if email_list:
                for e in email_list:
                    success = send_email_alert(
                    to_email=e,
                    subject="Webcam Cheating Report",
                    body=f"Detected cheating in {len(st.session_state.saved_frames)} frames.",
                    attachment=zip_path
                )
                if success:
                    st.success(f"📨 Email sent to {e}")
                else:
                    st.error(f"⚠️ Email failed for {e}")

            with open(zip_path, "rb") as f:
                st.download_button("⬇️ Download ZIP",
                                   f,
                                   "cheating_frames_webcam.zip",
                                   "application/zip")
        else:
            st.info("✅ No cheating detected during webcam session.")
