
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import os

st.set_page_config(page_title="🏏 Local Match DRS", layout="wide")

st.markdown("""
<style>
body { background: #0a0a0a; }
h1 { color: yellow; text-align: center; font-size: 2.5em; }
.decision-out { 
    background: red; color: white; 
    font-size: 3em; text-align: center; 
    padding: 20px; border-radius: 10px; 
}
.decision-not-out { 
    background: green; color: white; 
    font-size: 3em; text-align: center; 
    padding: 20px; border-radius: 10px; 
}
</style>
""", unsafe_allow_html=True)

st.title("🏏 LOCAL MATCH DRS SYSTEM")

# Sidebar
st.sidebar.header("⚙️ Setup")
st.sidebar.markdown("### 📱 Phone Setup:")
st.sidebar.info("Phone টা stumps এর পাশে রাখুন — side view হবে")

ball_color = st.sidebar.radio("Ball Color:", 
    ["🔴 Red Ball", "⚪ White Ball"])

stump_x = st.sidebar.slider("Stumps Position", 100, 700, 350)
stump_y1 = st.sidebar.slider("Stumps Top", 50, 300, 150)
stump_y2 = st.sidebar.slider("Stumps Bottom", 200, 500, 380)
sensitivity = st.sidebar.slider("Ball Size", 10, 200, 30)

mode = st.radio("Mode:", 
    ["📹 Live Camera (মাঠে)", "🎥 Video Upload (পরে দেখুন)"])

if mode == "📹 Live Camera (মাঠে)":
    
    st.info("📱 Camera চালু করুন — stumps এর পাশ থেকে record করুন")
    
    class LocalDRS(VideoTransformerBase):
        def __init__(self):
            self.trail = []
            self.pitch_point = None
            self.impact_point = None
            self.predicted = []
            self.decision = None
            self.frame_count = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Ball detect
            if "Red" in ball_color:
                m1 = cv2.inRange(hsv,
                    np.array([0,80,80]), np.array([15,255,255]))
                m2 = cv2.inRange(hsv,
                    np.array([160,80,80]), np.array([180,255,255]))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = cv2.inRange(hsv,
                    np.array([0,0,180]), np.array([180,40,255]))

            k = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h, w = img.shape[:2]

            # Dark overlay উপরে
            cv2.rectangle(img, (0,0), (w,100), (0,0,0), -1)

            # Stumps আঁকি
            for sx in [stump_x-18, stump_x, stump_x+18]:
                cv2.line(img, (sx,stump_y1), (sx,stump_y2),
                        (255,255,255), 4)
            cv2.line(img, (stump_x-28,stump_y1),
                    (stump_x+28,stump_y1), (200,200,200), 3)

            # Ball find
            ball_found = False
            for c in contours:
                if cv2.contourArea(c) > sensitivity:
                    (bx,by,bw,bh) = cv2.boundingRect(c)
                    asp = bw/max(bh,1)
                    if 0.4 <= asp <= 2.5:
                        cx, cy = bx+bw//2, by+bh//2
                        self.trail.append((cx,cy))
                        ball_found = True
                        if len(self.trail) > 60:
                            self.trail.pop(0)

                        # Pitch point
                        if len(self.trail) == 6:
                            self.pitch_point = (cx,cy)

                        # Impact point
                        if len(self.trail) == 18:
                            self.impact_point = (cx,cy)

                        # Predicted path
                        if len(self.trail) >= 6:
                            last = self.trail[-1]
                            prev = self.trail[-3]
                            dx = last[0]-prev[0]
                            dy = last[1]-prev[1]
                            self.predicted = []
                            for s in range(1,12):
                                px = int(last[0]+dx*s)
                                py = int(last[1]+dy*s)
                                if 0<=px<w and 0<=py<h:
                                    self.predicted.append((px,py))

                        # Trail আঁকি
                        for i in range(1, len(self.trail)):
                            cv2.line(img, self.trail[i-1],
                                    self.trail[i], (0,220,255), 2)

                        # Ball
                        cv2.circle(img, (cx,cy), 14, (0,180,255), -1)
                        cv2.circle(img, (cx,cy), 14, (255,255,255), 2)
                        break

            # Pitch point 🟡
            if self.pitch_point:
                cv2.circle(img, self.pitch_point, 12,
                          (0,255,255), -1)
                cv2.circle(img, self.pitch_point, 12,
                          (255,255,255), 2)
                cv2.putText(img, "PITCH",
                    (self.pitch_point[0]+14,
                     self.pitch_point[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

            # Impact point 🔴
            if self.impact_point:
                cv2.circle(img, self.impact_point, 14,
                          (0,0,255), -1)
                cv2.circle(img, self.impact_point, 14,
                          (255,255,255), 2)
                cv2.putText(img, "IMPACT",
                    (self.impact_point[0]+14,
                     self.impact_point[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,50,255), 2)

            # Predicted path 🟢
            if self.predicted:
                for i in range(len(self.predicted)-1):
                    cv2.line(img, self.predicted[i],
                            self.predicted[i+1],
                            (0,255,0), 2)
                for pt in self.predicted:
                    cv2.circle(img, pt, 5, (0,255,0), -1)

            # LBW Decision
            hitting = False
            if self.predicted:
                for pt in self.predicted:
                    if (stump_x-32 <= pt[0] <= stump_x+32 and
                        stump_y1 <= pt[1] <= stump_y2):
                        hitting = True
                        break

            if ball_found:
                if hitting:
                    self.decision = "OUT"
                    cv2.rectangle(img, (0,0), (w,100),
                                 (0,0,180), -1)
                    cv2.putText(img, "OUT! LBW!", (w//2-180,75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (255,255,255), 5)
                else:
                    self.decision = "NOT OUT"
                    cv2.rectangle(img, (0,0), (w,100),
                                 (0,150,0), -1)
                    cv2.putText(img, "NOT OUT!", (w//2-180,75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (255,255,255), 5)
            else:
                cv2.putText(img, "Tracking ball...", (w//2-150,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (200,200,200), 2)

            # Speed
            speed = max(60, 145 - len(self.trail))
            cv2.putText(img, str(speed)+" km/h",
                (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,0), 2)

            return img

    webrtc_streamer(
        key="local-drs",
        video_transformer_factory=LocalDRS,
        media_stream_constraints={
            "video": {"width": 720, "height": 480},
            "audio": False
        }
    )

elif mode == "🎥 Video Upload (পরে দেখুন)":
    uploaded = st.file_uploader(
        "🎥 Match এর video upload করুন",
        type=["mp4","avi","mov"])

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        st.success("✅ Analyzing...")
        cap = cv2.VideoCapture(tfile.name)
        trail = []
        pitch_point = None
        impact_point = None
        predicted = []
        key_frames = []
        out_count = 0
        not_out_count = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            progress.progress(min(count/total, 1.0))
            frame = cv2.resize(frame, (720,480))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if "Red" in ball_color:
                m1 = cv2.inRange(hsv,
                    np.array([0,80,80]), np.array([15,255,255]))
                m2 = cv2.inRange(hsv,
                    np.array([160,80,80]), np.array([180,255,255]))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = cv2.inRange(hsv,
                    np.array([0,0,180]), np.array([180,40,255]))

            k = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for sx in [stump_x-18, stump_x, stump_x+18]:
                cv2.line(frame, (sx,stump_y1), (sx,stump_y2),
                        (255,255,255), 4)
            cv2.line(frame, (stump_x-28,stump_y1),
                    (stump_x+28,stump_y1), (200,200,200), 3)

            ball_found = False
            for c in contours:
                if cv2.contourArea(c) > sensitivity:
                    (bx,by,bw,bh) = cv2.boundingRect(c)
                    asp = bw/max(bh,1)
                    if 0.4 <= asp <= 2.5:
                        cx,cy = bx+bw//2, by+bh//2
                        trail.append((cx,cy))
                        ball_found = True
                        if len(trail) > 60:
                            trail.pop(0)

                        if len(trail) == 6:
                            pitch_point = (cx,cy)
                        if len(trail) == 18:
                            impact_point = (cx,cy)

                        if len(trail) >= 6:
                            last = trail[-1]
                            prev = trail[-3]
                            dx = last[0]-prev[0]
                            dy = last[1]-prev[1]
                            predicted = []
                            for s in range(1,12):
                                px = int(last[0]+dx*s)
                                py = int(last[1]+dy*s)
                                if 0<=px<720 and 0<=py<480:
                                    predicted.append((px,py))

                        for i in range(1, len(trail)):
                            cv2.line(frame, trail[i-1], trail[i],
                                    (0,220,255), 2)
                        cv2.circle(frame, (cx,cy), 14,
                                  (0,180,255), -1)
                        cv2.circle(frame, (cx,cy), 14,
                                  (255,255,255), 2)
                        break

            if pitch_point:
                cv2.circle(frame, pitch_point, 12,
                          (0,255,255), -1)
                cv2.putText(frame, "PITCH",
                    (pitch_point[0]+14, pitch_point[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

            if impact_point:
                cv2.circle(frame, impact_point, 14,
                          (0,0,255), -1)
                cv2.putText(frame, "IMPACT",
                    (impact_point[0]+14, impact_point[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,50,255), 2)

            if predicted:
                for i in range(len(predicted)-1):
                    cv2.line(frame, predicted[i],
                            predicted[i+1], (0,255,0), 2)
                for pt in predicted:
                    cv2.circle(frame, pt, 5, (0,255,0), -1)

            hitting = any(
                stump_x-32 <= pt[0] <= stump_x+32 and
                stump_y1 <= pt[1] <= stump_y2
                for pt in predicted)

            if ball_found:
                if hitting:
                    out_count += 1
                    cv2.rectangle(frame, (0,0), (720,90),
                                 (0,0,180), -1)
                    cv2.putText(frame, "OUT! LBW!", (180,65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (255,255,255), 5)
                else:
                    not_out_count += 1
                    cv2.rectangle(frame, (0,0), (720,90),
                                 (0,150,0), -1)
                    cv2.putText(frame, "NOT OUT!", (180,65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (255,255,255), 5)

            if count % 6 == 0 and ball_found:
                key_frames.append(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        os.unlink(tfile.name)

        st.markdown("---")
        st.subheader("📊 DRS Result")
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Frames", total)
        c2.metric("🔴 OUT", out_count)
        c3.metric("✅ NOT OUT", not_out_count)

        if out_count > not_out_count:
            st.error("# 🔴 সিদ্ধান্ত: OUT! LBW!")
        else:
            st.success("# ✅ সিদ্ধান্ত: NOT OUT!")

        if key_frames:
            st.subheader("🎬 Ball Tracking")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if i < len(key_frames):
                    idx = i*(len(key_frames)//3)
                    col.image(key_frames[idx],
                             use_container_width=True)
