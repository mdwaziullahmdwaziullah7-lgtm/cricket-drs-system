import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import base64
import imageio

st.set_page_config(page_title="DRS System", layout="wide")

def add_bg_image():
    with open("cri.jfif", "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    h1 {{ color: yellow !important; text-shadow: 2px 2px 4px black; }}
    </style>
    """, unsafe_allow_html=True)

def draw_3d_stumps(frame, stump_x, stump_y1, stump_y2):
    shadow_offset = 8
    for sx in [stump_x-18, stump_x, stump_x+18]:
        cv2.line(frame,
                (sx+shadow_offset, stump_y1+shadow_offset),
                (sx+shadow_offset, stump_y2+shadow_offset),
                (50,50,50), 6)
    for sx in [stump_x-18, stump_x, stump_x+18]:
        cv2.line(frame, (sx, stump_y1), (sx, stump_y2), (255,255,255), 5)
        cv2.line(frame, (sx-2, stump_y1), (sx-2, stump_y2), (200,200,255), 2)
    cv2.line(frame, (stump_x-22, stump_y1),
             (stump_x+22, stump_y1), (255,220,100), 4)
    cv2.line(frame, (stump_x-20+shadow_offset, stump_y1+shadow_offset),
             (stump_x+20+shadow_offset, stump_y1+shadow_offset), (50,50,50), 3)
    cv2.ellipse(frame, (stump_x, stump_y2+5), (30,8), 0, 0, 180, (100,80,50), -1)
    return frame

def ai_detect_ball(frame, prev_frame=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array([0,80,80]), np.array([15,255,255]))
    m2 = cv2.inRange(hsv, np.array([160,80,80]), np.array([180,255,255]))
    m3 = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,40,255]))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (11,11), 0)
        diff = cv2.absdiff(blur, prev_blur)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(color_mask, motion_mask)
    else:
        combined = color_mask

    k = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect = w/max(h,1)
            if 0.5 <= aspect <= 2.0:
                cx, cy = x+w//2, y+h//2
                confidence = min(100, int(area/10))
                ball_candidates.append((cx, cy, confidence))

    if ball_candidates:
        ball_candidates.sort(key=lambda x: x[2], reverse=True)
        return ball_candidates[0]
    return None

add_bg_image()

st.markdown("""
<style>
@media (max-width: 768px) {
    .stApp { padding: 0px; }
    h1 { font-size: 1.5em !important; }
    .stButton button { width: 100%; font-size: 1.2em; padding: 15px; }
    .stSlider { width: 100%; }
    .stFileUploader { width: 100%; }
    .stMetric { font-size: 0.8em; }
}
.stButton button {
    background: linear-gradient(45deg, #1a6b1a, #2da82d);
    color: white; border: none;
    border-radius: 10px; font-weight: bold;
}
.stProgress > div > div {
    background: linear-gradient(45deg, #ff4444, #ff8800);
}
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.7) !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.stMetric {
    background: rgba(0,0,0,0.6);
    border-radius: 10px; padding: 10px;
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

st.title("🏏 HAWKEYE DRS SYSTEM")

st.sidebar.header("Settings")
ball_color = st.sidebar.radio("Ball Color:", ["Red Ball", "White Ball"])
stump_x = st.sidebar.slider("Stumps X", 100, 700, 350)
stump_y1 = st.sidebar.slider("Stumps Top", 50, 300, 150)
stump_y2 = st.sidebar.slider("Stumps Bottom", 200, 500, 380)
sensitivity = st.sidebar.slider("Ball Size", 10, 200, 30)

uploaded = st.file_uploader("Cricket video upload করুন", type=["mp4","avi","mov"])

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()

    st.success("🤖 AI Analyzing...")
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
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        progress.progress(min(count/total, 1.0))
        frame = cv2.resize(frame, (720, 480))

        frame = draw_3d_stumps(frame, stump_x, stump_y1, stump_y2)

        ball = ai_detect_ball(frame, prev_frame)
        ball_found = False

        if ball:
            cx, cy, confidence = ball
            ball_found = True
            trail.append((cx, cy))
            if len(trail) > 60:
                trail.pop(0)
            if len(trail) == 6:
                pitch_point = (cx, cy)
            if len(trail) == 18:
                impact_point = (cx, cy)
            if len(trail) >= 6:
                last = trail[-1]
                prev = trail[-3]
                dx = last[0]-prev[0]
                dy = last[1]-prev[1]
                predicted = []
                for s in range(1, 12):
                    px = int(last[0]+dx*s)
                    py = int(last[1]+dy*s)
                    if 0 <= px < 720 and 0 <= py < 480:
                        predicted.append((px, py))
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0,220,255), 2)
            cv2.circle(frame, (cx,cy), 14, (0,180,255), -1)
            cv2.putText(frame, "AI:"+str(confidence)+"%",
                (cx+15, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,255), 2)

        if pitch_point:
            cv2.circle(frame, pitch_point, 12, (0,255,255), -1)
            cv2.putText(frame, "PITCH",
                (pitch_point[0]+14, pitch_point[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if impact_point:
            cv2.circle(frame, impact_point, 14, (0,0,255), -1)
            cv2.putText(frame, "IMPACT",
                (impact_point[0]+14, impact_point[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,255), 2)

        if predicted:
            for i in range(len(predicted)-1):
                cv2.line(frame, predicted[i], predicted[i+1], (0,255,0), 2)
            for pt in predicted:
                cv2.circle(frame, pt, 5, (0,255,0), -1)

        hitting = any(
            stump_x-32 <= pt[0] <= stump_x+32 and
            stump_y1 <= pt[1] <= stump_y2
            for pt in predicted)

        if ball_found:
            if hitting:
                out_count += 1
                cv2.rectangle(frame, (0,0), (720,90), (0,0,180), -1)
                cv2.putText(frame, "OUT! LBW!", (180,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)
            else:
                not_out_count += 1
                cv2.rectangle(frame, (0,0), (720,90), (0,150,0), -1)
                cv2.putText(frame, "NOT OUT!", (180,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 5)

        if count % 6 == 0 and ball_found:
            key_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prev_frame = frame.copy()

    cap.release()
    os.unlink(tfile.name)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frames", total)
    c2.metric("OUT", out_count)
    c3.metric("NOT OUT", not_out_count)

    if out_count > not_out_count:
        st.error("# 🔴 OUT! LBW!")
    else:
        st.success("# ✅ NOT OUT!")

    if key_frames:
        st.subheader("🎬 Ball Tracking")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(key_frames):
                idx = i*(len(key_frames)//3)
                col.image(key_frames[idx], width=400)

        st.markdown("---")
        st.subheader("🎬 Slow Motion Replay")
        slow_speed = st.slider("Slow Motion Speed", 1, 10, 3)
        slow_frames = []
        for f in key_frames:
            for _ in range(slow_speed):
                slow_frames.append(f)
        gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        imageio.mimsave(gif_path.name, slow_frames, fps=10)
        with open(gif_path.name, "rb") as f:
            gif_data = f.read()
        st.image(gif_data, caption="Slow Motion Replay", width=600)
        os.unlink(gif_path.name)
