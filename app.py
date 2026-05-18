
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import base64

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
    h1 {{
        color: yellow !important;
        text-shadow: 2px 2px 4px black;
    }}
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(page_title="DRS System", layout="wide")
st.title("🏏 HAWKEYE DRS SYSTEM")
add_bg_image()
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

    st.success("Analyzing...")
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
            m1 = cv2.inRange(hsv, np.array([0,80,80]), np.array([15,255,255]))
            m2 = cv2.inRange(hsv, np.array([160,80,80]), np.array([180,255,255]))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,40,255]))

        k = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for sx in [stump_x-18, stump_x, stump_x+18]:
            cv2.line(frame, (sx,stump_y1), (sx,stump_y2), (255,255,255), 4)

        ball_found = False
        for c in contours:
            if cv2.contourArea(c) > sensitivity:
                (bx,by,bw,bh) = cv2.boundingRect(c)
                if 0.4 <= bw/max(bh,1) <= 2.5:
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
                            px,py = int(last[0]+dx*s), int(last[1]+dy*s)
                            if 0<=px<720 and 0<=py<480:
                                predicted.append((px,py))
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i-1], trail[i], (0,220,255), 2)
                    cv2.circle(frame, (cx,cy), 14, (0,180,255), -1)
                    break

        if pitch_point:
            cv2.circle(frame, pitch_point, 12, (0,255,255), -1)
            cv2.putText(frame, "PITCH", (pitch_point[0]+14, pitch_point[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if impact_point:
            cv2.circle(frame, impact_point, 14, (0,0,255), -1)
            cv2.putText(frame, "IMPACT", (impact_point[0]+14, impact_point[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,255), 2)

        if predicted:
            for i in range(len(predicted)-1):
                cv2.line(frame, predicted[i], predicted[i+1], (0,255,0), 2)
            for pt in predicted:
                cv2.circle(frame, pt, 5, (0,255,0), -1)

        hitting = any(stump_x-32 <= pt[0] <= stump_x+32 and
                     stump_y1 <= pt[1] <= stump_y2 for pt in predicted)

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

    cap.release()
    os.unlink(tfile.name)

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Frames", total)
    c2.metric("OUT", out_count)
    c3.metric("NOT OUT", not_out_count)

    if out_count > not_out_count:
        st.error("# 🔴 OUT! LBW!")
    else:
        st.success("# ✅ NOT OUT!")

    if key_frames:
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(key_frames):
                idx = i*(len(key_frames)//3)
                col.image(key_frames[idx], use_container_width=True)
# Slow motion section
st.markdown("---")
st.subheader("🎬 Slow Motion Replay")

if key_frames:
    slow_speed = st.slider("Slow Motion Speed", 1, 10, 3)
    
    # Slow motion GIF বানাই
    import imageio
    import tempfile
    
    gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    
    # Key frames কে slow করি
    slow_frames = []
    for f in key_frames:
        for _ in range(slow_speed):
            slow_frames.append(f)
    
    imageio.mimsave(gif_path.name, slow_frames, fps=10)
    
    with open(gif_path.name, "rb") as f:
        gif_data = f.read()
    
    st.image(gif_data, caption="Slow Motion Replay", use_container_width=True)
    os.unlink(gif_path.name)
