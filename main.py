import streamlit as st
import cv2
import tempfile
import ExerciseAiTrainer as exercise
from chatbot import chat_ui
import time
import numpy as np
import PoseModule2 as pm


# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="APEX · AI Gym Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg:        #080a0f;
    --surface:   #0d1018;
    --card:      #121720;
    --border:    #1e2a3a;
    --accent:    #f97316;
    --accent2:   #fb923c;
    --lime:      #a3e635;
    --muted:     #4a5568;
    --text:      #e2e8f0;
    --text-dim:  #94a3b8;
    --glow:      rgba(249,115,22,0.35);
    --font-head: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090c12 0%, #0b0f18 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 0 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--glow) !important;
}
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-size: 18px !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 28px !important;
    cursor: pointer !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 20px var(--glow) !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px var(--glow) !important;
}
[data-testid="stAlert"] {
    background: rgba(249,115,22,0.08) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 6px !important;
    color: var(--text-dim) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
video { border-radius: 12px; border: 1px solid var(--border); }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color .2s, box-shadow .2s;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--glow);
}
.metric-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 48px;
    line-height: 1;
    color: #fff;
    text-shadow: 0 0 20px var(--glow);
}
.apex-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0;
}
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    border: 1px solid rgba(249,115,22,0.4);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 99px;
}
.ex-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    transition: all .2s;
    margin-bottom: 12px;
}
.ex-card:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px var(--glow);
}
.ex-icon { font-size: 32px; margin-bottom: 8px; }
.ex-name { font-family: var(--font-head); font-size: 20px; letter-spacing: 0.05em; color: #fff; }
.ex-desc { font-size: 12px; color: var(--text-dim); margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:28px 20px 20px;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:40px;
                        letter-spacing:.08em;color:#fff;line-height:1;">⚡ APEX</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                        letter-spacing:.2em;text-transform:uppercase;color:#f97316;margin-top:2px;">
                AI GYM TRACKER
            </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,#1e2a3a,transparent);margin-bottom:20px;"></div>
        <div style="padding:0 16px;">
        <div class="badge">Navigation</div>
        <div style="height:8px;"></div>
        """, unsafe_allow_html=True)

        options = st.selectbox(
            "Mode",
            ("🏠  Dashboard", "🎬  Video Analysis", "📹  WebCam Live", "🤖  Auto Classify", "💬  AI Coach"),
            label_visibility="collapsed",
        )

        st.markdown("""
        <div style="height:20px;"></div>
        <div class="badge">System Status</div>
        <div style="padding:14px 0;display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#94a3b8;">
                <span>Exercises</span>
                <span style="color:#fff;font-family:'JetBrains Mono',monospace;">4</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#94a3b8;">
                <span>AI Model</span>
                <span style="color:#a3e635;font-family:'JetBrains Mono',monospace;">BiLSTM ✓</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#94a3b8;">
                <span>Pose Engine</span>
                <span style="color:#a3e635;font-family:'JetBrains Mono',monospace;">MediaPipe ✓</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    return options.split("  ")[-1].strip()


def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="padding:36px 40px 0;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:52px;
                    letter-spacing:.05em;color:#fff;line-height:1;">{title}</div>
        {"" if not subtitle else f'<div style="font-size:14px;color:#94a3b8;margin-top:6px;">{subtitle}</div>'}
        <div style="height:1px;background:linear-gradient(90deg,#1e2a3a,transparent);margin-top:20px;"></div>
    </div>
    <div style="height:24px;"></div>
    """, unsafe_allow_html=True)


def page_dashboard():
    page_header("Dashboard", "Your AI-powered workout intelligence hub")
    st.markdown('<div style="padding:0 40px;">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [("EXERCISES", "4", "Supported"), ("AI MODEL", "BiLSTM", "Architecture"),
             ("ACCURACY", "95%+", "Pose detection"), ("LATENCY", "~30ms", "Per frame")]
    for col, (label, val, desc) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div style="font-size:11px;color:#4a5568;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:28px;letter-spacing:.05em;color:#fff;margin-bottom:16px;">Available Exercises</div>', unsafe_allow_html=True)

    ec1, ec2, ec3, ec4 = st.columns(4)
    exercises = [("💪","Bicep Curl","Arm isolation · counts both sides"),
                 ("⬇️","Push Up","Upper body compound"),
                 ("🦵","Squat","Lower body compound"),
                 ("🏋️","Shoulder Press","Overhead pressing")]
    for col, (icon, name, desc) in zip([ec1, ec2, ec3, ec4], exercises):
        with col:
            st.markdown(f'<div class="ex-card"><div class="ex-icon">{icon}</div><div class="ex-name">{name}</div><div class="ex-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:28px;letter-spacing:.05em;color:#fff;margin-bottom:16px;">How It Works</div>', unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3)
    steps = [("01","Pose Detection","MediaPipe extracts 33 key body landmarks from each video frame in real-time."),
             ("02","AI Classification","A Bi-directional LSTM classifies the exercise and tracks movement phases."),
             ("03","Rep Counting","Joint angles are monitored to trigger rep counts and provide instant feedback.")]
    for col, (num, title, desc) in zip([hw1, hw2, hw3], steps):
        with col:
            st.markdown(f"""
            <div style="background:#121720;border:1px solid #1e2a3a;border-radius:12px;padding:22px 18px;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;color:rgba(249,115,22,0.25);line-height:1;">{num}</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:#fff;margin-top:4px;">{title}</div>
                <div style="font-size:13px;color:#94a3b8;margin-top:8px;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def page_video():
    page_header("Video Analysis", "Upload a workout clip and let AI count your reps")
    st.markdown('<div style="padding:0 40px;">', unsafe_allow_html=True)

    col_controls, col_main = st.columns([1, 2])

    with col_controls:
        st.markdown("""
        <div style="background:#121720;border:1px solid #1e2a3a;border-radius:12px;padding:22px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.2em;
                        text-transform:uppercase;color:#f97316;margin-bottom:14px;">Exercise Type</div>
        """, unsafe_allow_html=True)
        exercise_options = st.selectbox("Select Exercise", ("Bicept Curl","Push Up","Squat","Shoulder Press"), label_visibility="collapsed")
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#f97316;margin-bottom:10px;">Upload Video</div>', unsafe_allow_html=True)
        video_file_buffer = st.file_uploader("Upload", type=["mp4","mov","avi","asf","m4v"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#121720;border:1px solid #1e2a3a;border-radius:12px;padding:18px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#a3e635;margin-bottom:12px;">📋 Tips</div>
            <ul style="font-size:12px;color:#94a3b8;padding-left:16px;line-height:1.8;margin:0;">
                <li>Ensure full body is visible</li>
                <li>Good lighting = better accuracy</li>
                <li>Side or front angle works best</li>
                <li>Supported: MP4, MOV, AVI</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_main:
        if video_file_buffer is None:
            st.markdown("""
            <div style="background:#121720;border:1px dashed #1e2a3a;border-radius:16px;
                        padding:60px 40px;text-align:center;">
                <div style="font-size:48px;margin-bottom:16px;">🎬</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:#fff;margin-bottom:8px;">No Video Uploaded</div>
                <div style="font-size:13px;color:#4a5568;">Upload a workout video on the left to begin AI analysis</div>
            </div>""", unsafe_allow_html=True)
        else:
            tfflie = tempfile.NamedTemporaryFile(delete=False)
            tfflie.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tfflie.name)
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#f97316;margin-bottom:10px;">Input Video</div>', unsafe_allow_html=True)
            st.video(tfflie.name)
            st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#f97316;margin-bottom:14px;">Processed Output</div>', unsafe_allow_html=True)
            exer = exercise.Exercise()
            if exercise_options == "Bicept Curl":
                exer.bicept_curl(cap, is_video=True, counter=0, stage_right=None, stage_left=None)
            elif exercise_options == "Push Up":
                exer.push_up(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == "Squat":
                exer.squat(cap, is_video=True, counter=0, stage=None)
            elif exercise_options == "Shoulder Press":
                exer.shoulder_press(cap, is_video=True, counter=0, stage=None)

    st.markdown('</div>', unsafe_allow_html=True)


def page_webcam():
    page_header("Live WebCam", "Real-time pose estimation and rep counting")
    st.markdown('<div style="padding:0 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.3);
                border-radius:10px;padding:14px 18px;margin-bottom:24px;font-size:13px;color:#94a3b8;">
        ⚠️  <strong style="color:#f97316;">Cloud Note:</strong>
        WebCam may be unstable on Streamlit Cloud. Use Video Analysis if issues occur.
    </div>""", unsafe_allow_html=True)

    col_set, col_stream = st.columns([1, 2])
    with col_set:
        st.markdown('<div style="background:#121720;border:1px solid #1e2a3a;border-radius:12px;padding:22px;"><div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#f97316;margin-bottom:14px;">Exercise</div>', unsafe_allow_html=True)
        exercise_general = st.selectbox("Exercise", ("Bicept Curl","Push Up","Squat","Shoulder Press"), label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stream:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]},{"urls":["turn:openrelay.metered.ca:80"],"username":"openrelayproject","credential":"openrelayproject"}]})
        if "webrtc_started" not in st.session_state:
            st.session_state.webrtc_started = True

        class ExerciseProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = pm.posture_detector()
                self.counter = 0; self.stage = None
                self.stage_right = None; self.stage_left = None
                self.exercise_name = exercise_general
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = self.detector.find_person(img)
                landmark_list = self.detector.find_landmarks(img, draw=False)
                if len(landmark_list) != 0:
                    exer_inst = exercise.Exercise()
                    if self.exercise_name == "Bicept Curl":
                        from ExerciseAiTrainer import count_repetition_bicep_curl
                        self.stage_right, self.stage_left, self.counter = count_repetition_bicep_curl(self.detector, img, landmark_list, self.stage_right, self.stage_left, self.counter, exer_inst)
                    elif self.exercise_name == "Push Up":
                        from ExerciseAiTrainer import count_repetition_push_up
                        self.stage, self.counter = count_repetition_push_up(self.detector, img, landmark_list, self.stage, self.counter, exer_inst)
                    elif self.exercise_name == "Squat":
                        from ExerciseAiTrainer import count_repetition_squat
                        self.stage, self.counter = count_repetition_squat(self.detector, img, landmark_list, self.stage, self.counter, exer_inst)
                    elif self.exercise_name == "Shoulder Press":
                        from ExerciseAiTrainer import count_repetition_shoulder_press
                        self.stage, self.counter = count_repetition_shoulder_press(self.detector, img, landmark_list, self.stage, self.counter, exer_inst)
                overlay = img.copy()
                cv2.rectangle(overlay, (0,0), (200,80), (10,10,10), -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                cv2.rectangle(img, (0,0), (200,80), (249,115,22), 2)
                cv2.putText(img, "REPS", (14,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (249,115,22), 1)
                cv2.putText(img, str(self.counter), (10,68), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,255,255), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="stable-webcam", video_processor_factory=ExerciseProcessor,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video":True,"audio":False}, async_processing=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_auto_classify():
    page_header("Auto Classify", "AI detects and counts exercises automatically")
    st.markdown('<div style="padding:0 40px;">', unsafe_allow_html=True)
    col_info, col_action = st.columns([2,1])
    with col_info:
        st.markdown("""
        <div style="background:#121720;border:1px solid #1e2a3a;border-radius:14px;padding:28px;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:#fff;margin-bottom:14px;">How Auto Mode Works</div>
            <div style="font-size:14px;color:#94a3b8;line-height:1.8;">
                The Bi-directional LSTM model analyzes sequences of pose landmarks across frames and
                <strong style="color:#f97316;">automatically identifies</strong> which exercise you are performing —
                then begins counting reps immediately. No pre-selection needed.
            </div>
            <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
                <span class="badge">Bicep Curl</span>
                <span class="badge">Push Up</span>
                <span class="badge">Squat</span>
                <span class="badge">Shoulder Press</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_action:
        st.markdown("""
        <div style="background:#121720;border:1px solid #1e2a3a;border-radius:14px;padding:28px;text-align:center;">
            <div style="font-size:52px;margin-bottom:16px;">🤖</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:#fff;margin-bottom:8px;">Ready</div>
            <div style="font-size:12px;color:#4a5568;margin-bottom:20px;">Requires webcam access</div>
        """, unsafe_allow_html=True)
        if st.button("▶  Start Auto Classification"):
            exer = exercise.Exercise()
            exer.auto_classify_and_count()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_chatbot():
    page_header("AI Coach", "Your personal fitness trainer, powered by GPT-4o mini")
    st.markdown('<div style="padding:0 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#121720;border:1px solid #1e2a3a;border-radius:10px;padding:12px 18px;
                margin-bottom:20px;font-size:12px;color:#4a5568;font-family:'JetBrains Mono',monospace;">
        ℹ️  AI responses may not be medically accurate. Always consult a qualified professional.
    </div>""", unsafe_allow_html=True)
    chat_ui()
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    page = render_sidebar()
    if page == "Dashboard":
        page_dashboard()
    elif page == "Video Analysis":
        page_video()
    elif page == "WebCam Live":
        page_webcam()
    elif page == "Auto Classify":
        page_auto_classify()
    elif page == "AI Coach":
        page_chatbot()


if __name__ == "__main__":
    main()
