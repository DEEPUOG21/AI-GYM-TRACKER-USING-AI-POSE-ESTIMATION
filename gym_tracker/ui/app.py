"""Demo views. Inference and workout state live outside Streamlit."""
import json
import logging
import os
from io import BytesIO
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from dotenv import load_dotenv
from gym_tracker.classification import get_classifier
from gym_tracker.coaching import coach
from gym_tracker.pipeline import WorkoutTracker
from gym_tracker.pose import PoseDetector
from gym_tracker.repetitions import EXERCISES
from gym_tracker.video import analyze_video
from gym_tracker.ui.components import (TITLES, empty_state, format_duration, history_chart,
                                       html, motion_hero, page_heading, section_heading,
                                       session_charts, session_label)

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

PAGES = ["Dashboard", "Video Analysis", "WebCam Live", "Auto Classify", "Workout History", "AI Coach"]
PAGE_LABELS = {"Dashboard": "◈　Overview", "Video Analysis": "↗　Video studio",
               "WebCam Live": "◉　Live session", "Auto Classify": "✧　Auto recognition",
               "Workout History": "▥　Workout history", "AI Coach": "✳　AI coach"}


def navigate(page, demo=False):
    st.session_state["navigation"] = page
    if demo:
        st.session_state["sample_True"] = True


def setting(name, default=""):
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    # In Streamlit 1.35, .get() renders an error before raising for absent files.
    if st.secrets.load_if_toml_exists():
        value = st.secrets.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def make_tracker(mode, source):
    classifier = get_classifier() if mode == "auto" else None
    return WorkoutTracker(PoseDetector(), exercise=None if mode == "auto" else mode,
                          classifier=classifier, source=source)


def save_session(snapshot):
    if snapshot and snapshot["frames"]:
        sessions = st.session_state.setdefault("sessions", {})
        sessions[snapshot["session_id"]] = snapshot
        while len(sessions) > 20:
            del sessions[next(iter(sessions))]


def finish_webcam():
    processor = st.session_state.pop("last_processor", None)
    if processor is not None:
        processor.on_ended()
        save_session(processor.snapshot())


def show_metrics(snapshot):
    section_heading("Session intelligence", "RECORDED WORKOUT")
    columns = st.columns(3)
    columns[0].metric("Repetitions", sum(m["reps"] for m in snapshot["exercises"]))
    columns[1].metric("Time tracked", format_duration(snapshot["duration_seconds"]))
    coverage = snapshot["valid_pose_frames"] / snapshot["frames"] if snapshot["frames"] else 0
    columns[2].metric("Pose coverage", f"{coverage:.0%}" if snapshot["frames"] else "—")
    st.caption(f'{snapshot["valid_pose_frames"]:,} of {snapshot["frames"]:,} frames with a usable pose · '
               f'{snapshot["duration_seconds"]:.2f} seconds tracked')
    overview, breakdown, export = st.tabs(["Movement overview", "Exercise breakdown", "Export session"])
    with overview:
        session_charts(snapshot)
    with breakdown:
        st.dataframe([{"Exercise": TITLES[m["exercise"]], "Reps": m["reps"],
                       "Duration (s)": round(m["duration_seconds"], 2),
                       "Confidence": m["confidence"]} for m in snapshot["exercises"]],
                     use_container_width=True, hide_index=True,
                     column_config={"Confidence": st.column_config.ProgressColumn(
                         "Model confidence", min_value=0, max_value=1, format="%.3f")})
    with export:
        st.write("Your session, ready to take with you.")
        st.caption("Includes exercise counts, timestamps, duration, confidence and tracking coverage.")
        st.download_button("Download workout telemetry", json.dumps(snapshot, indent=2),
                           file_name="workout.json", mime="application/json", use_container_width=True)
    st.caption("Confidence is the mean winning probability per classification window; manual mode has no classifier confidence.")


def video_page(auto=False):
    page_heading("AUTOMATIC EXERCISE RECOGNITION" if auto else "YOUR PERSONAL VIDEO STUDIO",
                 "Let your movement do the talking." if auto else "A closer look at every rep.",
                 "Upload a workout. See the movement. Leave with a clearer picture.")
    controls, guidance = st.columns([1.65, 1], gap="large")
    with controls, st.container(border=True):
        section_heading("01 / Set up your session")
        mode = "auto" if auto else st.selectbox("Exercise", EXERCISES, format_func=TITLES.get, key="video_exercise")
        if auto:
            html('<div class="inline-note"><span class="status-dot"></span> The model selects the exercise for you.</div>')
        sample = st.toggle("Try the included demo clip", key=f"sample_{auto}")
        if sample:
            upload = BytesIO((ROOT / "assets/videos/bicep_curl_demo.mp4").read_bytes())
            upload.name = "bicep_curl_demo.mp4"
            st.caption("Sample: barbell curl clip · approximately 10 seconds. Counts and probabilities are computed when you analyze it.")
        else:
            upload = st.file_uploader("Workout clip", type=["mp4", "mov", "avi", "asf", "m4v"],
                                      key=f"video_upload_{auto}")
    with guidance:
        html('''<div class="guide-card"><span class="eyebrow">A BETTER VIEW. A BETTER SESSION.</span>
        <h3>Give your movement<br>room to be seen.</h3>
        <div class="guide-row"><b>01</b><span>Keep your full body in frame.</span></div>
        <div class="guide-row"><b>02</b><span>Use steady framing and even lighting.</span></div>
        <div class="guide-row"><b>03</b><span>Choose a clip with one person exercising.</span></div>
        <p>Tracking estimates movement. It does not assess exercise safety or technique.</p></div>''')
    context = (mode, sha256(upload.getbuffer()).hexdigest()) if upload is not None else None
    if st.session_state.get("video_context") != context:
        st.session_state.pop("video_result", None)
        st.session_state["video_context"] = context
    if upload:
        with st.expander("Preview your source video", expanded=False):
            st.video(upload)
    if st.button("Analyze video", disabled=upload is None, type="primary", use_container_width=True):
        st.session_state.pop("video_result", None)
        progress_bar = st.progress(0, text="Preparing your workout…")
        status, preview = st.empty(), st.empty()
        try:
            with TemporaryDirectory(prefix="gym-video-") as directory:
                input_path = Path(directory) / ("input" + Path(upload.name).suffix.lower())
                output_path = Path(directory) / "annotated.mp4"
                input_path.write_bytes(upload.getbuffer())
                status.info("Initializing pose tracking…")
                tracker = make_tracker(mode, "uploaded_video")

                def progress(count, total, frame):
                    status.write(f"Processed {count} / {total or '?'} frames")
                    if total:
                        progress_bar.progress(min(count / total, 1.0), text="Following your movement…")
                    preview.image(frame, channels="BGR", use_column_width=True)

                snapshot = analyze_video(input_path, output_path, tracker, progress)
                save_session(snapshot)
                st.session_state["video_result"] = (snapshot, output_path.read_bytes())
            status.success("Analysis complete")
            progress_bar.progress(1.0, text="Your session is ready to explore.")
        except Exception:
            logger.exception("Video analysis failed")
            st.error("Analysis failed. Check the video and model dependencies. See server logs for details.")
        finally:
            preview.empty()
            progress_bar.empty()
    if "video_result" in st.session_state:
        snapshot, video = st.session_state["video_result"]
        section_heading("02 / Your movement, decoded", "ANALYSIS COMPLETE")
        show_metrics(snapshot)
        with st.expander("Watch annotated playback", expanded=False):
            st.video(video)
            st.download_button("Download annotated video", video, "workout.mp4", "video/mp4")
    elif upload is None:
        empty_state("↗", "Your next insight starts here", "Upload a clip above, or switch on the demo to explore a real analysis.")


def webcam_page():
    from streamlit_webrtc import webrtc_streamer
    from gym_tracker.ui.webcam import WorkoutVideoProcessor
    page_heading("LIVE TRAINING SPACE", "Stay in the moment.",
                 "Your camera. Your movement. A session that grows with every rep.")
    st.caption("Allow browser camera access over HTTPS or localhost. Stop with STOP or join your wrists.")
    custom_ice = setting("WEBRTC_ICE_SERVERS")
    connection = st.selectbox("Camera connection", ["This computer / local network", "Remote server / cloud"],
                              index=1 if custom_ice else 0, key="camera_connection")
    local = connection == "This computer / local network"
    st.caption("Local mode connects directly without an external connection server. "
               "Choose Remote for a deployed app; some networks require a configured TURN server.")
    with st.expander("Camera stuck loading?"):
        st.write("For a local app, use the localhost address shown when Streamlit starts and choose This computer / local network. "
                 "Allow camera access in your browser's site settings, then press START. "
                 "Use SELECT DEVICE if you have more than one camera. Close other apps using the camera.")
        st.caption("A remote deployment must use HTTPS. Configure WEBRTC_ICE_SERVERS with your own "
                   "STUN/TURN service if the remote connection cannot complete.")
    mode = st.selectbox("Exercise", (*EXERCISES, "auto"), format_func=TITLES.get, key="webcam_exercise")
    if mode == "shoulder_press":
        st.caption("Keep both shoulders, elbows and wrists visible, including at full overhead extension. "
                   "The counter needs bent elbows followed by both arms extending overhead; legs can be outside the frame.")
    elif mode == "auto":
        st.caption("Auto recognition requires your full body in frame. For upper-body framing, select curls or shoulder presses directly.")
    elif mode == "bicep_curl":
        st.caption("Keep both shoulders, elbows and wrists visible. Extend both arms, then curl both arms to count a rep.")
    elif mode == "push_up":
        st.caption("Use a side view with a shoulder, elbow and wrist visible. The counter counts on the descent after arm extension.")
    elif mode == "squat":
        st.caption("Use a side view with a hip, knee and ankle visible. Stand tall before descending; the counter counts on the descent.")
    connection_id = "local" if local else sha256(custom_ice.encode()).hexdigest()[:8]
    if st.session_state.get("webcam_connection_id", connection_id) != connection_id:
        finish_webcam()
    st.session_state["webcam_connection_id"] = connection_id
    previous = st.session_state.get("last_processor")
    if previous is not None and previous.mode != mode:
        finish_webcam()
    ice_servers = [] if local else [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        if custom_ice and not local:
            ice_servers = json.loads(custom_ice)
            if not isinstance(ice_servers, list) or not all(
                    isinstance(server, dict) and server.get("urls") for server in ice_servers):
                raise ValueError("WEBRTC_ICE_SERVERS must be a JSON array of servers with urls")
        stream_key = f"gym-{mode}" if local else f"gym-{mode}-{connection_id}"
        ctx = webrtc_streamer(key=stream_key,
                              video_processor_factory=lambda: WorkoutVideoProcessor(mode, make_tracker),
                              rtc_configuration={"iceServers": ice_servers},
                              media_stream_constraints={"video": {
                                  "width": {"ideal": 640}, "height": {"ideal": 480},
                                  "frameRate": {"ideal": 24, "max": 30},
                                  "facingMode": "user"}, "audio": False},
                              async_processing=True)
    except Exception:
        logger.exception("WebRTC initialization failed")
        st.error("Webcam could not start. Check camera permission and ICE configuration.")
        return
    if ctx.video_processor:
        if st.session_state.get("last_processor") is not ctx.video_processor:
            finish_webcam()
        st.session_state["last_processor"] = ctx.video_processor
    st.button("Refresh workout metrics")
    processor = st.session_state.get("last_processor")
    if processor:
        snapshot = processor.snapshot()
        save_session(snapshot)
        if processor.error:
            st.error(processor.error)
        elif processor.finished:
            st.info("Session ended. Use STOP then START to begin a new workout.")
        if snapshot:
            show_metrics(snapshot)
    else:
        framing = {"shoulder_press": "both arms remain visible overhead",
                   "bicep_curl": "both arms are visible",
                   "push_up": "your shoulder, elbow and wrist are visible from the side",
                   "squat": "your hip, knee and ankle are visible from the side",
                   "auto": "your full body is visible"}[mode]
        empty_state("◉", "The floor is yours", f"Press START, allow camera access, and step back until {framing}.")


def coach_page():
    page_heading("CONTEXT. CLARITY. YOUR NEXT STEP.", "Meet your session coach.",
                 "A conversation grounded in the workout you actually recorded.")
    sessions = st.session_state.get("sessions", {})
    if not sessions:
        empty_state("✳", "Good coaching starts with your movement", "Record a session to give your coach something meaningful to work with.")
        st.info("Analyze a video or record a webcam workout first.")
        st.button("Explore a demo workout", on_click=navigate, args=("Auto Classify", True), type="primary")
        return
    selected = st.selectbox("Workout session", list(reversed(sessions)),
                            format_func=lambda key: session_label(sessions[key]))
    snapshot = sessions[selected]
    provider, model = setting("COACH_PROVIDER", "openrouter"), setting("COACH_MODEL")
    html('<div class="coach-intro"><span class="coach-icon">✳</span><div><h3>Let’s make sense of your session.</h3>'
         '<p>Ask about your rep counts, consistency or tracking coverage. Your coach sees session metrics, not your video.</p></div></div>')
    with st.expander("What your coach can see"):
        st.caption(f"Provider: {provider} · Model: {model or 'not configured'}. Requests send aggregate metrics and your question to this provider.")
        st.write("Repetitions, observed duration, model confidence and tracking coverage. No raw video is shared.")
    question = st.text_input("Ask about this workout", "Summarize this session and suggest one next step.", max_chars=2000)
    context = (selected, json.dumps(snapshot, sort_keys=True), question, provider, model)
    if st.button("Get session coaching", type="primary", use_container_width=True):
        st.session_state.pop("coaching_reply", None)
        try:
            with st.spinner("Reviewing session telemetry…"):
                reply = coach(snapshot, question, provider=provider, model=model,
                              api_key=setting(f"{provider.upper()}_API_KEY"))
            st.session_state["coaching_reply"] = (context, reply)
        except Exception:
            logger.exception("Coaching request failed")
            st.error("Coaching unavailable. Check the provider, model, API key and service quota.")
    reply = st.session_state.get("coaching_reply")
    if reply and reply[0] == context:
        with st.chat_message("assistant", avatar="✳"):
            st.markdown(reply[1])
    with st.expander("Review the workout behind this conversation"):
        show_metrics(snapshot)


def dashboard_page():
    page_heading("YOUR TRAINING SPACE", "Overview", "Small details. Stronger sessions.")
    motion_hero()
    actions = st.columns([1.15, 1.15, 1.7])
    actions[0].button("Analyze a workout ↗", on_click=navigate, args=("Video Analysis",),
                      type="primary", use_container_width=True)
    actions[1].button("Try the live demo ✧", on_click=navigate, args=("Auto Classify", True),
                      use_container_width=True)
    with actions[2]:
        html('<div class="action-caption">No wearable needed. Just your movement.</div>')
    sessions = list(st.session_state.get("sessions", {}).values())
    section_heading("Your training at a glance", "THIS BROWSER SESSION")
    counts = st.columns(3)
    counts[0].metric("Sessions recorded", len(sessions))
    counts[1].metric("Reps logged", sum(m["reps"] for s in sessions for m in s["exercises"]))
    counts[2].metric("Time tracked", format_duration(sum(s["duration_seconds"] for s in sessions)))
    main_col, side_col = st.columns([1.65, 1], gap="large")
    with main_col:
        section_heading("Your momentum", "EVERY SESSION COUNTS")
        if sessions:
            history_chart(sessions)
        else:
            empty_state("▥", "A fresh start. A clear direction.",
                        "Your workout history will grow here. Analyze your first clip to start seeing the bigger picture.")
        st.button("Open workout history →", on_click=navigate, args=("Workout History",),
                  use_container_width=True)
    with side_col:
        section_heading("The next step is yours", "GET STARTED")
        html('<div class="feature-card"><span class="feature-number">01 / CAPTURE</span>'
             '<h3>Bring your workout<br>into focus.</h3><p>Review a recorded clip or train with your browser camera. '
             'Get a visual account of your movement.</p></div>')
        st.button("Start a camera session ◉", on_click=navigate, args=("WebCam Live",),
                  use_container_width=True)
    section_heading("Four movements. One training space.", "EXERCISE LIBRARY")
    descriptions = {"bicep_curl": ("01", "Both arms. One complete curl.", "Upper body"),
                    "push_up": ("02", "Follow every press from the floor.", "Bodyweight"),
                    "squat": ("03", "Make your lower-body work visible.", "Lower body"),
                    "shoulder_press": ("04", "Track your overhead movement.", "Upper body")}
    for column, exercise in zip(st.columns(4), EXERCISES):
        number, description, category = descriptions[exercise]
        with column:
            html(f'<div class="exercise-card"><span class="exercise-index">{number}</span>'
                 f'<span class="exercise-category">{category}</span><h3>{TITLES[exercise]}</h3>'
                 f'<p>{description}</p></div>')
    with st.expander("See the original demonstration clip"):
        st.video(str(ROOT / "assets/videos/bicep_curl_demo.mp4"))
    st.caption("Counts are movement estimates. Model confidence is not measured accuracy or a form assessment.")


def history_page():
    page_heading("THE BIGGER PICTURE", "Build your momentum.",
                 "Revisit your sessions. Explore your movement. Take your data with you.")
    sessions = st.session_state.get("sessions", {})
    if not sessions:
        empty_state("▥", "Your story is still ahead of you", "Completed workouts will appear here, ready to revisit and compare.")
        st.button("Record your first session ↗", on_click=navigate, args=("Video Analysis",), type="primary")
        return
    choices = st.multiselect("Focus on exercises", EXERCISES, format_func=TITLES.get,
                             placeholder="All exercises", key="history_filter")
    filtered = {key: value for key, value in sessions.items()
                if not choices or any(m["exercise"] in choices and
                                      (m["duration_seconds"] > 0 or m["reps"] > 0)
                                      for m in value["exercises"])}
    if not filtered:
        st.info("No recorded sessions match these exercises yet.")
        return
    section_heading("Session rhythm", f"{len(filtered)} SESSIONS")
    history_chart(list(filtered.values()))
    selected = st.selectbox("Explore a session", list(reversed(filtered)),
                            format_func=lambda key: session_label(filtered[key]), key="history_session")
    show_metrics(filtered[selected])
    st.caption("Up to 20 sessions stay in this browser session. Download telemetry to keep a permanent copy.")


def main():
    load_dotenv()
    st.set_page_config(page_title="APEX · Movement Intelligence", page_icon="◈", layout="wide",
                       initial_sidebar_state="expanded")
    html(f'<style>{Path(__file__).with_name("styles.css").read_text()}</style>')
    with st.sidebar:
        html('<div class="brand"><span class="brand-mark">A<span></span></span>'
             '<div>APEX<small>MOVEMENT INTELLIGENCE</small></div></div>')
        html('<div class="nav-caption">YOUR WORKSPACE</div>')
        page = st.radio("Workspace", PAGES, format_func=PAGE_LABELS.get,
                        key="navigation", label_visibility="collapsed")
        html('<div class="sidebar-note"><span class="status-dot"></span> YOUR MOVEMENT. YOUR SPACE.'
             '<p>Turn effort into insight.<br>One session at a time.</p></div>')
        html('<div class="sidebar-footer">MEDIAPIPE + BiLSTM<br><span>Built for a clearer view of your training.</span></div>')
        with st.expander("About & credits"):
            st.markdown("**APEX · AI Gym Tracker**")
            st.caption("Created by")
            st.markdown("**P SAIDEEP REDDY**")
            st.caption("Workout tracking, session analytics and coaching in one application.")
            st.caption("Built with Streamlit, MediaPipe and TensorFlow/Keras. "
                       "Model artifact details and acknowledgements are documented in the repository.")
    html('<div class="topbar"><span>APEX / TRAINING INTELLIGENCE</span>'
         '<span class="topbar-pill">◈ &nbsp; MOVE WITH INTENTION</span></div>')
    if page != "WebCam Live":
        finish_webcam()
    views = {"Dashboard": dashboard_page, "Video Analysis": video_page,
             "WebCam Live": webcam_page, "Auto Classify": lambda: video_page(auto=True),
             "Workout History": history_page, "AI Coach": coach_page}
    views[page]()
    html('<div class="page-footer"><span>APEX <b> / </b> THE INTELLIGENCE IN YOUR MOVEMENT</span>'
         '<span>Created by P SAIDEEP REDDY</span></div>')
