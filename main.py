import subprocess, sys

# Install required packages into conda env
# Install into conda env
conda_pip = "/home/adminuser/.conda/bin/pip"
subprocess.run([conda_pip, "install", "-q",
                "imageio==2.34.0", "imageio-ffmpeg==0.5.1"], check=False)
subprocess.run([conda_pip, "install", "-q",
                "av", "streamlit-webrtc==0.47.1"], check=False)
subprocess.run([conda_pip, "install", "-q",
                "opencv-python-headless==4.9.0.80"], check=False)

import streamlit as st
import cv2
import tempfile
import ExerciseAiTrainer as exercise
from chatbot import chat_ui
import time
import numpy as np
import PoseModule2 as pm

def main():
    st.set_page_config(page_title='Fitness AI Coach', layout='centered')
    st.title('Fitness AI Coach')

    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam', 'Auto Classify', 'chatbot'))

    if options == 'chatbot':
        st.markdown('-------')
        st.markdown("The chatbot can make mistakes. Check important info.")
        chat_ui()

    if options == 'Video':
        st.markdown('-------')
        st.write('## Upload your video and select the correct type of Exercise to count repetitions')
        st.write("")
        st.write('Please ensure you are clearly visible and facing the camera directly. This will help the AI accurately track your movements.')
        st.sidebar.markdown('-------')
        exercise_options = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        st.sidebar.markdown('-------')
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)
        if video_file_buffer is not None:
            tfflie.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tfflie.name)
        else:
            st.warning("Please upload a video to proceed.")
            return
        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        st.markdown('## Input Video')
        st.video(tfflie.name)
        st.markdown('-------')
        st.markdown(' ## Output Video')
        if exercise_options == 'Bicept Curl':
            exer = exercise.Exercise()
            counter, stage_right, stage_left = 0, None, None
            exer.bicept_curl(cap, is_video=True, counter=counter, stage_right=stage_right, stage_left=stage_left)
        elif exercise_options == 'Push Up':
            st.write("The exercise need to be filmed showing your left side or facing frontally")
            exer = exercise.Exercise()
            counter, stage = 0, None
            exer.push_up(cap, is_video=True, counter=counter, stage=stage)
        elif exercise_options == 'Squat':
            exer = exercise.Exercise()
            counter, stage = 0, None
            exer.squat(cap, is_video=True, counter=counter, stage=stage)
        elif exercise_options == 'Shoulder Press':
            exer = exercise.Exercise()
            counter, stage = 0, None
            exer.shoulder_press(cap, is_video=True, counter=counter, stage=stage)

    elif options == 'Auto Classify':
        st.markdown('-------')
        st.write('Click button to start automatic exercise classification and repetition counting')
        st.markdown('-------')
        st.write("Please ensure you are clearly visible and facing the camera directly. This will help the AI accurately track your movements.")
        auto_classify_button = st.button('Start Auto Classification')
        if auto_classify_button:
            time.sleep(2)
            exer = exercise.Exercise()
            exer.auto_classify_and_count()

    elif options == 'WebCam':
        st.markdown('-------')
        st.sidebar.markdown('-------')
        exercise_general = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )
        st.write('Your browser camera will be used for live exercise tracking.')
        st.markdown('-------')

        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Store state in session
        if 'counter' not in st.session_state:
            st.session_state.counter = 0
        if 'stage' not in st.session_state:
            st.session_state.stage = None
        if 'stage_right' not in st.session_state:
            st.session_state.stage_right = None
        if 'stage_left' not in st.session_state:
            st.session_state.stage_left = None

        class ExerciseProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = pm.posture_detector()
                self.counter = 0
                self.stage = None
                self.stage_right = None
                self.stage_left = None
                self.exercise_name = exercise_general

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                img = self.detector.find_person(img)
                landmark_list = self.detector.find_landmarks(img, draw=False)

                if len(landmark_list) != 0:
                    if self.exercise_name == 'Bicept Curl':
                        from ExerciseAiTrainer import count_repetition_bicep_curl
                        exer_inst = exercise.Exercise()
                        self.stage_right, self.stage_left, self.counter = count_repetition_bicep_curl(
                            self.detector, img, landmark_list,
                            self.stage_right, self.stage_left, self.counter, exer_inst)
                    elif self.exercise_name == 'Push Up':
                        from ExerciseAiTrainer import count_repetition_push_up
                        exer_inst = exercise.Exercise()
                        self.stage, self.counter = count_repetition_push_up(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)
                    elif self.exercise_name == 'Squat':
                        from ExerciseAiTrainer import count_repetition_squat
                        exer_inst = exercise.Exercise()
                        self.stage, self.counter = count_repetition_squat(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)
                    elif self.exercise_name == 'Shoulder Press':
                        from ExerciseAiTrainer import count_repetition_shoulder_press
                        exer_inst = exercise.Exercise()
                        self.stage, self.counter = count_repetition_shoulder_press(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)

                # Draw rep counter on frame
                cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(img, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(self.counter), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key=f"exercise-{exercise_general}",
            video_processor_factory=ExerciseProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

if __name__ == '__main__':
    def load_css():
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    main()
