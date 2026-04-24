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

    options = st.sidebar.selectbox(
        'Select Option',
        ('Video', 'WebCam', 'Auto Classify', 'chatbot')
    )

    # ---------------- CHATBOT ----------------
    if options == 'chatbot':
        st.markdown('-------')
        st.markdown("The chatbot can make mistakes. Check important info.")
        chat_ui()

    # ---------------- VIDEO ----------------
    elif options == 'Video':
        st.markdown('-------')
        st.write('## Upload your video and select the correct type of Exercise to count repetitions')

        exercise_options = st.sidebar.selectbox(
            'Select Exercise', ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )

        video_file_buffer = st.sidebar.file_uploader(
            "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v']
        )

        if video_file_buffer is None:
            st.warning("Please upload a video to proceed.")
            return

        tfflie = tempfile.NamedTemporaryFile(delete=False)
        tfflie.write(video_file_buffer.read())

        cap = cv2.VideoCapture(tfflie.name)

        st.video(tfflie.name)
        st.markdown('## Output Video')

        exer = exercise.Exercise()

        if exercise_options == 'Bicept Curl':
            exer.bicept_curl(cap, is_video=True, counter=0, stage_right=None, stage_left=None)

        elif exercise_options == 'Push Up':
            exer.push_up(cap, is_video=True, counter=0, stage=None)

        elif exercise_options == 'Squat':
            exer.squat(cap, is_video=True, counter=0, stage=None)

        elif exercise_options == 'Shoulder Press':
            exer.shoulder_press(cap, is_video=True, counter=0, stage=None)

    # ---------------- AUTO CLASSIFY ----------------
    elif options == 'Auto Classify':
        st.markdown('-------')
        st.write('Click button to start automatic exercise classification')

        if st.button('Start Auto Classification'):
            exer = exercise.Exercise()
            exer.auto_classify_and_count()

    # ---------------- WEBCAM ----------------
    elif options == 'WebCam':
        st.markdown('-------')

        st.warning("⚠️ Webcam may be unstable on Streamlit Cloud. Use Video Upload if issues occur.")

        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        # ✅ FIX 1: Stable ICE servers (TURN added)
        RTC_CONFIGURATION = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                ]
            }
        )

        exercise_general = st.selectbox(
            'Select Exercise',
            ('Bicept Curl', 'Push Up', 'Squat', 'Shoulder Press')
        )

        # ✅ FIX 2: Prevent re-initialization
        if "webrtc_started" not in st.session_state:
            st.session_state.webrtc_started = True

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
                    exer_inst = exercise.Exercise()

                    if self.exercise_name == 'Bicept Curl':
                        from ExerciseAiTrainer import count_repetition_bicep_curl
                        self.stage_right, self.stage_left, self.counter = count_repetition_bicep_curl(
                            self.detector, img, landmark_list,
                            self.stage_right, self.stage_left, self.counter, exer_inst)

                    elif self.exercise_name == 'Push Up':
                        from ExerciseAiTrainer import count_repetition_push_up
                        self.stage, self.counter = count_repetition_push_up(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)

                    elif self.exercise_name == 'Squat':
                        from ExerciseAiTrainer import count_repetition_squat
                        self.stage, self.counter = count_repetition_squat(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)

                    elif self.exercise_name == 'Shoulder Press':
                        from ExerciseAiTrainer import count_repetition_shoulder_press
                        self.stage, self.counter = count_repetition_shoulder_press(
                            self.detector, img, landmark_list,
                            self.stage, self.counter, exer_inst)

                # Draw counter
                cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(img, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(img, str(self.counter), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ✅ FIX 3: STATIC KEY (CRITICAL)
        webrtc_streamer(
            key="stable-webcam",  # NEVER dynamic
            video_processor_factory=ExerciseProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )


if __name__ == '__main__':
    main()
