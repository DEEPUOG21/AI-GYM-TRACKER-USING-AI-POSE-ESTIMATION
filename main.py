import sys
import site
import subprocess
import os

# Step 1: Install numpy and cv2 to user site-packages
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.76"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "mediapipe==0.10.14"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "joblib", "scikit-learn==1.5.0"], check=False)

# Step 2: Force user site-packages to front of sys.path BEFORE any numpy import
user_site = site.getusersitepackages()
# Remove any existing numpy from sys.path to avoid conda's numpy 2.x loading
sys.path = [p for p in sys.path if 'numpy' not in p]
# Insert user site-packages at position 0 so it wins over conda
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Step 3: Force reload numpy from user path
import importlib
import numpy
importlib.reload(numpy)

# Step 4: Now import cv2
import cv2

import streamlit as st
import tempfile
import ExerciseAiTrainer as exercise
from chatbot import chat_ui
import time

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
        st.write(' Click button to start training')
        start_button = st.button('Start Exercise')
        if start_button:
            time.sleep(2)
            ready = True
            if exercise_general == 'Bicept Curl':
                while ready:
                    cap = cv2.VideoCapture(0)
                    exer = exercise.Exercise()
                    counter, stage_right, stage_left = 0, None, None
                    exer.bicept_curl(cap, counter=counter, stage_right=stage_right, stage_left=stage_left)
                    break
            elif exercise_general == 'Push Up':
                while ready:
                    cap = cv2.VideoCapture(0)
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.push_up(cap, counter=counter, stage=stage)
                    break
            elif exercise_general == 'Squat':
                while ready:
                    cap = cv2.VideoCapture(0)
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.squat(cap, counter=counter, stage=stage)
                    break
            elif exercise_general == 'Shoulder Press':
                while ready:
                    cap = cv2.VideoCapture(0)
                    exer = exercise.Exercise()
                    counter, stage = 0, None
                    exer.shoulder_press(cap, counter=counter, stage=stage)
                    break

if __name__ == '__main__':
    def load_css():
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    main()
