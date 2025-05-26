🏋️‍♂️ AI Gym Tracker — Real-Time Exercise Form Analyzer with Pose Estimation

AI Gym Tracker is an AI-powered fitness monitoring application that uses human pose estimation and machine learning to track workout repetitions and provide real-time feedback on form and posture. Designed for accessibility and ease of use, it leverages MediaPipe, TensorFlow, and OpenCV, with a clean, interactive interface built using Streamlit.

🚀 Features

Real-Time Pose Estimation using MediaPipe's holistic model

Exercise Detection & Rep Counting for bicep curls and other custom exercises

Form Feedback Mechanism to help correct user posture

Video Input Support for live webcam feed or pre-recorded workout videos

Voice Feedback (optional) for rep announcements using gTTS and playsound

Data Visualization with matplotlib and seaborn (if analytics are added)

Lightweight Streamlit UI for ease of deployment and testing

🧰 Tech Stack & Libraries

Python 3.9

TensorFlow for loading and running ML models

MediaPipe for pose detection and landmark extraction

OpenCV for video processing and annotation

Streamlit for building the web UI

NumPy & Pandas for data manipulation

Scikit-learn & Optuna (optional) for training and hyperparameter tuning

Matplotlib / Seaborn for data visualization (if extended)

gTTS & SpeechRecognition (optional) for voice output and input

🗂 Folder Structure

📁 ai-gym-tracker/
├── main.py                 # Streamlit entry point
├── ExerciseAiTrainer.py    # Exercise logic and pose evaluation
├── models/                 # Trained ML models
├── utils/                  # Helper functions (if separated)
├── assets/                 # Demo videos, images, icons
├── requirements.txt        # Python dependencies
└── README.md               # Project overview

✅ Getting Started

Clone the Repository

git clone https://github.com/your-username/ai-gym-tracker.git
cd ai-gym-tracker

Create and Activate Conda Environment

conda env create -f environment.yml
conda activate tf_streamlit

Run the Application
streamlit run main.py
