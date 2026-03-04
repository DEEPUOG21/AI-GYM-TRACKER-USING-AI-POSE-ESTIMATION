# 🏋️ AI Gym Tracker — Smart Fitness Coach using AI Pose Estimation

🚀 An AI-powered fitness assistant that automatically **detects exercises, counts repetitions, and provides real-time workout guidance** using Computer Vision and Deep Learning.

The system combines **MediaPipe Pose Estimation + BiLSTM Models + Large Language Models (LLMs)** to create a complete AI personal trainer experience.

---

## 🌟 Project Highlights

✅ Real-time Exercise Recognition  
✅ Automatic Rep Counting using Pose Angles  
✅ Webcam & Video Workout Analysis  
✅ AI Fitness Chatbot Coach  
✅ Deep Learning Exercise Classification  
✅ Streamlit Interactive Dashboard  

---

## 🧠 How It Works
Camera / Video Input
↓
MediaPipe Pose Detection
↓
Feature Extraction (Angles + Landmarks)
↓
BiLSTM Exercise Classification
↓
Repetition Counting Logic
↓
Streamlit AI Fitness Dashboard


---

## 🎬 Demo

## ⚠️ Demo Availability

Due to deployment limitations with Streamlit and certain computer vision dependencies, the live demo for this project is currently unavailable.

The full implementation and source code are provided in this repository.

For a reference on how the system works, you may watch a demonstration video of a similar AI pose estimation–based fitness tracking application.

If required, I would be happy to provide a walkthrough video explaining the functionality and architecture of this project.



---

## 🖥 Application Modes

### 🎥 Webcam Trainer
Perform exercises live while AI tracks movements and repetitions.

---

### 📹 Video Analysis
Upload recorded workouts and automatically analyze performance.

---

### 🤖 Auto Exercise Detection
AI automatically identifies exercises without manual selection.

Supported Exercises:
- Bicep Curl
- Push Ups
- Squats
- Shoulder Press

---

### 💬 AI Fitness Chatbot

Integrated AI assistant capable of answering:

- Workout planning
- Fat loss strategies
- Nutrition advice
- Exercise techniques
- Recovery guidance

Powered via LLM APIs (OpenRouter / OpenAI).

---

## 📂 Project Architecture
AI-GYM-TRACKER
│
├── main.py
├── chatbot.py
├── ExerciseAiTrainer.py
├── PoseModule2.py
├── AiTrainer_utils.py
│
├── ML Training
│ ├── extract_features.py
│ ├── create_sequence_of_features.py
│ └── train_bidirectionallstm.py
│
├── Models
│ ├── classifier_model.h5
│ ├── scaler.pkl
│ └── encoder.pkl
│
├── static/
├── requirements.txt
└── README.md

---

##🧬 Machine Learning Model

Model Used:

✅ Bidirectional LSTM (BiLSTM)

Input Features:

Joint Angles
Pose Coordinates
Normalized Distances

Optimized Using:

Accuracy
Precision
Recall
F1 Score

🛠 Tech Stack
Computer Vision
MediaPipe
OpenCV
Deep Learning
TensorFlow
LSTM / BiLSTM

AI Assistant

OpenAI API
OpenRouter

Frontend

Streamlit

Language

Python

🚀 Future Improvements

Posture Correction Feedback

Workout Analytics Dashboard

Mobile Deployment

Cloud AI Inference

Multi-person Tracking

👨‍💻 Author

Deepu Saideep

AI & Data Science Engineer

GitHub:
https://github.com/DEEPUOG21
