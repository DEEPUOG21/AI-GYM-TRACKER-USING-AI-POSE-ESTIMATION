import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
from AiTrainer_utils import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant landmarks indices
relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55, font_color=(255, 255, 255), font_thickness=2, bg_color=(0, 0, 0), padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)


def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    right_shoulder = landmark_list[12][1:]
    right_wrist = landmark_list[16][1:]
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    left_shoulder = landmark_list[11][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)

    if left_arm_angle < 220:
        stage = "down"
    if left_arm_angle > 240 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter


def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    right_leg = landmark_list[26][1:]
    exercise_instance.visualize_angle(img, right_leg_angle, right_leg)

    if right_leg_angle > 160 and left_leg_angle < 220:
        stage = "down"
    if right_leg_angle < 140 and left_leg_angle > 210 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])

    if right_arm_angle > 160 and right_arm_angle < 200:
        stage_right = "down"
    if left_arm_angle < 200 and left_arm_angle > 140:
        stage_left = "down"
    
    if stage_right == "down" and (right_arm_angle > 310 or right_arm_angle < 60) and (left_arm_angle > 310 or left_arm_angle < 60) and stage_left == "down":
        stage_right = "up"
        stage_left = "up"
        counter += 1
    
    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_elbow = landmark_list[14][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_elbow)

    if right_arm_angle > 280 and left_arm_angle < 80:
        stage = "down"
    if right_arm_angle < 240 and left_arm_angle > 120 and stage == "down":
        stage = "up"
        counter += 1
    
    return stage, counter


class Exercise:
    def __init__(self):
        try:
            self.lstm_model = load_model('final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.lstm_model = None
        
        try:
            self.scaler = joblib.load('thesis_bidirectionallstm_scaler.pkl')
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
        
        try:
            self.label_encoder = joblib.load('thesis_bidirectionallstm_label_encoder.pkl')
            self.exercise_classes = self.label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = None
            self.exercise_classes = []

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))

            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),
                calculate_distance(landmarks[18:21], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30]),
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[6:9], landmarks[24:27]),
                calculate_distance(landmarks[9:12], landmarks[27:30]),
                calculate_distance(landmarks[12:15], landmarks[0:3]),
                calculate_distance(landmarks[15:18], landmarks[3:6]),
                calculate_distance(landmarks[12:15], landmarks[18:21]),
                calculate_distance(landmarks[15:18], landmarks[21:24])
            ]

            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),
                calculate_y_distance(landmarks[9:12], landmarks[3:6])
            ]

            normalization_factor = -1
            distances_to_check = [
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30])
            ]

            for distance in distances_to_check:
                if distance > 0:
                    normalization_factor = distance
                    break
            
            if normalization_factor == -1:
                normalization_factor = 0.5

            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

            features.extend(normalized_distances)
            features.extend(normalized_y_distances)

        else:
            features = [-1.0] * 22
        return features
    
    def preprocess_frame(self, frame, pose):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def visualize_angle(self, img, angle, landmark):
        cx, cy = int(landmark[0]), int(landmark[1])
        h, w = img.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        cv2.putText(img, str(int(angle)),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def auto_classify_and_count(self):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening webcam.")
            return

        window_size = 30
        landmarks_window = []
        frame_count = 0
        current_prediction = "No prediction yet"
        counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}

        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.preprocess_frame(frame, pose)
            if len(landmarks) == len(relevant_landmarks_indices) * 3:
                features = self.extract_features(landmarks)
                if len(features) == 22:
                    landmarks_window.append(features)

            frame_count += 1

            if len(landmarks_window) == window_size:
                landmarks_window_np = np.array(landmarks_window).flatten().reshape(1, -1)
                scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22)
                prediction = self.lstm_model.predict(scaled_landmarks_window)

                if prediction.shape[1] != len(self.exercise_classes):
                    return

                predicted_class = np.argmax(prediction, axis=1)[0]
                if predicted_class >= len(self.exercise_classes):
                    return

                current_prediction = self.exercise_classes[predicted_class]
                landmarks_window = []
                frame_count = 0

            detector.find_person(frame, draw=True)
            landmark_list = detector.find_landmarks(frame, draw=True)
            if len(landmark_list) > 0:
                if self.are_hands_joined(landmark_list, stop=True):
                    break

                if current_prediction == 'push-up':
                    stages['push_up'], counters['push_up'] = count_repetition_push_up(detector, frame, landmark_list, stages['push_up'], counters['push_up'], self)
                elif current_prediction == 'squat':
                    stages['squat'], counters['squat'] = count_repetition_squat(detector, frame, landmark_list, stages['squat'], counters['squat'], self)
                elif current_prediction == 'barbell biceps curl':
                    stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'] = count_repetition_bicep_curl(detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], counters['bicep_curl'], self)
                elif current_prediction == 'shoulder press':
                    stages['shoulder_press'], counters['shoulder_press'] = count_repetition_shoulder_press(detector, frame, landmark_list, stages['shoulder_press'], counters['shoulder_press'], self)

            exercise_name_map = {
                'push_up': 'Push-up',
                'squat': 'Squat',
                'bicep_curl': 'Curl',
                'shoulder_press': 'Press'
            }

            height, width, _ = frame.shape
            num_exercises = len(counters)
            vertical_spacing = height // (num_exercises + 1)

            cv2.rectangle(frame, (0, 0), (120, height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)

            short_name = exercise_name_map.get(current_prediction, current_prediction)
            draw_styled_text(frame, f"Exercise: {short_name}", ((width - 290) // 2 + 100, 20))

            for idx, (exercise, count) in enumerate(counters.items()):
                short_name = exercise_name_map.get(exercise, exercise)
                draw_styled_text(frame, f"{short_name}: {count}", (10, (idx + 1) * vertical_spacing))

            stframe.image(frame, channels='BGR', use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def are_hands_joined(self, landmark_list, stop, is_video=False):
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            stop = True
            return stop
        return False

    def visualize_angle(self, img, angle, landmark):
        cx, cy = int(landmark[0]), int(landmark[1])
        h, w = img.shape[:2]
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        cv2.putText(img, str(int(angle)),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def repetitions_counter(self, img, counter):
        cv2.rectangle(img, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(img, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    def push_up(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage)

    def squat(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage)

    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left)

    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage)

    def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None):
        if is_video:
            import tempfile, os
            stframe = st.empty()
            progress_text = st.empty()
            detector = pm.posture_detector()

            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out_path = out_file.name
            out_file.close()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, original_fps, (width, height))

            frame_count = 0
            progress_text.text("Processing video... please wait.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                img = detector.find_person(frame)
                landmark_list = detector.find_landmarks(img, draw=False)

                if len(landmark_list) != 0:
                    if multi_stage:
                        stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                    else:
                        stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                    if self.are_hands_joined(landmark_list, stop=False, is_video=is_video):
                        break

                self.repetitions_counter(img, counter)
                out.write(img)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            progress_text.text("Re-encoding for browser...")
            h264_path = out_path.replace('.mp4', '_h264.mp4')
            os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -pix_fmt yuv420p -acodec aac {h264_path}")

            progress_text.text("Processing complete!")

            if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                with open(h264_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
            else:
                with open(out_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)

        else:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            detector = pm.posture_detector()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img = detector.find_person(frame)
                landmark_list = detector.find_landmarks(img, draw=False)

                if len(landmark_list) != 0:
                    if multi_stage:
                        stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                    else:
                        stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)

                    if self.are_hands_joined(landmark_list, stop=False):
                        break

                self.repetitions_counter(img, counter)
                stframe.image(img, channels='BGR', use_column_width=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
