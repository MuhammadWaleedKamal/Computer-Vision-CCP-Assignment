# The video is live recorded from webcam.
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Path and save configurations
video_source = 0  # 0 for live webcam
# video_source = "vid2.mp4"
model_path = "absolute\\path\\to\\pose_landmarker_full.task"  
output_file = "output_processed.mp4"  
 
# Modern Tasks API Setup of MediaPipe PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Smoothing Filter
class KeypointSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        # Store history for critical joints to reduce coordinate jitter
        self.history = {i: deque(maxlen=window_size) for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]}

    def smooth(self, idx, pt):
        if idx not in self.history:
            return pt
        self.history[idx].append(pt)
        xs = [p[0] for p in self.history[idx]]
        ys = [p[1] for p in self.history[idx]]
        return (int(np.mean(xs)), int(np.mean(ys)))


# Angle math function
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

# Ground truth
GROUND_TRUTH_TIMELINE = {
    (1, 100): "Standing",
    (101, 300): "Raising Arm",
    (301, 500): "Squatting",
}

def get_ground_truth_activity(current_frame):
    for (start, end), activity in GROUND_TRUTH_TIMELINE.items():
        if start <= current_frame <= end:
            return activity
        
    return "Standing"

# Tracking Arrays for Data Plots
frame_numbers = []
knee_angles = []
hip_angles = []
elbow_angles = []
running_accuracies = []

# Evaluation counters
frame_idx = 0
correct_predictions = 0
total_evaluated_frames = 0

# Video capture and writer setup
cap = cv2.VideoCapture(video_source)

if cap.get(cv2.CAP_PROP_FPS) > 0:
    fps = cap.get(cv2.CAP_PROP_FPS)
    
else:
    fps = 30
    
width, height = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
smoother = KeypointSmoother(window_size=5)

with PoseLandmarker.create_from_options(options) as landmarker:    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame_idx += 1
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int((frame_idx / fps) * 1000)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        activity = "Standing"
        joint_color = (0, 255, 0)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            # Extract, scale, and apply the smoothing filter
            def get_smoothed_joint(idx):
                raw_pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                return smoother.smooth(idx, raw_pt)
            
            # Base skeleton overlay visualization
            for i in range(len(landmarks)):
                pt = (int(landmarks[i].x * w), int(landmarks[i].y * h))
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
            
            # Capture structural tracks
            l_shoulder = get_smoothed_joint(11)
            l_elbow = get_smoothed_joint(13)
            l_wrist = get_smoothed_joint(15)
            l_hip = get_smoothed_joint(23)
            l_knee = get_smoothed_joint(25)
            l_ankle = get_smoothed_joint(27)
            
            r_shoulder = get_smoothed_joint(12)
            r_elbow = get_smoothed_joint(14)
            r_wrist = get_smoothed_joint(16)
            r_hip = get_smoothed_joint(24)
            r_knee = get_smoothed_joint(26)
            r_ankle = get_smoothed_joint(28)
            
            # Compute Joint Angles
            knee_ang = get_angle(l_hip, l_knee, l_ankle)
            hip_ang = get_angle(l_shoulder, l_hip, l_knee)
            elbow_ang = get_angle(l_shoulder, l_elbow, l_wrist)
            
            frame_numbers.append(frame_idx)
            knee_angles.append(knee_ang)
            hip_angles.append(hip_ang)
            elbow_angles.append(elbow_ang)
            
            # For checking if hands are raised or not
            left_hand_raised = (l_elbow[1] < l_shoulder[1]) and (l_wrist[1] < l_elbow[1])
            right_hand_raised = (r_elbow[1] < r_shoulder[1]) and (r_wrist[1] < r_elbow[1])
            
            # Rule-Based classifier
            if knee_ang < 170 and hip_ang < 135:
                activity = "Squatting"
                joint_color = (0, 0, 255)
                
            elif left_hand_raised or right_hand_raised:
                activity = "Raising Arm"
                joint_color = (255, 0, 255)
                
            else:
                activity = "Standing"
                joint_color = (0, 255, 0)
                
            # For pointing and activate the skeleton joints
            for joint in [l_shoulder, l_elbow, l_wrist, l_hip, l_knee, l_ankle, r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle]:
                cv2.circle(frame, joint, 6, joint_color, -1)
                
            cv2.putText(frame, f"State: {activity}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, joint_color, 2)
            cv2.putText(frame, f"Knee: {knee_ang} | Hip: {hip_ang} | Elbow: {elbow_ang}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Per frame Accuracy
        ground_truth = get_ground_truth_activity(frame_idx)
        if results.pose_landmarks:
            total_evaluated_frames += 1
            if activity == ground_truth:
                correct_predictions += 1
                match_status = "MATCH"
                status_color = (0, 255, 0)
                
            else:
                match_status = "MISMATCH"
                status_color = (0, 0, 255)
                
            current_accuracy = (correct_predictions / total_evaluated_frames) * 100
            running_accuracies.append(current_accuracy)
            
            cv2.putText(frame, f"GT: {ground_truth} | Eval: {match_status}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        else:
            if len(frame_numbers) > len(running_accuracies):
                running_accuracies.append(running_accuracies[-1] if running_accuracies else 100.0)

        out.write(frame)
        cv2.imshow('CCP Assignment Final Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot angle fluctuation and accuracy charts
if len(frame_numbers) > 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot angle fluctuations over time
    ax1.plot(frame_numbers, knee_angles, label='Knee Angle', color='crimson')
    ax1.plot(frame_numbers, hip_angles, label='Hip Angle', color='teal')
    ax1.plot(frame_numbers, elbow_angles, label='Elbow Angle', color='royalblue', linestyle='--')
    ax1.set_title('Geometrical Joint Angles Fluctuation Chart')
    ax1.set_ylabel('Degrees')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy progression
    plot_acc = running_accuracies[:len(frame_numbers)]
    ax2.plot(frame_numbers[:len(plot_acc)], plot_acc, label='Pipeline Accuracy', color='darkorange', linewidth=2)
    ax2.set_title('Running Pipeline Accuracy Progression Curve')
    ax2.set_xlabel('Frame Count')
    ax2.set_ylabel('Accuracy %')
    ax2.set_ylim([0, 105])
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ccp_assignment_evaluation_charts.png', dpi=300)
    plt.show()
