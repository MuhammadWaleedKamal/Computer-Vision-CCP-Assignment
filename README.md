# Real-Time Human Pose Tracking and Activity Classifier

## 📌 Project Overview
This repository contains a real-time computer vision pipeline developed for a **Computer Vision Complex Computing Problem (CCP)** assignment. The system processes sequential video frames or live webcam feeds to detect human poses, calculate critical biomechanical joint angles, and employ a rule-based decision tree to classify three distinct human behaviors: **Standing**, **Squatting**, and **Raising Arm**. 

The system tracks algorithm performance by logging classification matches frame-by-frame against an expected ground-truth timeline and dynamically plotting running accuracy metrics.

---

## 🚀 Key Features

### 1. Pose Detection & Coordinate Smoothing (Task 1)
* Utilizes the modern **MediaPipe Tasks API** (`PoseLandmarker`) to extract 33 distinct skeletal keypoints.
* Applies a custom **Temporal Moving Average Filter** with a window size of $5$ frames to eliminate sensor jitter and smooth coordinates across time.
* Renders a real-time structural skeleton overlay with dynamic state color indications.

### 2. Kinematic Joint Angle Tracking (Task 2)
* Dynamically computes three critical biological joint angles over time using vector calculus: **Left Knee Angle**, **Left Hip Angle**, and **Left Elbow Angle**.
* Automatically exports a matplotlib evaluation visualization graphing angular fluctuations parallel to frame counts.

### 3. Rule-Based Classification Engine (Task 3)
* Uses deterministic, multi-variable heuristic decision boundaries to classify structural poses.
* Employs bilateral upper-limb spatial coordinate parsing to register arm-raising states across the vertical axis.
* Generates a live running accuracy progression curve tracking predictive compliance against manually defined ground truth models.

---

## 🛠️ System Architecture & Logic Design

### Angle Calculation Formula
Joint angles ($\theta$) at any central vertex joint $B$ flanked by keypoints $A$ and $C$ are computed using the two-argument arctangent function (`atan2`):

$$\theta = \left| \text{atan2}(C_y - B_y, C_x - B_x) - \text{atan2}(A_y - B_y, A_x - B_x) \right| \times \frac{180}{\pi}$$

### Rule Threshold Decisions
* **Squatting:** Triggered if lower-body extension compresses simultaneously: $\theta_{\text{knee}} < 130^\circ \text{ AND } \theta_{\text{hip}} < 105^\circ$.
* **Raising Arm:** Triggered if a sequential vertical spatial stack is verified on either limb (where the wrist rises above the elbow, and the elbow sits above the shoulder line):
  $$\text{Arm Raised} = (Y_{\text{elbow}} < Y_{\text{shoulder}}) \land (Y_{\text{wrist}} < Y_{\text{elbow}})$$
  *(Note: In OpenCV, smaller pixel Y-values denote higher physical elevation on screen).*
* **Standing:** Neutral fallback state activated whenever tracking kinematics fail to meet squatting or arm-raising conditions.

---

## 📦 Installation & Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed along with the necessary library dependencies:
```bash
pip install opencv-python mediapipe numpy matplotlib

### Model Asset Setup
1. Download the pre-trained model file: `pose_landmarker_full.task` from MediaPipe's official model distribution.
2. Place the `.task` file inside your project directory and update the `MODEL_PATH` variable in the script.

### Running the Project
Execute the pipeline via your terminal:
```bash
python ccp_final_save.py

Press 'q' on the active video display window to stop processing, finish saving the recorded video, and render the evaluation charts.

### 📊 Empirical Evaluation Charts
The execution outputs two core quantitative tracking subplots automatically saved to disk as ccp_assignment_evaluation_charts.png:

Geometrical Joint Angles Fluctuation Chart: Tracks the exact degrees of the knee, hip, and elbow variations frame-by-frame to isolate transition milestones.

Running Pipeline Accuracy Progression Curve: Visualizes real-time performance, showing brief dipping intervals at movement transition thresholds before stabilizing over sustained actions.

🎥 Project Demo (How to View Video)
To show your final processed video execution directly on your GitHub repository page, you can embed a looping demo snippet inside this README. Follow these instructions to upload it:

Option A: Embed an Animated GIF (Recommended for Instant Autoplay)
Convert your saved output_processed.mp4 file into an .gif file using an online converter or tool like ScreenToGif.

Upload the .gif file to your repository (e.g., inside a folder named assets).

View it immediately by displaying it below:

Option B: Upload the Raw Video File
Simply drag and drop your final output_processed.mp4 file directly into your GitHub repository file space via the web browser interface.

Link it in your README so reviewers can click to view it natively:

👉 Click here to watch the full Processed Execution Demo Video (output_processed.mp4)
