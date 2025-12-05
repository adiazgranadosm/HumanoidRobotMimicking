# Human Motion Mimicking in Humanoid GRX Robot

HumanoidRobotMimickingMotion

The goal of the project is to use a computer vision model to control humanoid robotic arms and hands by mimicking human movements in real time.
The solution integrates computer vision models (YOLOv11) with a ROS 2–based control architecture.

Body Model: 

YOLOv11-Nano Pose (yolo11n-pose) architecture: small and fast version optimized for real-time applications
COCO-Pose pretrained model, dataset supports 17 keypoints

Hands Model

YOLOv11-Nano Pose (yolo11n-pose)
Train the weights pre-trained on large human-pose model using dataset hand-keypoints of 26,768 images of hands (640 x 640) 


Pipeline 

1. Callback: 30 FPS 
Inference: Runs both YOLO models.
Extraction: Calculates angles/ratios for body and hands.
Mapping and Optimization: Applies the Optimized Calibration map and convert inputs to robot joints.
Smoothing: Filters the result.
Publishing: Sends a JointTrajectory message to the ROS 2 controller.

2. Mapping:
Mapping using Geometric Mimicry with angle calculation.
Angle Extraction
Linear Mapping 
Finger Control: Instead of angles (noisy for small fingers), it calculates the Ratio of Distance (Tip-to-Wrist divided by Palm-Size)

3. Optimization: 
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) stochastic optimization algorithm.
Calibration using CMA-ES: Every human moves differently. This optimization is using to auto-tune itself to different users avoiding poor robot movement.
The system reads a recorded CSV file of the user performing movements.
Calculate a Cost Function which searches for mapping parameters (min, max) that maximize the usage of the robot's range
A custom profile of the motion is created for a specific user.

4. Safety & Smoothing:
Smoothing (EMA): Uses an Exponential Moving Average to filter out high-frequency noise. Lower alpha values result in smoother but slower movement.
Clamping (clamp): Validation to sent to the robot values that exceeds the physical limits defined in the URDF, preventing hardware damage.

Results

The robot is tracking the human motion (the shoulder axis is inverted).
The robot has restrictions in its range of motion due to URDF limits.
Smoothing is moving the robot in a stable motion





