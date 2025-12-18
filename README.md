# Human Motion Mimicking for GR-2 Humanoid Robot

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E.svg)](https://docs.ros.org/en/humble/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-blue)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

## Overview

This project implements a marker-less teleoperation framework for the **Fourier Intelligence GRX (GR-2)** humanoid robot. Using a single RGB camera, the system captures human motion in real-time and retargets it to the robot's kinematic chain.

The architecture bypasses the need for expensive motion capture suits by leveraging **YOLOv11-Nano** for pose estimation and **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) for automated user-to-robot calibration.

## Key Features

* **Dual-Stream Perception:** Utilizes two parallel YOLOv11 instances—one for gross body mechanics (COCO-Pose) and a custom-trained model for fine hand dexterity (21 keypoints).
* **Hybrid Motion Retargeting:**
    * **Arms:** Geometric angle extraction between skeletal vectors.
    * **Hands:** "Distance Ratio" control (Tip-to-Wrist normalized by palm size) to mitigate keypoint jitter.
* **Auto-Calibration (CMA-ES):** A stochastic optimization routine that automatically tunes mapping parameters (min/max ranges) to fit the specific biomechanics of the user.
* **Safety & Smoothing:** Implements Exponential Moving Average (EMA) filtering and URDF-based clamping to prevent hardware damage.

## System Architecture

The pipeline consists of three functional layers:

1.  **Perception Layer:** Captures video (MJPG, 960x540) and extracts 17 body keypoints and 21 hand keypoints.
2.  **Retargeting Layer:** Converts 2D pixel coordinates into 3D joint commands using geometric mimicry.
3.  **Control Layer:** Publishes `JointTrajectory` messages via ROS 2 middleware to the robot's low-level controllers.

<img width="1760" height="696" alt="Architecture" src="https://github.com/user-attachments/assets/4cbd90c6-c560-459a-9096-70452a594c56" />


## Technical Approach

### 1. Hybrid Mapping Strategy
To handle the difference in scale and signal noise, different methods are used for different limbs:

| Limb | Method | Description |
| :--- | :--- | :--- |
| **Shoulder/Elbow** | **Geometric Angles** | Calculates Euclidean angles between vectors (e.g., Shoulder-Elbow vs. Vertical Axis). |
| **Fingers** | **Distance Ratio** | Calculates $R = \frac{||p_{tip} - p_{wrist}||}{L_{palm}}$. This ratio provides robust actuation 0.0 (closed) to 1.0 (open) despite 2D depth ambiguity. |

### 2. CMA-ES Calibration
Human limb proportions vary, making static mapping inefficient. We use **CMA-ES** to optimize the linear mapping function:
$$\theta_{out} = \theta_{min}^{robot} + \left( \frac{v_{in} - v_{min}^{user}}{v_{max}^{user} - v_{min}^{user}} \right) \cdot (\theta_{max}^{robot} - \theta_{min}^{robot})$$
The algorithm minimizes a cost function that penalizes "clipping" (exceeding robot limits) and rewards "range utilization" (ensuring the user can reach the robot's full extension).

### 3. Signal Smoothing
Raw vision data is noisy. An EMA filter is applied with tuned alpha values:
* $\alpha = 0.6$ for Arms (Prioritizing stability).
* $\alpha = 0.7$ for Hands (Prioritizing responsiveness).

## Dependencies

* **ROS 2 Humble** (Desktop Full)
* `ultralytics` (YOLOv11)
* `cma` (Optimization library)
* `roboticstoolbox-python` (Kinematics helper)
* `opencv-python`
* `torch` (CUDA recommended)

## 🔧 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/gr2-mimic.git](https://github.com/yourusername/gr2-mimic.git)
    cd gr2-mimic
    ```

2.  **Install Python requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the ROS 2 workspace:**
    ```bash
    colcon build --symlink-install
    source install/setup.bash
    ```

4.  **Run the Calibration (First time users):**
    ```bash
    ros2 run gr2_mimic calibrate_user --record
    ```

5.  **Start Teleoperation:**
    ```bash
    ros2 run gr2_mimic mimic_node
    ```

## 📊 Results

<img width="900"  alt="image" src="https://github.com/user-attachments/assets/866908a5-bc8c-4da6-81f4-024dbedda45d" />

<img width="900"  alt="image" src="https://github.com/user-attachments/assets/c3b3dc9e-4f01-4b53-9107-d86d75af026f" />


* **Latency:** The system operates with a trajectory buffer of ~120ms to ensure smooth interpolation.
* **Accuracy:** Inverse correlations observed in Shoulder Roll (-0.69) and Index Finger (-0.19) confirm effective mapping from human input to robot actuation.

##  References

* **Corke, P., & Haviland, J. (2021).** Not your grandmother’s toolbox—The Robotics Toolbox for Python. *IEEE Robotics & Automation Magazine*.
* **Seekircher, A., et al. (2013).** Motion capture and contemporary optimization algorithms for robust and stable motions on simulated biped robots. *RoboCup 2012*.
* **Fourier Intelligence.** (2024). GR-2 Humanoid Robot Technical Specifications.




