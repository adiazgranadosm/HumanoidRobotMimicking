import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cma
import time
import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ultralytics import YOLO
import roboticstoolbox as rtb
import spatialmath as sm
import csv
import datetime
import os
import pandas as pd



# =============================================================================
# Lists
# =============================================================================
# Body keypoints (COCO format)
BODY_KPTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

# 21-point hand model 
HAND_IDXS = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}


# Robot joints 
JOINTS: Dict[str, List[str] | str] = {
    # Arms
    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow": "left_elbow_pitch_joint",
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_pitch_joint",
    # Wrists
    "left_wrist_yaw": "left_wrist_yaw_joint",
    "left_wrist_pitch": "left_wrist_pitch_joint",
    "left_wrist_roll": "left_wrist_roll_joint",
    "right_wrist_yaw": "right_wrist_yaw_joint",
    "right_wrist_pitch": "right_wrist_pitch_joint",
    "right_wrist_roll": "right_wrist_roll_joint",
    # Fingers
    "L_thumb": ["L_thumb_proximal_pitch_joint", "L_thumb_distal_joint"],
    "L_index": ["L_index_proximal_joint", "L_index_intermediate_joint"],
    "L_middle": ["L_middle_proximal_joint", "L_middle_intermediate_joint"],
    "L_ring": ["L_ring_proximal_joint", "L_ring_intermediate_joint"],
    "L_pinky": ["L_pinky_proximal_joint", "L_pinky_intermediate_joint"],
    "R_thumb": ["R_thumb_proximal_pitch_joint", "R_thumb_distal_joint"],
    "R_index": ["R_index_proximal_joint", "R_index_intermediate_joint"],
    "R_middle": ["R_middle_proximal_joint", "R_middle_intermediate_joint"],
    "R_ring": ["R_ring_proximal_joint", "R_ring_intermediate_joint"],
    "R_pinky": ["R_pinky_proximal_joint", "R_pinky_intermediate_joint"],
}

# Robot limits (taken from the URDF)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "left_shoulder_pitch_joint": (-2.96, 1.92),
    "left_shoulder_roll_joint": (-0.52, 2.79),
    "left_shoulder_yaw_joint": (-1.83, 1.83),
    "left_elbow_pitch_joint": (-1.52, 0.48),
    "right_shoulder_pitch_joint": (-2.96, 1.92),
    "right_shoulder_roll_joint": (-2.79, 0.52),  # mirrored
    "right_shoulder_yaw_joint": (-1.83, 1.83),
    "right_elbow_pitch_joint": (-1.52, 0.48),
    "left_wrist_yaw_joint": (-1.83, 1.83),
    "left_wrist_pitch_joint": (-0.61, 0.61),
    "left_wrist_roll_joint": (-0.95, 0.95),
    "right_wrist_yaw_joint": (-1.83, 1.83),
    "right_wrist_pitch_joint": (-0.61, 0.61),
    "right_wrist_roll_joint": (-0.95, 0.95),
    "L_thumb_proximal_yaw_joint": (-1.74, 0.0),
    "L_thumb_proximal_pitch_joint": (0.0, 1.22),
    "L_thumb_distal_joint": (0.0, 1.23),
    "L_index_proximal_joint": (-1.57, 0.0),
    "L_index_intermediate_joint": (-1.74, 0.0),
    "L_middle_proximal_joint": (-1.57, 0.0),
    "L_middle_intermediate_joint": (-1.74, 0.0),
    "L_ring_proximal_joint": (-1.57, 0.0),
    "L_ring_intermediate_joint": (-1.74, 0.0),
    "L_pinky_proximal_joint": (-1.57, 0.0),
    "L_pinky_intermediate_joint": (-1.74, 0.0),
    "R_thumb_proximal_yaw_joint": (-1.74, 0.0),
    "R_thumb_proximal_pitch_joint": (0.0, 1.22),
    "R_thumb_distal_joint": (0.0, 1.23),
    "R_index_proximal_joint": (-1.57, 0.0),
    "R_index_intermediate_joint": (-1.74, 0.0),
    "R_middle_proximal_joint": (-1.57, 0.0),
    "R_middle_intermediate_joint": (-1.74, 0.0),
    "R_ring_proximal_joint": (-1.57, 0.0),
    "R_ring_intermediate_joint": (-1.74, 0.0),
    "R_pinky_proximal_joint": (-1.57, 0.0),
    "R_pinky_intermediate_joint": (-1.74, 0.0),
}

class GR2Mimic(Node):
    def __init__(self) -> None:
        super().__init__("gr2_mimic")
        self.get_logger().info("Starting GR2 Mimic")

        self.publisher = self.create_publisher(
            JointTrajectory, "/forward_position_controller/joint_trajectory", 10
        )        

        # setup models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_body = YOLO("yolo11n-pose.pt")
        self.model_body.to(self.device)
        self.get_logger().info(f"Using body model: yolo11n-pose.pt on {self.device}")

        here = Path(os.path.abspath(__file__)).parent
        hand_model = here / "Outputs" / "hand_keypoint_model.pt"
        hand_name = str(hand_model if hand_model.exists() else "yolo11n-pose.pt")
        self.model_hand = YOLO(hand_name)
        self.model_hand.to(self.device)
        self.get_logger().info(f"Using hand model: {hand_name} on {self.device}")

        # Setup Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        self.window_name = "GR2 Mimic"
        #cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
       
        # setup joints and pose
        self.joint_names = self.expand_joint_names()
        self.reset_pose()  
        
        # setup other parameters
        self.alpha_body = 0.6  # smoothing
        self.alpha_hand = 0.7  # smoothing
        self.confidence_body = 0.5
        self.confidence_hand = 0.4
        self.init_arm_len_left = None
        self.init_arm_len_right = None
        self.prev_time = 0
        self.fps_avg = 0
        self.calibration_file = here / "calibration_data.csv"   
        self.apply_optimization = True

        self.init_data_recording()

        urdf_file = here.parent  / "urdf" / "GR2v2.1.1_fourier_hand_6dof.urdf"

        # optional setup to work with ik
        self.load_robot_ik(urdf_file)

        # optional setup to work CMA-ES optimization
        self.run_calibration()

        self.timer = self.create_timer(1.0 / 15.0, self.timer_callback)    
            
    # ----------------------------------------------------------------------
    # Function for init
    # ----------------------------------------------------------------------
    def expand_joint_names(self) -> List[str]:
        names: List[str] = []
        for val in JOINTS.values():
            if isinstance(val, list):
                names.extend(val)
            else:
                names.append(val)
        return names
    
    def init_data_recording(self):
        self.frame_count = 0
        self.data_log = [] 
        
        self.joints_to_record = [            
            'right_shoulder_roll', 
            'right_elbow', 
            'R_thumb',
            'R_index',
        ]
    # Reset joints the robot pose to initial state
    def reset_pose(self):
        self.get_logger().info("Resetting Robot Pose...")
        # Reset internal state variables to 0.0
        self.current_pos = {}
        for val in JOINTS.values():
            if isinstance(val, list):
                for name in val: self.current_pos[name] = 0.0
            else:
                self.current_pos[val] = 0.0

    # Loads the GR2 robot from URDF
    def load_robot_ik(self, urdf_path: Path):       
        try:
            # ERobot loads a robot defined by a URDF file
            self.robot_ik = rtb.ERobot.URDF(urdf_path)
        except Exception as e:
            self.get_logger().info(f"Error loading URDF from {urdf_path}: {e}")
            return None
        
    # Optimizes src_min and src_max for a specific joint.
    def optimize_joint(self, joint_name, human_data_array, robot_limits):
        
        dst_min, dst_max = robot_limits
        
        def fitness(params):
            src_min, src_max = params           
            
            if abs(src_max - src_min) < 0.01: return 100.0            
       
            t = (human_data_array - src_min) / (src_max - src_min)
            
            # Calculate Clipping            
            clipped_mask = (t < 0) | (t > 1)
            clipping_penalty = np.sum(clipped_mask) / len(t) # Percent of frames clipped
            
            # Calculate Range Usage           
            spread = np.std(t)
            spread_score = (0.3 - spread)**2 
            
            # Total Cost: Minimize Clipping + Ensure Spread
            return (clipping_penalty * 10.0) + spread_score
        
        # Run CMA
        # Guess based on min/max of data
        start_guess = [np.min(human_data_array), np.max(human_data_array)]
        es = cma.CMAEvolutionStrategy(start_guess, 0.2, {'verbose': -1})
        es.optimize(fitness, iterations=50)
        
        return es.result.xbest
        
    
    # setup CMA - ES optimization for all joints in init process to reduce steps during the callback
    def run_calibration(self):
        try:
            df = pd.read_csv(self.calibration_file)
        except FileNotFoundError:
            self.get_logger().info("CSV file not found.")
            return
        
        optimized_map = {}

        # Left Elbow
        if 'left_arm_extension_raw' in df.columns:           
            src = self.optimize_joint("left_elbow_pitch_joint", df['left_arm_extension_raw'], JOINT_LIMITS['left_elbow_pitch_joint'])
            optimized_map['left_elbow_pitch_joint'] = src
        elif 'left_elbow_pitch_joint' in df.columns:
             src = self.optimize_joint("left_elbow_pitch_joint", df['left_elbow_pitch_joint'], JOINT_LIMITS['left_elbow_pitch_joint'])
             optimized_map['left_elbow_pitch_joint'] = src

        # Right Elbow
        if 'right_arm_extension_raw' in df.columns:           
            src = self.optimize_joint("right_elbow_pitch_joint", df['right_arm_extension_raw'], JOINT_LIMITS['right_elbow_pitch_joint'])
            optimized_map['right_elbow_pitch_joint'] = src
        elif 'right_elbow_pitch_joint' in df.columns:
             src = self.optimize_joint("right_elbow_pitch_joint", df['right_elbow_pitch_joint'], JOINT_LIMITS['right_elbow_pitch_joint'])
             optimized_map['right_elbow_pitch_joint'] = src

        # Loop through all other keys in JOINT_LIMITS
        for key, value in JOINTS.items():             

            # Check if this joint name exists as a column in the CSV
            if key if isinstance(value, list) else value in df.columns: 
                item = key if isinstance(value, list) else value    
                if isinstance(value, list): 
                    limit = value[0]  # Just use first joint for limits
                else:
                    limit = value

                print(f"Optimizing {item}...")
                print(limit)                   
                # Check variance to ensure have data 
                if df[item].std() < 0.01:                   
                    optimized_map[item] = (0.0, 1.0)
                else:
                    best_src = self.optimize_joint(item, df[item], JOINT_LIMITS.get(limit))
                    optimized_map[item] = best_src

     
        self.get_logger().info("CMA-ES Optimization Loaded")   
        self.get_logger().info("JOINT_OPTIMIZED:")
        
        for k, v in optimized_map.items():
            self.get_logger().info(f'    "{k}": ({v[0]:.4f}, {v[1]:.4f}),')            

        self.optimize_joints = optimized_map
       
    # ----------------------------------------------------------------------
    # Callback
    # ----------------------------------------------------------------------   
    def timer_callback(self) -> None:

        start_time = time.time()        
        
        fingers = {'thumb','index','middle', 'ring', 'pinky'}
        
        ret, frame = self.cap.read()
        if not ret:
            return

        height, width, _ = frame.shape
        targets: Dict[str, float] = {}

        # Body pose  
        body_targets, human_targets_body = self.extract_body_targets(frame, width, height)        
        targets.update(body_targets)
    
        # Hands
        hand_targets, human_targets_hand = self.extract_hand_targets(frame, width, height)  
        targets.update(hand_targets)

        # Smooth
        for name in self.joint_names:
            if name not in targets:
                continue
            desired = targets.get(name, self.current_pos[name])    
            if name.split('_')[0] in ['L', 'R'] and any(finger in name for finger in fingers):                
                smoothed = (1.0 - self.alpha_hand) * self.current_pos[name] + self.alpha_hand * desired          
               
            else: 
                smoothed = (1.0 - self.alpha_body) * self.current_pos[name] + self.alpha_body * desired
            self.current_pos[name] = self.clamp(smoothed, JOINT_LIMITS.get(name))
            #if self.current_pos[name] != 0.0:
            #    self.get_logger().info(f"Clamp value {name} value {self.current_pos[name]}")

        # Record data for plots
        row = {'frame': self.frame_count}
        
        for name in self.joints_to_record:         
            urdf_key = JOINTS.get(name)   
            if isinstance(urdf_key, list):
                actual_val = self.current_pos.get(urdf_key[0], 0.0)
                human_val = human_targets_hand.get(urdf_key[0], 0.0)
            else:
                actual_val = self.current_pos.get(urdf_key, 0.0)
                human_val = human_targets_body.get(urdf_key, 0.0)
            row[f'{name}_robot'] = actual_val
            row[f'{name}_human'] = human_val
            
            
        self.data_log.append(row)

        # Publish
        self.publish()

        self.frame_count += 1
        end_time = time.time()
        process_time = end_time - start_time

        if process_time > 0:
            current_fps = 1.0 / process_time 
            self.fps_avg = (self.fps_avg * 0.9) + (current_fps * 0.1)           

        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Loop Rate: {self.fps_avg:.1f} FPS")
            
        cv2.putText(frame, f"FPS: {int(self.fps_avg)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),1)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)     

    # ----------------------------------------------------------------------
    # Common functions
    # ----------------------------------------------------------------------

    # Gets the 2D point for a given body keypoint name.
    def get_points(self, name, keypoints):
            idx = BODY_KPTS[name]
            if keypoints[idx][2] < 0.5:
                return None
            return keypoints[idx][:2]
    
    # Calculates the angle at a joint if the confidence is valid.
    def angle_if_valid(self, i0: int, i1: int, i2: int, conf, kpts) -> float | None:
            if conf[i0] < self.confidence_body or conf[i1] < self.confidence_body or conf[i2] < self.confidence_body:
                return None
            return self.angle_at_joint(kpts[i0][:2], kpts[i1][:2], kpts[i2][:2])
    
    # Calculates the angle (in degrees) at point 'b' formed by points a-b-c.
    def angle_at_joint(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        return self.angle_between(ba, bc)
    
    # Draws a line segment between two points on the frame.
    def draw_segment(self, frame: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: Tuple[int, int, int]) -> None:
        cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, 2)
        cv2.circle(frame, (int(p0[0]), int(p0[1])), 4, color, -1)
        cv2.circle(frame, (int(p1[0]), int(p1[1])), 4, color, -1)

    # Clamps a value to the given bounds.
    def clamp(self, val: float, bounds: Tuple[float, float] | None) -> float:
        if not bounds:
            return val
        lo, hi = bounds
        if val < lo or val > hi:
            print(f"Clamping value {val} to range ({lo}, {hi})")       
        return max(lo, min(hi, val))
    
    # Calculates the angle (in radians) at point 'b' formed by points a-b-c.
    def calculate_human_angle(self, a, b, c):
        if a is None or b is None or c is None: return 0.0        
       
        ba = a - b
        bc = c - b        
      
        angle_ba = np.arctan2(ba[1], ba[0])
        angle_bc = np.arctan2(bc[1], bc[0])   
        angle = np.abs(angle_ba - angle_bc)       
      
        if angle > np.pi:
            angle = 2*np.pi - angle
            
        return angle   

    # Maps a value from one range to another with optional inversion.
    def map_range_adj(self, val: float, src: Tuple[float, float], dst: Tuple[float, float], invert: bool = False) -> float:
        s0, s1 = src
        d0, d1 = dst
            
        # Normalize to 0.0 - 1.0
        t = (val - s0) / (s1 - s0 + 1e-6)
        t = max(0.0, min(1.0, t))
        
        if invert:
            t = 1.0 - t
            
        return d0 + t * (d1 - d0) 
    
    # Calculates the angle (in degrees) between two vectors.
    def angle_between(self, a: np.ndarray, b: np.ndarray) -> float :
        dot = float(np.dot(a, b))
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return math.degrees(math.acos(np.clip(dot / denom, -1.0, 1.0)))
    
    # Maps a value from one range to another for IK use (optional).
    def map_value_ik(self, value, in_min, in_max, out_min, out_max):
        """Maps a value from one range to another (like Arduino map)."""
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    
    # ----------------------------------------------------------------------
    # Body mapping
    # ----------------------------------------------------------------------
    def extract_body_targets(self, frame: np.ndarray, w: int, h: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out_human: Dict[str, float] = {}
        results = self.model_body(frame, verbose=False, stream=True)

        if not results:
            return out, out_human

        keypoints = None
        for res in results:
            if res.keypoints is not None and res.keypoints.data.shape[0] > 0:
                keypoints = res.keypoints.data[0].cpu().numpy()
                break


        if keypoints is None or keypoints.shape[0] < max(BODY_KPTS.values()) + 1:
            return out, out_human
        
        conf = keypoints[:, 2]
       
        for side in ("left", "right"):
            shoulder = self.get_points(f"{side}_shoulder", keypoints)
            elbow = self.get_points(f"{side}_elbow", keypoints)
            wrist = self.get_points(f"{side}_wrist", keypoints)
            if side == "left":
                self.left_wrist = wrist
            else:
                self.right_wrist = wrist
            hip = self.get_points(f"{side}_hip", keypoints)
            
            human_elbow_rad = self.calculate_human_angle(shoulder, elbow, wrist)            
            right_shoulder_roll = self.calculate_human_angle(hip, shoulder, elbow)      
           
            if shoulder is None or elbow is None or wrist is None or hip is None:
                continue

            if conf[BODY_KPTS[f"{side}_shoulder"]] < self.confidence_body or \
               conf[BODY_KPTS[f"{side}_elbow"]] < self.confidence_body or \
               conf[BODY_KPTS[f"{side}_wrist"]] < self.confidence_body:
                continue

            arm_len = np.linalg.norm(shoulder - elbow) + 1e-8 
            base = hip if hip is not None else shoulder + np.array([0, 100])                             

            if (self.init_arm_len_left is None or self.init_arm_len_left < arm_len) and side == "left":
                self.init_arm_len_left = arm_len
                self.get_logger().info(f"Initial left arm length: {self.init_arm_len_left:.2f}px")
            if (self.init_arm_len_right is None or self.init_arm_len_right < arm_len) and side == "right":
                self.init_arm_len_right = arm_len
                self.get_logger().info(f"Initial right arm length: {self.init_arm_len_right:.2f}px")           

            # Shoulder roll            
            move_roll = False 
            elbow_hip_offset = abs(base[0] - elbow[0])              
            roll_deg = self.angle_between(base - shoulder, elbow - shoulder)    
            #roll_deg = self.angle_at_joint(hip, shoulder, elbow)         
            roll_sign = 1.0 if side == "left" else -1.0
            if self.apply_optimization:
                best_src = self.optimize_joints.get(f"{side}_shoulder_roll_joint", (0.0, 1.0))
                roll = self.map_range_adj(roll_deg, (best_src[0], best_src[1]), (-0.1, 2.5 * roll_sign))
            else:
                roll = self.map_range_adj(roll_deg, (20, 100), (-0.1, 2.5 * roll_sign))
            roll_joint = JOINTS[f"{side}_shoulder_roll"]
            if elbow_hip_offset > 50:    
                out[roll_joint] = roll

             # Shoulder pitch         
            pitch_offset = (shoulder[1] - elbow[1]) / arm_len                          
            pitch_joint = JOINTS[f"{side}_shoulder_pitch"]   
            if self.apply_optimization: 
                best_src = self.optimize_joints.get(f"{side}_shoulder_pitch_joint", (0.0, 1.0))            
                pitch = self.map_range_adj(pitch_offset, (best_src[0], best_src[1]), (0.0, -1.2), invert=False)    
            else:
                pitch = self.map_range_adj(pitch_offset, (-1, 1), (0.0, -1.2), invert=False) 
            #if elbow_hip_offset < 50 and roll_deg < 20: 
            out[pitch_joint] = pitch     

            # Shoulder yaw             
            yaw = 0.0
            yaw_offset = (elbow[0] - shoulder[0]) / arm_len
            elbow__offset = abs(base[0] - wrist[0])   
            if side == "left":   
                if self.apply_optimization: 
                    best_src = self.optimize_joints.get(f"{side}_shoulder_yaw_joint", (0.0, 1.0))            
                    yaw = self.map_range_adj(yaw_offset, (best_src[0], best_src[1]), (-0.3, 1.2), invert=False)
                else: 
                    yaw = self.map_range_adj(yaw_offset, (0.1, 0.6), (-0.3, 1.2), invert=False)
            else:     
                if self.apply_optimization: 
                    best_src = self.optimize_joints.get(f"{side}_shoulder_yaw_joint", (0.0, 1.0))            
                    yaw = self.map_range_adj(yaw_offset, (best_src[0], best_src[1]), (-1.2, 0.3), invert=False)
                else: 
                    yaw = self.map_range_adj(yaw_offset, (-0.6, -0.1), (-1.2, 0.3), invert=False)
            yaw_joint = JOINTS[f"{side}_shoulder_yaw"]
            out[yaw_joint] = yaw

            # Elbow flex    
            elbow_deg = self.angle_at_joint(shoulder, elbow, wrist)            
            elbow_joint = JOINTS[f"{side}_elbow"]
            if self.apply_optimization: 
                best_src = self.optimize_joints.get(f"{side}_elbow_pitch_joint", (0.0, 1.0))
                elbow_rad = self.map_range_adj(elbow_deg, (best_src[1], best_src[0]), (0.4, -1.5), invert=False)          
            else:   
                elbow_rad = self.map_range_adj(elbow_deg, (170, 10), (0.4, -1.5), invert=False)
                       
            out[elbow_joint] = elbow_rad          

            # Save data for plotting            
            if side == "right":                
                out_human[elbow_joint] = human_elbow_rad              
                out_human[roll_joint] = right_shoulder_roll

            # Print measures in screen
            self.draw_segment(frame, shoulder, elbow, (0, 255, 255) if side == "left" else (255, 0, 255))
            self.draw_segment(frame, elbow, wrist, (0, 255, 255) if side == "left" else (255, 0, 255))
            cv2.circle(frame, (int(base[0]), int(base[1])), 4, (255, 0, 0), -1)           

            dbg_color = (0, 100, 200)
            cv2.putText(
                frame,
                f"{side[0].upper()} sh r{roll:.2f} p{pitch:.2f} y{yaw:.2f} el{elbow_rad:.2f}",
                (int(shoulder[0]) - 50, int(shoulder[1]) - (20 if side == "left" else 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                dbg_color,
                2,
            )
            cv2.putText(
                frame,
                f"{side[0].upper()} r_deg{roll_deg:.2f} p_off{pitch_offset:.2f} y_off{yaw_offset:.2f} el_deg{elbow_deg:.2f}",
                (int(wrist[0]) - 80, int(wrist[1]) + (20 if side == "left" else 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                dbg_color,
                2,
            )                 

            self.get_logger().info(f"Shoulder {side} elbow hip {elbow_hip_offset}")
        return out, out_human    

    # ----------------------------------------------------------------------
    # Hand mapping
    # ----------------------------------------------------------------------
    def extract_hand_targets(self, frame: np.ndarray, frame_width: int, frame_height: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out_human: Dict[str, float] = {}
        results = self.model_hand(frame, verbose=False, stream=False)
        hands: List[np.ndarray] = []
        for res in results:
            if res.keypoints is None:
                continue
            for kpts in res.keypoints.data:
                k = kpts.cpu().numpy()
                if k.shape[0] >= 21:
                    hands.append(k)

        if not hands:
            return out, out_human

        for idx, kpts in enumerate(hands):
            thumb_x = kpts[HAND_IDXS["thumb_tip"]][0]
            pinky_x = kpts[HAND_IDXS["pinky_tip"]][0]  
            wrist = kpts[HAND_IDXS["wrist"]][:2]
            wrist_conf = kpts[HAND_IDXS["wrist"]][2]
            wrist_left = self.left_wrist
            wrist_right = self.right_wrist

            # Identify side of the hand (left or right)
            dist_left, dist_right = float('inf'), float('inf')

            if wrist_conf > self.confidence_body and wrist_left is not None:                 
                dist_left = np.linalg.norm(wrist - wrist_left[:2])
            if wrist_conf > self.confidence_body and wrist_right is not None:
                dist_right = np.linalg.norm(wrist - wrist_right[:2])

            if dist_left != float('inf') and dist_right != float('inf'): 
                side = "left" if (dist_left < dist_right) else "right"
            elif dist_left != float('inf'):
                side = "left"
            elif dist_right != float('inf'):
                side = "right"
            else:
                if thumb_x < pinky_x: side = "left"
                else:side = "right"
           
            out, out_human = self.hand_to_targets(kpts, frame, side, frame_width, frame_height)
            
        return out, out_human

    def hand_to_targets(self, kpts: np.ndarray, frame: np.ndarray, side: str, frame_width: int, frame_height: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out_human: Dict[str, float] = {}
        conf = kpts[:, 2]
        color = (0, 255, 255) if side == "left" else (255, 0, 255)
       
        wrist = kpts[HAND_IDXS["wrist"]][:2]
        middle_mcp = kpts[HAND_IDXS["middle_mcp"]][:2]
        index_mcp = kpts[HAND_IDXS["index_mcp"]][:2]
        ring_mcp = kpts[HAND_IDXS["ring_mcp"]][:2]
        pinky_mcp = kpts[HAND_IDXS["pinky_mcp"]][:2]

        palm_size = np.linalg.norm(wrist - middle_mcp) + 1e-8
        sign = 1.0 if side == "left" else -1.0

        if conf[HAND_IDXS["wrist"]] < self.confidence_hand:
            return out, out_human

        # Wrist roll (optimized)                   
        vec = middle_mcp - wrist
        roll_angle = math.atan2(vec[1], vec[0])
        roll_joint = JOINTS[f"{side}_wrist_roll"]
        if self.apply_optimization:
            best_src = self.optimize_joints.get(f"{side}_wrist_roll_joint", (0.0, 1.0))     
            roll = self.map_range_adj(roll_angle, (best_src[0], best_src[1]), (0.8 * sign, -0.8 * sign), invert=False)
        else: 
            roll = self.map_range_adj(roll_angle, (-2.5, 2.5), (0.8 * sign, -0.8 * sign), invert=False)
        #out[roll_joint] = roll

        # Wrist pitch (optimizeed)
        best_src = self.optimize_joints.get(f"{side}_wrist_pitch_joint", (0.0, 1.0)) 
        rel_y = (wrist[1] - middle_mcp[1]) / palm_size
        pitch_joint = JOINTS[f"{side}_wrist_pitch"]
        if self.apply_optimization:
            best_src = self.optimize_joints.get(f"{side}_wrist_pitch_joint", (0.0, 1.0)) 
            pitch = self.map_range_adj(rel_y, (best_src[0], best_src[1]), (0.0, 0.6))
        else:
            pitch = self.map_range_adj(rel_y, (-2.0, -0.8), (0.0, 0.6))
        out[pitch_joint] = pitch

        # Wrist yaw (optimized)
        spread_angle = math.atan2(pinky_mcp[1] - index_mcp[1], pinky_mcp[0] - index_mcp[0])
        yaw_joint = JOINTS[f"{side}_wrist_yaw"]
        if self.apply_optimization:
            best_src = self.optimize_joints.get(f"{side}_wrist_yaw_joint", (0.0, 1.0)) 
            yaw = self.map_range_adj(spread_angle, (best_src[0], best_src[1]), (-1.4 * sign, 1.4 * sign))
        else:
            yaw = self.map_range_adj(spread_angle, (-1.5, 1.5), (-1.4 * sign, 1.4 * sign))
        #out[yaw_joint] = yaw

        prefix = "L_" if side == "left" else "R_"

        # Thumb 
        thumb_tip = kpts[HAND_IDXS["thumb_tip"]]
        pinky_mcp = kpts[HAND_IDXS["pinky_mcp"]]

        if conf[HAND_IDXS["thumb_tip"]] < self.confidence_hand:
            return out, out_human

        # Distance from Thumb Tip to Pinky  
        thumb_dist = np.linalg.norm(thumb_tip - pinky_mcp)
        thumb_ratio = thumb_dist / palm_size

        # Map
        if self.apply_optimization:
            best_src = self.optimize_joints.get(prefix + "thumb", (0.0, 1.0)) 
            thumb_val = self.map_range_adj(thumb_ratio, (best_src[1], best_src[0]), (1.0, 0.0))
        else:
            thumb_val = self.map_range_adj(thumb_ratio, (0.5, 1.5), (1.0, 0.0))
        
        # Assign to Joints
        key = prefix + "thumb"
        if key in JOINTS:
            joints = JOINTS[key]
            for j_name in joints:              
                out[j_name] = thumb_val
                out_human[j_name] = thumb_ratio
               
        finger_sets = {
            "index": ("index_mcp", "index_pip", "index_dip", "index_tip"),
            "middle": ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
            "ring": ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
            "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
        }

        # Fingers except thumb 
        for fname, (mcp_key, pip_key, dip_key, tip_key) in finger_sets.items():
            
            tip_idx = HAND_IDXS[tip_key]
            pip_idx = HAND_IDXS[pip_key]
            dip_idx = HAND_IDXS[dip_key]
            mcp_idx = HAND_IDXS[mcp_key]            
    
            if conf[tip_idx] < self.confidence_hand:
                continue

            tip = kpts[tip_idx][:2]
            dist_to_wrist = np.linalg.norm(tip - wrist)  
            ratio = dist_to_wrist / palm_size
            key = prefix + fname
             
            if self.apply_optimization:
                best_src = self.optimize_joints.get(key, (0.0, 1.0)) 
                curl_val = self.map_range_adj(ratio, (best_src[1], best_src[0]), (0.0, -1.5)) 
            else:
                curl_val = self.map_range_adj(ratio, (1.6, 1.0), (0.0, -1.5))     
            
            if key in JOINTS:
                joints = JOINTS[key]
                if isinstance(joints, list):
                    for j_name in joints:
                        out[j_name] = curl_val
                        out_human[j_name] = ratio
                else:
                    out[joints] = curl_val
                    out_human[joints] = ratio                   
        
            # Print measures in screen
            cv2.circle(frame, (int(kpts[mcp_idx][0]), int(kpts[mcp_idx][1])), 4, color, -1)
            cv2.circle(frame, (int(kpts[pip_idx][0]), int(kpts[pip_idx][1])), 4, color, -1)
            cv2.circle(frame, (int(kpts[dip_idx][0]), int(kpts[dip_idx][1])), 4, color, -1)
            cv2.circle(frame, (int(kpts[tip_idx][0]), int(kpts[tip_idx][1])), 4, color, -1)

            dbg_color = (100, 200, 200)
            """cv2.putText(
                frame,
                f"{fname} {ratio:.2f} curl_val{curl_val:.2f} palm_size{palm_size:.2f}",
                (int(kpts[tip_idx][0]) + 2, int(kpts[tip_idx][1]) + (10 if side == "left" else 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                dbg_color,
                1,
            )      """   
        
        # Print measures in screen
        cv2.circle(frame, (int(kpts[HAND_IDXS["thumb_cmc"]][0]), int(kpts[HAND_IDXS["thumb_cmc"]][1])), 4, color, -1)
        cv2.circle(frame, (int(kpts[HAND_IDXS["thumb_mcp"]][0]), int(kpts[HAND_IDXS["thumb_mcp"]][1])), 4, color, -1)
        cv2.circle(frame, (int(kpts[HAND_IDXS["thumb_ip"]][0]), int(kpts[HAND_IDXS["thumb_ip"]][1])), 4, color, -1)
        cv2.circle(frame, (int(kpts[HAND_IDXS["thumb_tip"]][0]), int(kpts[HAND_IDXS["thumb_tip"]][1])), 4, color, -1)

        cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 4, color, -1)
        cv2.putText(frame, side, (int(wrist[0]), int(wrist[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)     

        return out, out_human

    # ----------------------------------------------------------------------
    # Publish 
    # ----------------------------------------------------------------------
    def publish(self) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(self.current_pos[name]) for name in self.joint_names]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 120_000_000  # 120 ms
        msg.points.append(point)
        self.publisher.publish(msg)

    # ----------------------------------------------------------------------
    # Close functions
    # ----------------------------------------------------------------------
    def destroy_node(self) -> None:
        self.save_data_to_csv()
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()   


    # Save data for plotting to CSV
    def save_data_to_csv(self):
        if not self.data_log:
            self.get_logger().info("No data to save.")
            return

        # Create unique filename based on time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gr2_movement_log_{timestamp}.csv"
        
        # Get headers dynamically from the first row
        headers = self.data_log[0].keys()
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.data_log)
            self.get_logger().info(f"Successfully saved movement data to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV: {e}")

def main(args=None) -> None:
    rclpy.init(args=args)
    cv2.destroyAllWindows()
    node = GR2Mimic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
