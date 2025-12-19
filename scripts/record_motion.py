import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import datetime
import math
import os
from pathlib import Path


# Motion Recorder for VR/AR Calibration using YOLO Pose Estimation
class MotionRecorder:
    def __init__(self):
        print("Initializing Motion Recorder...")        
       
       # Load model hands and body
        try:
            here = Path(os.path.abspath(__file__)).parent
            hand_model = here / "Outputs" / "hand_keypoint_model.pt"
            hand_name = str(hand_model if hand_model.exists() else "yolo11n-pose.pt")
            self.model_hand = YOLO(hand_name) 
            print("Loaded Custom Hand Model.")
        except:
            print("Warning: Custom Hand Model not found.")            
            
        self.model_body = YOLO("yolov11n-pose.pt")
        
        # Setup Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Data Storage
        self.data_log = []
        self.frame_count = 0
        self.is_recording = True

    def get_angle(self, a, b, c):
        """Calculates angle ABC in degrees"""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return False
        
        h, w, _ = frame.shape
        row_data = {'frame': self.frame_count}

       
        # BODY METRICS (Shoulder, Elbow, Arm Angles)      
        results_body = self.model_body(frame, verbose=False, stream=True)
        for r in results_body:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kpts = r.keypoints.data[0].cpu().numpy()
                if kpts.shape[0] >= 17:
                    self.extract_body_metrics(kpts, row_data, frame)
                    break # Only one person


        wrist_left_body, wrist_right_body = None, None
        if kpts[9][2] > 0.5:
            wrist_left_body = kpts[9][:2] 
        if kpts[10][2] > 0.5:
            wrist_right_body = kpts[10][:2]

        # HAND METRICS (Wrist, Fingers)
        results_hand = self.model_hand(frame, verbose=False, stream=True)
        detected_hands = []
        kpts_np = None
        for r in results_hand:
            if r.keypoints is not None:
                for kpts in r.keypoints.data:
                    kpts_np = kpts.cpu().numpy()
                    if kpts_np.shape[0] >= 21: detected_hands.append(kpts_np)

        wrist_hand = None
        if kpts_np is not None:
            wrist_hand = kpts_np[0][:2]       

        dist_left, dist_right = float('inf'), float('inf')
        if wrist_left_body is not None and wrist_hand is not None:
            dist_left = np.linalg.norm(wrist_hand - wrist_left_body)
        if wrist_right_body is not None and wrist_hand is not None:
            dist_right = np.linalg.norm(wrist_hand - wrist_right_body)

        # Sort hands L/R
        detected_hands.sort(key=lambda k: k[0][0])
        for kpts in detected_hands:      
            if dist_left != float('inf') and dist_right != float('inf'): 
                side = "left" if (dist_left < dist_right) else "right"
            elif dist_left != float('inf'):
                side = "left"
            elif dist_right != float('inf'):
                side = "right"
            else:
                side = "left" if kpts[0][0] < w / 2 else "Right"
            self.extract_hand_metrics(kpts, row_data, frame, side)

        # Store Data        
        if 'left_shoulder_roll_joint' in row_data or 'right_shoulder_roll_joint' in row_data:
            self.data_log.append(row_data)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Rec Indicator

        self.frame_count += 1
        cv2.imshow("Calibration Recorder", frame)
        return True

    def extract_body_metrics(self, kpts, row, img):       
        arms = [
            ("left", 5, 7, 9, 11), 
            ("right", 6, 8, 10, 12)
        ]

        for side, s_idx, e_idx, w_idx, h_idx in arms:
            if kpts[s_idx][2] < 0.5 or kpts[e_idx][2] < 0.5: continue

            s = kpts[s_idx][:2]
            e = kpts[e_idx][:2]
            w = kpts[w_idx][:2]
            
            # Use real hip or virtual hip
            if kpts[h_idx][2] > 0.5:
                h_pt = kpts[h_idx][:2]
            else:
                h_pt = np.array([s[0], s[1] + 100])

            # METRIC 1: ARM LENGTH (Pixel Reference)
            arm_len = np.linalg.norm(s - e) + 1e-6

            # METRIC 2: SHOULDER ROLL (Angle between Hip-Shoulder-Elbow)      
            roll_angle = self.get_angle(h_pt, s, e)
            row[f'{side}_shoulder_roll_joint'] = roll_angle

            # METRIC 3: SHOULDER PITCH (Vertical Offset Normalized)     
            # Positive = Elbow Up, Negative = Elbow Down
            pitch_ratio = (s[1] - e[1]) / arm_len
            row[f'{side}_shoulder_pitch_joint'] = pitch_ratio

            # METRIC 4: SHOULDER YAW (Horizontal Offset Normalized)           
            # Positive (Left) or Negative (Right) depends on direction
            yaw_ratio = (e[0] - s[0]) / arm_len
            row[f'{side}_shoulder_yaw_joint'] = yaw_ratio

            # METRIC 5: ELBOW ANGLE (Geometry)
            # Angle Shoulder-Elbow-Wrist
            elbow_angle = self.get_angle(s, e, w)
            row[f'{side}_elbow_pitch_joint'] = elbow_angle
            
            # METRIC 6: ARM EXTENSION (Shoulder to Wrist Distance)      
            ext_dist = np.linalg.norm(s - w) / arm_len
            row[f'{side}_arm_extension_raw'] = ext_dist

            color = (0, 255, 255) if side == "left" else (255, 0, 255)
            cv2.line(img, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), color, 2)
            cv2.line(img, (int(e[0]), int(e[1])), (int(w[0]), int(w[1])), color, 2)

    def extract_hand_metrics(self, kpts, row, img, side):       
        wrist = kpts[0][:2]
        middle_mcp = kpts[9][:2]
        index_mcp = kpts[5][:2]
        pinky_mcp = kpts[17][:2]       
 
        palm_size = np.linalg.norm(wrist - middle_mcp) + 1e-6

        side_key = "L_" if side == "left" else "R_"

        # METRIC 1: WRIST ROLL (Knuckle Tilt)      
        vec = pinky_mcp - index_mcp
        roll_angle = math.atan2(vec[1], vec[0])
        row[f'{side.lower()}_wrist_roll_joint'] = roll_angle

        # METRIC 2: WRIST PITCH (Forearm Alignment)      
        vec_hand = middle_mcp - wrist
        pitch_angle = math.atan2(vec_hand[1], vec_hand[0])
        row[f'{side.lower()}_wrist_pitch_joint'] = pitch_angle

         # METRIC 2: YAW PITCH (Forearm Alignment)       
        vec_hand = middle_mcp - wrist
        yaw_angle = math.atan2(pinky_mcp[1] - index_mcp[1], pinky_mcp[0] - index_mcp[0])
        row[f'{side.lower()}_wrist_yaw_joint'] = yaw_angle

        # METRIC 3: FINGER CURL RATIOS     
        fingers = {
            'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20
        }
        
        for fname, idx in fingers.items():
            if kpts[idx][2] > 0.5:
                tip = kpts[idx][:2]
                
                # Special Case for Thumb 
                if fname == 'thumb':
                    dist = np.linalg.norm(tip - pinky_mcp)
                else:
                    dist = np.linalg.norm(tip - wrist)
                
                ratio = dist / palm_size
                row[f'{side_key}{fname}'] = ratio

                cv2.circle(img, (int(tip[0]), int(tip[1])), 5, (255, 255, 0), -1)
                cv2.putText(img, f"{side_key.upper()}", (int(tip[0]), int(tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)    
                  
        cv2.circle(img, (int(wrist[0]), int(wrist[1])), 5, (0, 255, 0), -1)

    def save(self):
        if not self.data_log:
            print("No data recorded.")
            return

        filename = "calibration_data.csv"
        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} frames to {filename}")

    def run(self):
        print("Recording... Press 'q' to stop.")
        while True:
            if not self.process_frame(): break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.save()

if __name__ == '__main__':
    rec = MotionRecorder()
    rec.run()