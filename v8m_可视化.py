import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


VIDEO_FOLDER = 'video_yz/Fall'
CSV_PATH = 'video_yz/Fall.csv'
MODEL_PATH = 'yolov8m-pose.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_EVAL = True  # 保存可视化视频
EVAL_OUTPUT_FOLDER = "eval_results"
PLOT_FOLDER = "eval_plots"

# 跌倒检测阈值
FALL_SCORE_THRESHOLD = 0.55

os.makedirs(EVAL_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

print(f"使用设备: {DEVICE}")
print(f"跌倒分数阈值: {FALL_SCORE_THRESHOLD}")

# 加载模型
model_yolo = YOLO(MODEL_PATH).to(DEVICE)

# 骨骼连线
skeleton = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

BODY_CHAINS = {
    "torso_center": [0, 1, 2, 5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "head": [0, 1, 2, 3, 4],
    "shoulder_hip": [5, 11, 6, 12]
}


class KalmanFilter:
    """优化的卡尔曼滤波器"""

    def __init__(self, process_noise=0.0001, measurement_noise=0.05, error_cov_post=0.5):
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = process_noise * np.eye(4, dtype=np.float32)
        self.R = measurement_noise * np.eye(2, dtype=np.float32)
        self.P = error_cov_post * np.eye(4, dtype=np.float32)
        self.prediction_count = 0
        self.last_valid_measurement = None

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.prediction_count += 1
        return self.state[:2].flatten()

    def update(self, measurement):
        if np.any(np.isnan(measurement)):
            return self.state[:2].flatten()

        if self.last_valid_measurement is not None:
            movement = np.linalg.norm(measurement - self.last_valid_measurement)
            if movement > 100:
                return self.state[:2].flatten()

        self.prediction_count = 0
        self.last_valid_measurement = measurement.copy()
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()


class EnhancedPoseDetector:
    """优化的姿态检测器"""

    def __init__(self):
        self.pose_history = []
        self.max_history = 8
        self.last_keypoints = None
        self.last_confidences = None
        self.kalman_filters = [KalmanFilter() for _ in range(17)]
        self.fall_static_mode = False
        self.fall_static_counter = 0
        self.last_reliable_pose = None
        self.last_reliable_conf = None
        self.low_confidence_counter = 0
        self.stable_pose_cache = None
        self.frame_counter = 0

        self.min_detection_confidence = 0.1
        self.min_keypoint_confidence = 0.05

    def chain_interpolation(self, keypoints, confidences):
        kpts = keypoints.copy()
        confs = confidences.copy()

        torso_points = BODY_CHAINS["torso_center"]
        torso_indices = [i for i in torso_points if confs[i] > 0.2]
        if len(torso_indices) >= 2:
            valid_torso_kpts = [kpts[i] for i in torso_indices]
            torso_center = np.mean(valid_torso_kpts, axis=0)

            for i in range(17):
                if confs[i] < 0.05 and np.all(kpts[i] < 0):
                    if i in [0, 1, 2, 3, 4]:
                        kpts[i] = torso_center + np.array([0, -50])
                        confs[i] = 0.2
                    elif i in [5, 7, 9]:
                        kpts[i] = torso_center + np.array([-35, 0])
                        confs[i] = 0.2
                    elif i in [6, 8, 10]:
                        kpts[i] = torso_center + np.array([35, 0])
                        confs[i] = 0.2

        for chain_name, chain_indices in BODY_CHAINS.items():
            if chain_name == "shoulder_hip":
                continue

            valid_indices = [idx for idx in chain_indices if confs[idx] > 0.15]

            if len(valid_indices) >= 2:
                for i in range(len(chain_indices)):
                    idx = chain_indices[i]
                    if confs[idx] < 0.2:
                        prev_valid = None
                        next_valid = None

                        for j in range(i - 1, -1, -1):
                            if confs[chain_indices[j]] > 0.15:
                                prev_valid = chain_indices[j]
                                break

                        for j in range(i + 1, len(chain_indices)):
                            if confs[chain_indices[j]] > 0.15:
                                next_valid = chain_indices[j]
                                break

                        if prev_valid is not None and next_valid is not None:
                            prev_pos, next_pos = kpts[prev_valid], kpts[next_valid]
                            total_dist = np.linalg.norm(next_pos - prev_pos)

                            if total_dist > 1:
                                prev_idx_in_chain = chain_indices.index(prev_valid)
                                next_idx_in_chain = chain_indices.index(next_valid)
                                ratio = (i - prev_idx_in_chain) / (next_idx_in_chain - prev_idx_in_chain)

                                kpts[idx] = prev_pos + ratio * (next_pos - prev_pos)
                                confs[idx] = max(confs[idx], 0.3)

        for left_idx, right_idx in [(5, 6), (11, 12)]:
            if confs[left_idx] > 0.2 and confs[right_idx] > 0.2:
                center = (kpts[left_idx] + kpts[right_idx]) / 2
                if confs[left_idx] > confs[right_idx]:
                    kpts[right_idx] = 2 * center - kpts[left_idx]
                    confs[right_idx] = max(confs[right_idx], 0.3)
                else:
                    kpts[left_idx] = 2 * center - kpts[right_idx]
                    confs[left_idx] = max(confs[left_idx], 0.3)

        kpts, confs = self.apply_body_proportions(kpts, confs)
        return kpts, confs

    def apply_body_proportions(self, keypoints, confidences):
        kpts = keypoints.copy()
        confs = confidences.copy()
        visible_indices = [i for i in range(17) if confs[i] > 0.2]

        if len(visible_indices) >= 4:
            shoulder_width = 0
            if confs[5] > 0.2 and confs[6] > 0.2:
                shoulder_width = np.linalg.norm(kpts[5] - kpts[6])

            if shoulder_width > 10:
                arm_scale = 1.3

                if confs[5] > 0.15 and confs[7] > 0.1:
                    arm_length = np.linalg.norm(kpts[7] - kpts[5])
                    if arm_length > shoulder_width * 3.0 or arm_length < shoulder_width * 0.3:
                        direction = kpts[7] - kpts[5]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            kpts[7] = kpts[5] + direction * shoulder_width * arm_scale
                            confs[7] = max(confs[7], 0.25)

                if confs[6] > 0.15 and confs[8] > 0.1:
                    arm_length = np.linalg.norm(kpts[8] - kpts[6])
                    if arm_length > shoulder_width * 3.0 or arm_length < shoulder_width * 0.3:
                        direction = kpts[8] - kpts[6]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            kpts[8] = kpts[6] + direction * shoulder_width * arm_scale
                            confs[8] = max(confs[8], 0.25)

        return kpts, confs

    def temporal_smoothing(self, current_keypoints, current_confidences, frame_shape):
        smoothed_kpts = current_keypoints.copy()
        smoothed_conf = current_confidences.copy()

        h, w = frame_shape[:2]

        visible_confs = current_confidences[current_confidences > 0.1]
        avg_confidence = np.mean(visible_confs) if len(visible_confs) > 0 else 0

        movement = 0
        valid_count = 0
        if len(self.pose_history) > 0:
            prev_kpts, prev_conf = self.pose_history[-1]
            for i in range(17):
                if current_confidences[i] > 0.1 and prev_conf[i] > 0.1:
                    movement += np.linalg.norm(current_keypoints[i] - prev_kpts[i])
                    valid_count += 1

        avg_movement = movement / max(valid_count, 1) if valid_count > 0 else 0

        is_low_confidence = avg_confidence < 0.25
        is_low_movement = avg_movement < 2.5

        if is_low_confidence:
            self.low_confidence_counter += 1
        else:
            self.low_confidence_counter = max(0, self.low_confidence_counter - 1)

        if not self.fall_static_mode and self.low_confidence_counter > 5 and is_low_movement:
            self.fall_static_mode = True
            self.fall_static_counter = 0

            if len(self.pose_history) > 0:
                for hist_kpts, hist_confs in reversed(self.pose_history):
                    if np.mean(hist_confs[hist_confs > 0]) > 0.3:
                        self.stable_pose_cache = hist_kpts.copy()
                        break
                if self.stable_pose_cache is None:
                    self.stable_pose_cache = self.pose_history[-1][0].copy()

        if self.fall_static_mode:
            self.fall_static_counter += 1

            if avg_movement > 10.0 or avg_confidence > 0.4 or self.fall_static_counter > 100:
                self.fall_static_mode = False
                self.low_confidence_counter = 0
                self.fall_static_counter = 0

        if self.fall_static_mode and self.stable_pose_cache is not None:
            for i in range(17):
                cache_kpt = self.stable_pose_cache[i]

                if current_confidences[i] < 0.15 or np.all(current_keypoints[i] < 0):
                    smoothed_kpts[i] = cache_kpt
                    smoothed_conf[i] = 0.25
                else:
                    weight = 0.8
                    smoothed_kpts[i] = (1 - weight) * current_keypoints[i] + weight * cache_kpt
                    smoothed_conf[i] = current_confidences[i] * (1 - weight * 0.3)

                    smoothed_kpts[i][0] = np.clip(smoothed_kpts[i][0], 0, w - 1)
                    smoothed_kpts[i][1] = np.clip(smoothed_kpts[i][1], 0, h - 1)

            self.pose_history.append((smoothed_kpts.copy(), smoothed_conf.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

            self.frame_counter += 1
            return smoothed_kpts, smoothed_conf

        if avg_movement > 50:
            process_noise = 0.05
            measurement_noise = 0.05
        elif avg_movement > 20:
            process_noise = 0.01
            measurement_noise = 0.03
        else:
            process_noise = 0.0001
            measurement_noise = 0.01

        for kf in self.kalman_filters:
            kf.Q = process_noise * np.eye(4, dtype=np.float32)
            kf.R = measurement_noise * np.eye(2, dtype=np.float32)

        for i in range(17):
            measurement = current_keypoints[i]
            conf = current_confidences[i]

            predicted = self.kalman_filters[i].predict()

            if conf > 0.15 and not np.any(np.isnan(measurement)):
                smoothed_kpts[i] = self.kalman_filters[i].update(measurement)
                smoothed_conf[i] = conf

                if conf > 0.4 and not self.fall_static_mode:
                    self.last_reliable_pose = smoothed_kpts.copy()
                    self.last_reliable_conf = smoothed_conf.copy()
                    self.stable_pose_cache = smoothed_kpts.copy()
            else:
                smoothed_kpts[i] = predicted
                smoothed_conf[i] = 0.2

            if conf < 0.15 and len(self.pose_history) > 0 and prev_conf[i] > 0.15 and avg_movement < 20:
                smoothed_kpts[i] = 0.3 * smoothed_kpts[i] + 0.7 * prev_kpts[i]
                smoothed_conf[i] = prev_conf[i] * 0.6

        if np.mean(current_confidences[current_confidences > 0]) > 0.15:
            self.pose_history.append((smoothed_kpts.copy(), smoothed_conf.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

        self.frame_counter += 1
        return smoothed_kpts, smoothed_conf

    def detect_pose(self, image):
        """检测姿态"""
        if self.fall_static_mode:
            conf_thresh = 0.05
            iou_thresh = 0.2
        else:
            conf_thresh = 0.15
            iou_thresh = 0.3

        results = model_yolo(image, imgsz=640, conf=conf_thresh, iou=iou_thresh,
                             device=DEVICE, verbose=False, max_det=3)

        kpts, confs = self.process_yolo_results(results, image.shape)

        if kpts is not None:
            self.last_keypoints = kpts.copy()
            self.last_confidences = confs.copy()

            avg_conf = np.mean(confs[confs > 0.1]) if len(confs[confs > 0.1]) > 0 else 0
            if avg_conf > 0.3:
                self.last_reliable_pose = kpts.copy()
                self.last_reliable_conf = confs.copy()
                if not self.fall_static_mode:
                    self.stable_pose_cache = kpts.copy()
        else:
            if self.fall_static_mode and self.last_reliable_pose is not None:
                kpts = self.last_reliable_pose.copy()
                confs = self.last_reliable_conf.copy() * 0.8
            elif self.last_keypoints is not None:
                kpts = self.last_keypoints.copy()
                confs = self.last_confidences.copy() * 0.7
            else:
                return None, None

        kpts, confs = self.chain_interpolation(kpts, confs)
        kpts, confs = self.temporal_smoothing(kpts, confs, image.shape)

        return kpts, confs

    def process_yolo_results(self, results, image_shape):
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None, None
        boxes = results[0].boxes
        keypoints_data = results[0].keypoints
        max_area, best_idx = 0, 0
        for i, box in enumerate(boxes):
            area = box.xywh[0][2].item() * box.xywh[0][3].item()
            if area > max_area:
                max_area = area
                best_idx = i
        if best_idx >= len(keypoints_data.xy) or len(keypoints_data.xy[best_idx]) == 0:
            return None, None
        xy = keypoints_data.xy[best_idx].cpu().numpy()
        conf = keypoints_data.conf[best_idx].cpu().numpy() if hasattr(keypoints_data,
                                                                      'conf') and keypoints_data.conf is not None else np.ones(
            17) * 0.3
        h, w = image_shape[:2]
        kpts = np.full((17, 2), -1.0, dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)
        for i in range(17):
            if i < len(xy):
                x, y = xy[i]
                if 0 < x < w and 0 < y < h:
                    kpts[i] = [float(x), float(y)]
                    confs[i] = float(conf[i] if i < len(conf) else 0.0)
        return kpts, confs


class PoseHistory:
    """姿态历史管理"""

    def __init__(self, max_frames=15):
        self.max_frames = max_frames
        self.ankle_positions = []
        self.fall_flags = []
        self.fall_scores = []
        self.fall_confirmed_frames = 0
        self.visible_keypoints_history = []
        self.fall_start_frame = None
        self.recovery_frames = 0
        self.consecutive_non_fall = 0

    def add_ankle_positions(self, left_ankle, right_ankle):
        self.ankle_positions.append((left_ankle, right_ankle))
        if len(self.ankle_positions) > self.max_frames:
            self.ankle_positions.pop(0)

    def add_fall_flag(self, is_falling, fall_score=0):
        self.fall_flags.append(1 if is_falling else 0)
        self.fall_scores.append(fall_score)

        if len(self.fall_flags) > 10:
            self.fall_flags.pop(0)
            self.fall_scores.pop(0)

        if is_falling:
            if self.fall_start_frame is None:
                self.fall_start_frame = len(self.fall_flags) - 1
            self.fall_confirmed_frames = min(self.fall_confirmed_frames + 2, 20)
            self.consecutive_non_fall = 0
            self.recovery_frames = 0
        else:
            self.fall_confirmed_frames = max(self.fall_confirmed_frames - 1, 0)
            self.consecutive_non_fall += 1
            if self.consecutive_non_fall > 10:
                self.recovery_frames += 1
                if self.recovery_frames > 15:
                    self.fall_start_frame = None

    def add_visible_keypoints(self, visible_count):
        self.visible_keypoints_history.append(visible_count)
        if len(self.visible_keypoints_history) > 5:
            self.visible_keypoints_history.pop(0)

    def get_movement_magnitude(self):
        if len(self.ankle_positions) < 2:
            return 0
        total = count = 0
        for i in range(1, len(self.ankle_positions)):
            prev_l, prev_r = self.ankle_positions[i - 1]
            curr_l, curr_r = self.ankle_positions[i]
            if prev_l and curr_l:
                total += np.linalg.norm(np.array(curr_l) - np.array(prev_l))
                count += 1
            if prev_r and curr_r:
                total += np.linalg.norm(np.array(curr_r) - np.array(prev_r))
                count += 1
        return total / count if count > 0 else 0

    def is_fall_confirmed(self):
        if len(self.fall_flags) < 4:
            return False

        recent_flags = self.fall_flags[-4:]
        if sum(recent_flags) >= 3:
            return True

        if self.fall_confirmed_frames >= 4:
            return True

        if len(self.fall_scores) >= 3:
            recent_scores = self.fall_scores[-3:]
            if all(score > 0.6 for score in recent_scores):
                return True

        return False

    def should_end_fall_detection(self):
        if len(self.fall_flags) < 8:
            return False

        recent_flags = self.fall_flags[-8:]
        if sum(recent_flags) <= 1:
            return True

        if len(self.fall_scores) >= 5:
            recent_scores = self.fall_scores[-5:]
            if max(recent_scores) < 0.4:
                return True

        return False


def detect_walking(keypoints, history, width, height):
    if len(keypoints) < 17:
        return False
    leg_indices = [13, 14, 15, 16]
    valid_leg_points = sum(1 for i in leg_indices if keypoints[i][0] > 0)
    if valid_leg_points < 2:
        return False
    left_ankle = (keypoints[15][0], keypoints[15][1]) if keypoints[15][0] > 0 else None
    right_ankle = (keypoints[16][0], keypoints[16][1]) if keypoints[16][0] > 0 else None
    history.add_ankle_positions(left_ankle, right_ankle)
    if len(history.ankle_positions) < 3:
        return False
    movement = history.get_movement_magnitude()
    return 3.0 < movement < 40.0


def detect_falling(keypoints, width, height, history=None):
    """改进的跌倒检测算法"""
    try:
        required_indices = [5, 6, 11, 12]
        visible_count = 0
        for i in required_indices:
            if 0 < keypoints[i][0] < width and 0 < keypoints[i][1] < height:
                visible_count += 1

        if visible_count < 3:
            return False, 0, False, 0.0, 0.0

        chest_x = 0
        chest_y = 0
        waist_x = 0
        waist_y = 0
        chest_count = 0
        waist_count = 0

        if keypoints[5][0] > 0:
            chest_x += keypoints[5][0]
            chest_y += keypoints[5][1]
            chest_count += 1
        if keypoints[6][0] > 0:
            chest_x += keypoints[6][0]
            chest_y += keypoints[6][1]
            chest_count += 1
        if keypoints[11][0] > 0:
            waist_x += keypoints[11][0]
            waist_y += keypoints[11][1]
            waist_count += 1
        if keypoints[12][0] > 0:
            waist_x += keypoints[12][0]
            waist_y += keypoints[12][1]
            waist_count += 1

        if chest_count == 0 or waist_count == 0:
            return False, 0, False, 0.0, 0.0

        chest_x /= chest_count
        chest_y /= chest_count
        waist_x /= waist_count
        waist_y /= waist_count

        v_up = np.array([0, -1])
        v_body = np.array([waist_x - chest_x, waist_y - chest_y])

        if np.linalg.norm(v_body) > 0:
            v_body = v_body / np.linalg.norm(v_body)
            angle = np.degrees(np.arccos(np.clip(np.dot(v_up, v_body), -1.0, 1.0)))
        else:
            angle = 0

        torso_points = []
        for i in [5, 6, 11, 12]:
            if 0 < keypoints[i][0] < width:
                torso_points.append(keypoints[i])

        if len(torso_points) >= 2:
            xs = [p[0] for p in torso_points]
            ys = [p[1] for p in torso_points]
            width_body = max(xs) - min(xs)
            height_body = max(ys) - min(ys)
            aspect = width_body / height_body if height_body > 10 else 0
        else:
            aspect = 0

        angle_condition = angle > 40
        vertical_condition = chest_y > waist_y + 10
        aspect_condition = aspect > 0.8

        head_near_ground = False
        if keypoints[0][1] > 0:
            head_near_ground = keypoints[0][1] > height * 0.75

        ankle_ys = []
        for i in [15, 16]:
            if keypoints[i][1] > 0:
                ankle_ys.append(keypoints[i][1])

        ankles_near_ground = False
        if ankle_ys:
            avg_ankle_y = np.mean(ankle_ys)
            ankles_near_ground = avg_ankle_y > height * 0.8

        fall_score = 0.0
        if angle_condition:
            fall_score += 0.40
        if vertical_condition:
            fall_score += 0.35
        if aspect_condition:
            fall_score += 0.15
        if head_near_ground:
            fall_score += 0.07
        if ankles_near_ground:
            fall_score += 0.03

        is_fall_candidate = fall_score >= FALL_SCORE_THRESHOLD

        return is_fall_candidate, angle, vertical_condition, aspect, fall_score

    except Exception:
        return False, 0, False, 0.0, 0.0


def draw_keypoints_and_status(frame, keypoints, confidences, status, fall_details=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for (i, j) in skeleton:
        if keypoints[i][0] > 0 and keypoints[j][0] > 0:
            conf = (confidences[i] + confidences[j]) / 2
            color_intensity = int(255 * conf)
            cv2.line(overlay,
                     (int(keypoints[i][0]), int(keypoints[i][1])),
                     (int(keypoints[j][0]), int(keypoints[j][1])),
                     (0, color_intensity, 0), 2)

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            conf = confidences[i]
            color = (0, 0, 255) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 255, 255)
            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
            cv2.putText(overlay, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    status_colors = {"Falling": (0, 0, 255), "Walking": (255, 0, 0), "Standing": (0, 255, 0),
                     "Not detected": (128, 128, 128)}
    color = status_colors.get(status, (255, 255, 255))
    cv2.putText(overlay, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    y_offset = 60
    if status == "Falling" and fall_details:
        angle, vertical_cond, aspect, fall_score = fall_details
        cv2.putText(overlay, f"Angle: {angle:.1f}deg", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
        cv2.putText(overlay, f"Chest>Waist: {'True' if vertical_cond else 'False'}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
        cv2.putText(overlay, f"Aspect: {aspect:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
        cv2.putText(overlay, f"Score: {fall_score:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    visible_count = sum(1 for x, y in keypoints if x > 0 and y > 0)
    cv2.putText(overlay, f"Keypoints: {visible_count}/17", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return overlay


def parse_falling_intervals(classes_str):
    intervals = []
    pattern = r"Falling.*?(\[([\d\.]+)\s*to\s*([\d\.]+)\]|\[([\d\.]+)\])"
    matches = re.findall(pattern, classes_str)
    for match in matches:
        if match[1] and match[2]:
            intervals.append((float(match[1]), float(match[2])))
        elif match[3]:
            intervals.append((float(match[3]), float('inf')))
    return intervals


def get_frame_label(frame_idx, fps, falling_intervals):
    if fps <= 0:
        return False
    current_sec = frame_idx / fps
    for start, end in falling_intervals:
        if current_sec >= start and (end == float('inf') or current_sec <= end):
            return True
    return False


def optimize_predictions(y_pred, fall_scores, window_size=7):
    """优化预测结果"""
    if len(y_pred) != len(fall_scores):
        return y_pred

    optimized_pred = y_pred.copy()

    for i in range(len(y_pred)):
        start = max(0, i - window_size)
        end = min(len(y_pred), i + window_size + 1)
        window_scores = fall_scores[start:end]
        window_preds = y_pred[start:end]

        window_fall_count = sum(window_preds)
        window_avg_score = np.mean(window_scores) if len(window_scores) > 0 else 0

        if window_fall_count >= 3 and window_avg_score > 0.6:
            if abs(i - (start + end) // 2) <= window_size // 2:
                optimized_pred[i] = True

        elif window_avg_score < 0.4 and window_fall_count <= 1:
            optimized_pred[i] = False

        elif fall_scores[i] > 0.7 and window_fall_count == 1:
            optimized_pred[i] = False

    return optimized_pred


def calculate_detailed_metrics(y_true, y_pred, video_name=""):
    """计算详细的评估指标"""
    tp = fp = tn = fn = 0

    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif not t and not p:
            tn += 1
        elif t and not p:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    balanced_accuracy = (recall + specificity) / 2

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': len(y_true)
    }


def evaluate_single_video(video_path, falling_intervals, output_video_path=None):
    """评估单个视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detection_stats = {
        'frames_with_kpts': 0,
        'frames_without_kpts': 0,
        'frames_with_high_conf': 0,
        'fall_candidate_frames': 0,
        'fall_score_details': [],
        'angle_details': [],
        'visible_kpts_counts': [],
        'confidence_details': []
    }

    out_vid = None
    if OUTPUT_EVAL and output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        out_vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    detector = EnhancedPoseDetector()
    history = PoseHistory()

    y_true = []
    y_pred = []
    fall_scores = []

    pbar = tqdm(total=total_frames, desc=os.path.basename(video_path), leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kpts, confs = detector.detect_pose(frame)

        if kpts is not None:
            detection_stats['frames_with_kpts'] += 1
            visible_count = sum(1 for x, y in kpts if x > 0 and y > 0)
            detection_stats['visible_kpts_counts'].append(visible_count)

            if len(confs[confs > 0]) > 0:
                avg_conf = np.mean(confs[confs > 0])
            else:
                avg_conf = 0
            detection_stats['confidence_details'].append(avg_conf)

            if np.mean(confs[confs > 0]) > 0.3:
                detection_stats['frames_with_high_conf'] += 1

            is_fall_candidate, angle, vertical_cond, aspect, fall_score = detect_falling(kpts, width, height, history)

            detection_stats['fall_score_details'].append(fall_score)
            detection_stats['angle_details'].append(angle)

            if is_fall_candidate:
                detection_stats['fall_candidate_frames'] += 1

            history.add_fall_flag(is_fall_candidate, fall_score)

            pred_falling = history.is_fall_confirmed()

            if pred_falling and history.should_end_fall_detection():
                pred_falling = False

            status = "Falling" if pred_falling else (
                "Walking" if detect_walking(kpts, history, width, height) else "Standing")
            details = (angle, vertical_cond, aspect, fall_score) if pred_falling else None
            current_kpts = kpts
            current_confs = confs
        else:
            detection_stats['frames_without_kpts'] += 1
            detection_stats['confidence_details'].append(0)
            pred_falling = False
            status = "Not detected"
            details = None
            fall_score = 0.0
            current_kpts = np.full((17, 2), -1.0)
            current_confs = np.zeros(17)
            is_fall_candidate = False

        if out_vid is not None:
            out_vid.write(draw_keypoints_and_status(frame, current_kpts, current_confs, status, details))

        y_true.append(get_frame_label(len(y_true), fps, falling_intervals))
        y_pred.append(pred_falling)
        fall_scores.append(fall_score)

        pbar.update(1)

    pbar.close()
    cap.release()
    if out_vid is not None:
        out_vid.release()

    if len(fall_scores) > 0:
        optimized_pred = optimize_predictions(y_pred, fall_scores)
        y_pred = optimized_pred

    if len(detection_stats['fall_score_details']) > 0:
        detection_stats['avg_fall_score'] = np.mean(detection_stats['fall_score_details'])
        detection_stats['max_fall_score'] = np.max(detection_stats['fall_score_details'])
        detection_stats['avg_angle'] = np.mean(detection_stats['angle_details'])
        detection_stats['max_angle'] = np.max(detection_stats['angle_details'])
    else:
        detection_stats['avg_fall_score'] = 0
        detection_stats['max_fall_score'] = 0
        detection_stats['avg_angle'] = 0
        detection_stats['max_angle'] = 0

    if len(detection_stats['visible_kpts_counts']) > 0:
        detection_stats['avg_visible_kpts'] = np.mean(detection_stats['visible_kpts_counts'])
    else:
        detection_stats['avg_visible_kpts'] = 0

    if len(detection_stats['confidence_details']) > 0:
        detection_stats['avg_confidence'] = np.mean(detection_stats['confidence_details'])
    else:
        detection_stats['avg_confidence'] = 0

    return y_true, y_pred, fall_scores, detection_stats


# 可视化函数
def plot_per_video_metrics(video_names, metrics):
    """每视频指标图"""
    plt.style.use('seaborn-v0_8-darkgrid')

    x = np.arange(len(video_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(12, len(video_names) * 0.7), 9))

    colors = plt.cm.viridis([0.2, 0.4, 0.6, 0.8])

    rects1 = ax.bar(x - width * 1.5, metrics['prec'], width, label='Precision', color=colors[0],
                    edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    rects2 = ax.bar(x - width * 0.5, metrics['rec'], width, label='Recall', color=colors[1],
                    edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    rects3 = ax.bar(x + width * 0.5, metrics['f1'], width, label='F1 Score', color=colors[2],
                    edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    rects4 = ax.bar(x + width * 1.5, metrics['acc'], width, label='Accuracy', color=colors[3],
                    edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)

    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Videos', fontsize=14, fontweight='bold')
    ax.set_title('Performance Metrics per Video', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 1.15])

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, shadow=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(axis='both', which='major', labelsize=12)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='gray',
                                      linewidth=0.5))

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'per_video_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(PLOT_FOLDER, 'per_video_metrics.svg'), format='svg', bbox_inches='tight',
                facecolor='white')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """混淆矩阵"""
    plt.style.use('seaborn-v0_8-darkgrid')

    cm = confusion_matrix(y_true, y_pred)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'],
                annot_kws={"size": 16, "fontweight": "bold"},
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'{cm_percent[i, j]:.1f}%',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, fontweight='bold', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'confusion_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(PLOT_FOLDER, 'confusion_matrix.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pr_roc(y_true, y_pred, fall_scores):
    """PR-ROC曲线"""
    if sum(y_true) == 0 or len(set(fall_scores)) < 2:
        return

    plt.style.use('seaborn-v0_8-darkgrid')

    precision, recall, _ = precision_recall_curve(y_true, fall_scores)
    fpr, tpr, _ = roc_curve(y_true, fall_scores)
    auc_pr = auc(recall, precision)
    auc_roc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(recall, precision, 'b-', linewidth=3, label=f'PR Curve (AUC = {auc_pr:.3f})', alpha=0.8)
    ax1.fill_between(recall, precision, alpha=0.2, color='blue')
    ax1.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax1.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=12, framealpha=0.9, shadow=True)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2.plot(fpr, tpr, 'r-', linewidth=3, label=f'ROC Curve (AUC = {auc_roc:.3f})', alpha=0.8)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2)
    ax2.fill_between(fpr, tpr, alpha=0.2, color='red')
    ax2.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=12, framealpha=0.9, shadow=True)
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'pr_roc_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(PLOT_FOLDER, 'pr_roc_curves.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_performance_summary(metrics):
    """整体性能总结"""
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Specificity', 'Balanced Accuracy']
    metrics_values = [
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['accuracy'],
        metrics['specificity'],
        metrics['balanced_accuracy']
    ]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 6))

    for idx, (ax, metric_name, value, color) in enumerate(zip(axes.flatten(), metrics_names, metrics_values, colors)):
        bars = ax.bar([0], [value], color=color, edgecolor='black', linewidth=2, width=0.7,
                      alpha=0.9, zorder=3)

        for bar in bars:
            bar.set_hatch('//')

        ax.set_ylim([0, 1.15])
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.set_xticks([])
        ax.set_axisbelow(True)

        ax.text(0, value + 0.02, f'{value:.3f}', ha='center', va='bottom',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('gray')

    plt.suptitle('Overall Performance Summary', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'performance_summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(PLOT_FOLDER, 'performance_summary.svg'), format='svg', bbox_inches='tight',
                facecolor='white')
    plt.close()


def plot_keypoint_analysis(video_diagnostics):
    """关键点分析图表"""
    if not video_diagnostics:
        return

    plt.style.use('seaborn-v0_8-darkgrid')

    video_names = [diag['video'] for diag in video_diagnostics]
    visible_kpts = [diag['detection_stats']['avg_visible_kpts'] for diag in video_diagnostics]
    confidences = [diag['detection_stats']['avg_confidence'] for diag in video_diagnostics]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.bar(video_names, visible_kpts, color=plt.cm.coolwarm(np.linspace(0.3, 0.8, len(video_names))),
            edgecolor='black', linewidth=1.5, alpha=0.9)
    ax1.set_title('Average Visible Keypoints per Video', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Video', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Keypoints', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(video_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.4)
    ax1.set_ylim([0, 18])
    ax1.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='Good threshold (12)')
    ax1.legend()

    for i, v in enumerate(visible_kpts):
        ax1.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    bars = ax2.bar(video_names, confidences, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(video_names))),
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    ax2.set_title('Average Keypoint Confidence per Video', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Video', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(video_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, axis='y', alpha=0.4)
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Threshold (0.3)')
    ax2.legend()

    for i, v in enumerate(confidences):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    sc = ax3.scatter(visible_kpts, confidences, c=range(len(video_names)),
                     cmap='viridis', s=200, alpha=0.8, edgecolors='black', linewidth=2)
    ax3.set_title('Keypoints vs Confidence Correlation', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Average Keypoints', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.4)
    ax3.set_xlim([0, 18])
    ax3.set_ylim([0, 1])

    for i, (x, y, name) in enumerate(zip(visible_kpts, confidences, video_names)):
        ax3.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold', alpha=0.8)

    all_visible_kpts = []
    for diag in video_diagnostics:
        all_visible_kpts.extend(diag['detection_stats']['visible_kpts_counts'])

    if all_visible_kpts:
        ax4.hist(all_visible_kpts, bins=range(0, 18, 1), color='skyblue',
                 edgecolor='black', linewidth=1.5, alpha=0.8, density=True)
        ax4.set_title('Distribution of Visible Keypoints (All Frames)', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Number of Visible Keypoints', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.4)
        ax4.set_xlim([0, 18])

        mean_val = np.mean(all_visible_kpts)
        median_val = np.median(all_visible_kpts)
        ax4.axvline(x=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax4.axvline(x=median_val, color='green', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Median: {median_val:.1f}')
        ax4.legend()

    plt.suptitle('Keypoint Detection Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, 'keypoint_analysis.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(PLOT_FOLDER, 'keypoint_analysis.svg'), format='svg', bbox_inches='tight',
                facecolor='white')
    plt.close()


# 主程序
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"标注文件不存在: {CSV_PATH}")
        exit()

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    all_y_true = []
    all_y_pred = []
    all_fall_scores = []

    video_names = []
    video_metrics = {'prec': [], 'rec': [], 'f1': [], 'acc': []}
    detailed_results = []
    video_diagnostics = []

    print("开始评估...")

    for idx, row in df.iterrows():
        video_name = row["File Name"].strip()
        video_path = os.path.join(VIDEO_FOLDER, video_name)

        if not os.path.exists(video_path):
            continue

        falling_intervals = parse_falling_intervals(row["Classes"])

        if not falling_intervals:
            continue

        output_eval_path = os.path.join(EVAL_OUTPUT_FOLDER, video_name) if OUTPUT_EVAL else None

        print(f"处理: {video_name}")
        y_true, y_pred, fall_scores, detection_stats = evaluate_single_video(
            video_path, falling_intervals, output_eval_path
        )

        if y_true is None:
            continue

        metrics = calculate_detailed_metrics(y_true, y_pred, video_name)

        video_names.append(video_name[:15])
        video_metrics['prec'].append(metrics['precision'])
        video_metrics['rec'].append(metrics['recall'])
        video_metrics['f1'].append(metrics['f1'])
        video_metrics['acc'].append(metrics['accuracy'])

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        if fall_scores:
            all_fall_scores.extend(fall_scores)

        detailed_results.append({
            'video': video_name,
            'total_frames': metrics['total'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy']
        })

        video_diagnostics.append({
            'video': video_name,
            'detection_stats': detection_stats,
            'metrics': metrics
        })

    # 保存结果
    if detailed_results:
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv(os.path.join(PLOT_FOLDER, 'detailed_results.csv'), index=False)

    # 绘制图表
    if video_names:
        plot_per_video_metrics(video_names, video_metrics)

        if len(all_y_true) > 0:
            plot_confusion_matrix(all_y_true, all_y_pred)

            if all_fall_scores and len(set(all_fall_scores)) > 1:
                plot_pr_roc(all_y_true, all_y_pred, all_fall_scores)

            overall_metrics = calculate_detailed_metrics(all_y_true, all_y_pred, "所有视频")
            plot_performance_summary(overall_metrics)

            plot_keypoint_analysis(video_diagnostics)

    # 输出结果
    if all_y_true:
        print("\n所有视频的评估结果:")
        print(f"处理的总视频数: {len(video_names)}")
        print(f"使用总帧数: {len(all_y_true)}")

        overall_metrics = calculate_detailed_metrics(all_y_true, all_y_pred, "所有视频")
        print(f"\n整体指标:")
        print(f"Precision: {overall_metrics['precision']:.4f}")
        print(f"Recall: {overall_metrics['recall']:.4f}")
        print(f"F1 Score: {overall_metrics['f1']:.4f}")
        print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"Specificity: {overall_metrics['specificity']:.4f}")
        print(f"Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")

        print(f"\n混淆矩阵:")
        print(f"TP: {overall_metrics['tp']}")
        print(f"FP: {overall_metrics['fp']}")
        print(f"TN: {overall_metrics['tn']}")
        print(f"FN: {overall_metrics['fn']}")

    print(f"\n可视化视频已保存至: {EVAL_OUTPUT_FOLDER}")
    print(f"分析图表已保存至: {PLOT_FOLDER}")
    if detailed_results:
        print(f"详细结果已保存至: {os.path.join(PLOT_FOLDER, 'detailed_results.csv')}")