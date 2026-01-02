import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont  # 添加PIL库用于中文显示

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载你的最佳模型
model_yolo = YOLO('yolov8m-pose.pt').to(device)

# COCO关键点定义
KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# 人体关键点链式结构
BODY_CHAINS = {
    "torso_center": [0, 1, 2, 5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "head": [0, 1, 2, 3, 4],
    "shoulder_hip": [5, 11, 6, 12]
}

# 骨骼连接关系（用于绘制）
skeleton = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15),  # 左腿
    (12, 14), (14, 16)  # 右腿
]


# 卡尔曼滤波器类（保持不变）
class KalmanFilter:
    def __init__(self, process_noise=0.0001, measurement_noise=0.05, error_cov_post=0.5):
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.Q = process_noise * np.eye(4, dtype=np.float32)
        self.R = measurement_noise * np.eye(2, dtype=np.float32)
        self.P = error_cov_post * np.eye(4, dtype=np.float32)

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, measurement):
        if np.any(np.isnan(measurement)):
            return self.state[:2].flatten()
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()


# 增强姿态检测器类（保持不变）
class EnhancedPoseDetector:
    def __init__(self):
        self.pose_history = []
        self.max_history = 8
        self.last_keypoints = None
        self.last_confidences = None
        self.kalman_filters = [KalmanFilter() for _ in range(17)]
        self.debug_info = []

    def has_valid_chain(self, chain_name, confs):
        chain_indices = BODY_CHAINS[chain_name]
        return any(confs[idx] > 0.25 for idx in chain_indices)

    def mirror_chain(self, kpts, confs, source_chain, target_chain):
        source_indices = BODY_CHAINS[source_chain]
        target_indices = BODY_CHAINS[target_chain]

        if len(source_indices) != len(target_indices):
            return kpts, confs

        if confs[5] > 0.25 and confs[6] > 0.25:
            center_x = (kpts[5][0] + kpts[6][0]) / 2
        elif confs[11] > 0.25 and confs[12] > 0.25:
            center_x = (kpts[11][0] + kpts[12][0]) / 2
        else:
            return kpts, confs

        for src_idx, tgt_idx in zip(source_indices, target_indices):
            if confs[src_idx] > 0.25 and confs[tgt_idx] < 0.15:
                kpts[tgt_idx][0] = 2 * center_x - kpts[src_idx][0]
                kpts[tgt_idx][1] = kpts[src_idx][1]
                confs[tgt_idx] = confs[src_idx] * 0.7

        return kpts, confs

    def body_model_completion(self, kpts, confs):
        torso_visible = all(confs[i] > 0.25 for i in [5, 6, 11, 12])

        if torso_visible:
            shoulder_center = (kpts[5] + kpts[6]) / 2
            hip_center = (kpts[11] + kpts[12]) / 2
            torso_height = np.linalg.norm(hip_center - shoulder_center)

            if torso_height < 1:
                return kpts, confs

            if confs[0] < 0.15:
                kpts[0] = shoulder_center + np.array([0, -torso_height * 0.25])
                confs[0] = 0.3

            for eye_idx in [1, 2]:
                if confs[eye_idx] < 0.15:
                    kpts[eye_idx] = kpts[0] + np.array([-10 if eye_idx == 1 else 10, -5])
                    confs[eye_idx] = 0.25

            if confs[13] < 0.15 and confs[11] > 0.25:
                leg_direction = np.array([0, 1])
                kpts[13] = kpts[11] + leg_direction * torso_height * 0.7
                confs[13] = 0.3

            if confs[14] < 0.15 and confs[12] > 0.25:
                leg_direction = np.array([0, 1])
                kpts[14] = kpts[12] + leg_direction * torso_height * 0.7
                confs[14] = 0.3

            if confs[13] > 0.25 and confs[15] < 0.15:
                thigh_vec = kpts[13] - kpts[11]
                kpts[15] = kpts[13] + thigh_vec * 0.8
                confs[15] = 0.3

            if confs[14] > 0.25 and confs[16] < 0.15:
                thigh_vec = kpts[14] - kpts[12]
                kpts[16] = kpts[14] + thigh_vec * 0.8
                confs[16] = 0.3

        return kpts, confs

    def chain_interpolation(self, keypoints, confidences, image_shape=None):
        kpts = keypoints.copy()
        confs = confidences.copy()

        if image_shape is not None:
            h, w = image_shape[:2]
        else:
            h, w = 1000, 1000

        invalid_before = sum(1 for i in range(17) if confs[i] < 0.1 or
                             kpts[i][0] < 0 or kpts[i][1] < 0 or
                             kpts[i][0] > w * 1.5 or kpts[i][1] > h * 1.5)

        for i in range(17):
            if (confs[i] < 0.1 or
                    kpts[i][0] < 0 or kpts[i][1] < 0 or
                    kpts[i][0] > w * 1.5 or kpts[i][1] > h * 1.5):
                kpts[i] = np.array([-1000, -1000], dtype=np.float32)
                confs[i] = 0.0

        if not self.has_valid_chain("left_arm", confs) and self.has_valid_chain("right_arm", confs):
            kpts, confs = self.mirror_chain(kpts, confs, "right_arm", "left_arm")

        if not self.has_valid_chain("right_arm", confs) and self.has_valid_chain("left_arm", confs):
            kpts, confs = self.mirror_chain(kpts, confs, "left_arm", "right_arm")

        if not self.has_valid_chain("left_leg", confs) and self.has_valid_chain("right_leg", confs):
            kpts, confs = self.mirror_chain(kpts, confs, "right_leg", "left_leg")

        if not self.has_valid_chain("right_leg", confs) and self.has_valid_chain("left_leg", confs):
            kpts, confs = self.mirror_chain(kpts, confs, "left_leg", "right_leg")

        torso_points = BODY_CHAINS["torso_center"]
        torso_indices = [i for i in torso_points if confs[i] > 0.15]

        if len(torso_indices) >= 2:
            valid_torso_kpts = [kpts[i] for i in torso_indices]
            torso_center = np.mean(valid_torso_kpts, axis=0)

            for i in range(17):
                if confs[i] < 0.1 and np.all(kpts[i] < 0):
                    if i in [0, 1, 2, 3, 4]:
                        kpts[i] = torso_center + np.array([0, -50])
                        confs[i] = 0.2
                    elif i in [5, 7, 9]:
                        kpts[i] = torso_center + np.array([-30, 0])
                        confs[i] = 0.2
                    elif i in [6, 8, 10]:
                        kpts[i] = torso_center + np.array([30, 0])
                        confs[i] = 0.2
                    elif i in [11, 13, 15]:
                        kpts[i] = torso_center + np.array([-15, 30])
                        confs[i] = 0.2
                    elif i in [12, 14, 16]:
                        kpts[i] = torso_center + np.array([15, 30])
                        confs[i] = 0.2

        for chain_name, chain_indices in BODY_CHAINS.items():
            if chain_name == "shoulder_hip":
                continue

            valid_indices = []
            valid_kpts = []
            for idx in chain_indices:
                if confs[idx] > 0.1:
                    valid_indices.append(idx)
                    valid_kpts.append(kpts[idx])

            if len(valid_indices) >= 2:
                for i in range(len(chain_indices)):
                    idx = chain_indices[i]
                    if confs[idx] < 0.2:
                        prev_idx = None
                        for j in range(i - 1, -1, -1):
                            if confs[chain_indices[j]] > 0.1:
                                prev_idx = chain_indices[j]
                                break

                        next_idx = None
                        for j in range(i + 1, len(chain_indices)):
                            if confs[chain_indices[j]] > 0.1:
                                next_idx = chain_indices[j]
                                break

                        if prev_idx is not None and next_idx is not None:
                            prev_pos = kpts[prev_idx]
                            next_pos = kpts[next_idx]

                            total_dist = np.linalg.norm(next_pos - prev_pos)
                            if total_dist > 1:
                                prev_dist = np.linalg.norm(kpts[idx] - prev_pos) if confs[idx] > 0 else 0
                                if prev_dist == 0 or confs[idx] < 0.05:
                                    idx_in_chain = i
                                    prev_idx_in_chain = chain_indices.index(prev_idx)
                                    next_idx_in_chain = chain_indices.index(next_idx)
                                    ratio = (idx_in_chain - prev_idx_in_chain) / (next_idx_in_chain - prev_idx_in_chain)

                                    kpts[idx] = prev_pos + ratio * (next_pos - prev_pos)
                                    confs[idx] = max(confs[idx], 0.3)
                                else:
                                    ratio = prev_dist / total_dist
                                    projected = prev_pos + ratio * (next_pos - prev_pos)
                                    kpts[idx] = 0.2 * kpts[idx] + 0.8 * projected
                                    confs[idx] = max(confs[idx], 0.35)

        shoulder_indices = [5, 6]
        hip_indices = [11, 12]

        for left_idx, right_idx in [(5, 6), (11, 12)]:
            if confs[left_idx] > 0.25 and confs[right_idx] > 0.25:
                center = (kpts[left_idx] + kpts[right_idx]) / 2
                if confs[left_idx] > confs[right_idx]:
                    kpts[right_idx] = 2 * center - kpts[left_idx]
                    confs[right_idx] = max(confs[right_idx], 0.3)
                else:
                    kpts[left_idx] = 2 * center - kpts[right_idx]
                    confs[left_idx] = max(confs[left_idx], 0.3)

        kpts, confs = self.apply_body_proportions(kpts, confs)

        kpts, confs = self.body_model_completion(kpts, confs)

        invalid_after = sum(1 for i in range(17) if confs[i] < 0.1)
        self.debug_info.append({
            "frame": len(self.pose_history) if self.pose_history else 0,
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "improvement": invalid_before - invalid_after
        })

        return kpts, confs

    def apply_body_proportions(self, keypoints, confidences):
        kpts = keypoints.copy()
        confs = confidences.copy()

        visible_indices = [i for i in range(17) if confs[i] > 0.2]
        if len(visible_indices) >= 4:
            shoulder_width = 0
            if confs[5] > 0.25 and confs[6] > 0.25:
                shoulder_width = np.linalg.norm(kpts[5] - kpts[6])

            hip_width = 0
            if confs[11] > 0.25 and confs[12] > 0.25:
                hip_width = np.linalg.norm(kpts[11] - kpts[12])

            if shoulder_width > 10:
                arm_scale = 1.3

                if confs[5] > 0.25 and confs[7] > 0.25:
                    arm_length = np.linalg.norm(kpts[7] - kpts[5])
                    if arm_length > shoulder_width * 2.5 or arm_length < shoulder_width * 0.5:
                        direction = kpts[7] - kpts[5]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            kpts[7] = kpts[5] + direction * shoulder_width * arm_scale
                            confs[7] = max(confs[7], 0.3)

                if confs[6] > 0.25 and confs[8] > 0.25:
                    arm_length = np.linalg.norm(kpts[8] - kpts[6])
                    if arm_length > shoulder_width * 2.5 or arm_length < shoulder_width * 0.5:
                        direction = kpts[8] - kpts[6]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            kpts[8] = kpts[6] + direction * shoulder_width * arm_scale
                            confs[8] = max(confs[8], 0.3)

        return kpts, confs

    def temporal_smoothing(self, current_keypoints, current_confidences, image_shape=None):
        smoothed_kpts = current_keypoints.copy()
        smoothed_conf = current_confidences.copy()

        if image_shape is not None:
            h, w = image_shape[:2]
        else:
            h, w = 1000, 1000

        movement = 0
        valid_count = 0
        if len(self.pose_history) > 0:
            prev_kpts, prev_conf = self.pose_history[-1]
            for i in range(17):
                if current_confidences[i] > 0.2 and prev_conf[i] > 0.2:
                    movement += np.linalg.norm(current_keypoints[i] - prev_kpts[i])
                    valid_count += 1

        avg_movement = movement / max(valid_count, 1)

        if avg_movement > 50:
            process_noise = 0.05
        elif avg_movement > 20:
            process_noise = 0.01
        else:
            process_noise = 0.0001

        for kf in self.kalman_filters:
            kf.Q = process_noise * np.eye(4, dtype=np.float32)

        for i in range(17):
            measurement = current_keypoints[i]
            conf = current_confidences[i]

            is_valid_measurement = (conf > 0.15 and
                                    measurement[0] > 0 and measurement[1] > 0 and
                                    measurement[0] < w and measurement[1] < h)

            predicted = self.kalman_filters[i].predict()

            if is_valid_measurement:
                smoothed_kpts[i] = self.kalman_filters[i].update(measurement)
                smoothed_conf[i] = conf
            else:
                smoothed_kpts[i] = predicted
                smoothed_conf[i] = 0.25

                if len(self.pose_history) > 0 and prev_conf[i] > 0.2 and avg_movement < 40:
                    smoothed_kpts[i] = 0.1 * smoothed_kpts[i] + 0.9 * prev_kpts[i]
                    smoothed_conf[i] = prev_conf[i] * 0.8

                    self.kalman_filters[i].state[:2] = smoothed_kpts[i].reshape(2, 1)

        if np.mean(current_confidences) > 0.15:
            self.pose_history.append((smoothed_kpts.copy(), smoothed_conf.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

        return smoothed_kpts, smoothed_conf

    def detect_pose(self, image):
        global frame_count
        h, w = image.shape[:2]

        if frame_count % 2 == 0 or self.last_keypoints is None:
            results = model_yolo(image, imgsz=640, conf=0.15, iou=0.45, device=device, verbose=False)
            kpts, confs = self.process_yolo_results(results, image.shape)

            if kpts is not None:
                self.last_keypoints = kpts.copy()
                self.last_confidences = confs.copy()
        else:
            kpts = self.last_keypoints.copy() if self.last_keypoints is not None else None
            confs = self.last_confidences.copy() if self.last_confidences is not None else None

        if kpts is None:
            if len(self.pose_history) > 0:
                kpts, confs = self.pose_history[-1]
                confs = confs * 0.7
            else:
                return None, None

        kpts, confs = self.chain_interpolation(kpts, confs, image.shape)

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
        conf = keypoints_data.conf[best_idx].cpu().numpy() if keypoints_data.has_visible else np.ones(17) * 0.3

        h, w = image_shape[:2]
        kpts = np.full((17, 2), -1.0, dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)

        for i in range(17):
            if i < len(xy):
                x, y = xy[i]
                if 0 < x < w and 0 < y < h:
                    kpts[i] = [float(x), float(y)]
                    confs[i] = float(conf[i] if i < len(conf) else 0.0)
                else:
                    kpts[i] = [-1.0, -1.0]
                    confs[i] = 0.0
            else:
                kpts[i] = [-1.0, -1.0]
                confs[i] = 0.0

        return kpts, confs


class PoseHistory:
    def __init__(self, max_frames=10):
        self.max_frames = max_frames
        self.ankle_positions = []
        self.fall_flags = []
        self.fall_confirmed_frames = 0
        self.movement_history = []

    def add_ankle_positions(self, left_ankle, right_ankle):
        self.ankle_positions.append((left_ankle, right_ankle))
        if len(self.ankle_positions) > self.max_frames:
            self.ankle_positions.pop(0)

    def add_fall_flag(self, is_falling):
        self.fall_flags.append(1 if is_falling else 0)
        if len(self.fall_flags) > 5:
            self.fall_flags.pop(0)

        if is_falling:
            self.fall_confirmed_frames = min(self.fall_confirmed_frames + 1, 10)
        else:
            self.fall_confirmed_frames = max(self.fall_confirmed_frames - 1, 0)

    def get_movement_magnitude(self):
        if len(self.ankle_positions) < 2:
            return 0

        total = 0
        count = 0

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
        if len(self.fall_flags) < 3:
            return False

        return sum(self.fall_flags[-3:]) >= 2 or self.fall_confirmed_frames >= 3


def detect_walking(keypoints, history, width, height):
    if len(keypoints) < 17:
        return False

    leg_indices = [13, 14, 15, 16]
    valid_leg_points = sum(1 for i in leg_indices if keypoints[i][0] > 0 and keypoints[i][1] > 0)

    if valid_leg_points < 2:
        return False

    left_ankle = (keypoints[15][0], keypoints[15][1]) if keypoints[15][0] > 0 else None
    right_ankle = (keypoints[16][0], keypoints[16][1]) if keypoints[16][0] > 0 else None

    history.add_ankle_positions(left_ankle, right_ankle)

    if len(history.ankle_positions) < 3:
        return False

    movement = history.get_movement_magnitude()
    return movement > 3.0 and movement < 50.0


def detect_falling(keypoints, width, height):
    try:
        required_indices = [5, 6, 11, 12]
        for i in required_indices:
            if not (0 < keypoints[i][0] < width and 0 < keypoints[i][1] < height):
                return False, 0, False, 0.0

        chest_x = (keypoints[5][0] + keypoints[6][0]) / 2
        chest_y = (keypoints[5][1] + keypoints[6][1]) / 2
        waist_x = (keypoints[11][0] + keypoints[12][0]) / 2
        waist_y = (keypoints[11][1] + keypoints[12][1]) / 2

        v_up = np.array([0, -1])
        v_body = np.array([waist_x - chest_x, waist_y - chest_y])

        if np.linalg.norm(v_body) > 0:
            v_body = v_body / np.linalg.norm(v_body)

        angle = np.degrees(np.arccos(np.clip(np.dot(v_up, v_body), -1.0, 1.0)))

        torso_points = [keypoints[i] for i in [5, 6, 11, 12] if 0 < keypoints[i][0] < width]
        if len(torso_points) >= 2:
            xs = [p[0] for p in torso_points]
            ys = [p[1] for p in torso_points]
            width_body = max(xs) - min(xs)
            height_body = max(ys) - min(ys)
            aspect = width_body / height_body if height_body > 0 else 0
        else:
            aspect = 0

        angle_condition = angle > 45
        vertical_condition = chest_y > waist_y + 10
        aspect_condition = aspect > 0.8

        is_fall_candidate = sum([angle_condition, vertical_condition, aspect_condition]) >= 2

        return is_fall_candidate, angle, vertical_condition, aspect

    except Exception as e:
        print(f"跌倒检测错误: {e}")
        return False, 0, False, 0.0


def draw_keypoints_and_status(frame, keypoints, confidences, status, fall_details=None):
    """绘制关键点和状态 - 修改为显示中文"""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # 绘制骨骼
    for (i, j) in skeleton:
        if keypoints[i][0] > 0 and keypoints[j][0] > 0:
            conf = (confidences[i] + confidences[j]) / 2
            color_intensity = int(255 * conf)
            cv2.line(overlay,
                     (int(keypoints[i][0]), int(keypoints[i][1])),
                     (int(keypoints[j][0]), int(keypoints[j][1])),
                     (0, color_intensity, 0), 2)

    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            conf = confidences[i]
            if conf > 0.7:
                color = (0, 0, 255)
            elif conf > 0.4:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)

            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
            cv2.putText(overlay, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 状态映射到中文
    status_map = {
        "Falling": "跌倒",
        "Walking": "行走",
        "Standing": "站立",
        "Not detected": "未检测到"
    }

    status_chinese = status_map.get(status, status)

    # 状态颜色
    status_colors = {
        "跌倒": (0, 0, 255),  # 红色
        "行走": (255, 0, 0),  # 蓝色
        "站立": (0, 255, 0),  # 绿色
        "未检测到": (128, 128, 128)  # 灰色
    }

    color = status_colors.get(status_chinese, (255, 255, 255))

    # 使用PIL绘制中文字符
    # 将OpenCV图像转换为PIL图像
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        # Windows系统常见中文字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows黑体
            "C:/Windows/Fonts/msyh.ttc",  # Windows微软雅黑
            "/System/Library/Fonts/PingFang.ttc",  # MacOS苹方
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Linux文泉驿
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 30)
                    break
                except:
                    continue

        if font is None:
            # 如果找不到中文字体，使用默认字体（可能不支持中文）
            font = ImageFont.load_default()
            print("警告: 未找到中文字体，使用默认字体")
    except:
        font = ImageFont.load_default()
        print("警告: 字体加载失败，使用默认字体")

    # 绘制状态文本
    status_text = f"状态: {status_chinese}"
    draw.text((10, 10), status_text, font=font, fill=(color[2], color[1], color[0]))

    # 绘制详细信息
    y_offset = 50
    if status == "Falling" and fall_details:
        angle, vertical_cond, aspect = fall_details

        # # 中文详细信息
        # details_texts = [
        #     f"角度: {angle:.1f}度",
        #     f"胸部低于腰部: {'是' if vertical_cond else '否'}",
        #     f"宽高比: {aspect:.2f}"
        # ]
        #
        # for text in details_texts:
        #     draw.text((10, y_offset), text, font=font, fill=(color[2], color[1], color[0]))
        #     y_offset += 25

    # 显示可见关键点数量（中文）
    visible_count = sum(1 for x, y in keypoints if x > 0 and y > 0)
    keypoints_text = f"关键点: {visible_count}/17"

    # 使用小一点的字体
    try:
        small_font = ImageFont.truetype(font_paths[0] if os.path.exists(font_paths[0]) else None, 20)
    except:
        small_font = font

    draw.text((10, h - 40), keypoints_text, font=small_font, fill=(color[2], color[1], color[0]))

    # 将PIL图像转换回OpenCV图像
    overlay = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return overlay


def process_video(input_path, output_path='results/enhanced_fall_detection.mp4'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    detector = EnhancedPoseDetector()
    history = PoseHistory()

    global frame_count
    frame_count = 0

    print("开始处理视频...")

    total_frames = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kpts, confs = detector.detect_pose(frame)

        if kpts is None:
            kpts = np.full((17, 2), -1.0)
            confs = np.zeros(17)
            status = "Not detected"
            details = None
        else:
            detected_frames += 1
            is_fall_candidate, angle, vertical_cond, aspect = detect_falling(kpts, w, h)
            history.add_fall_flag(is_fall_candidate)

            if history.is_fall_confirmed():
                status = "Falling"
                details = (angle, vertical_cond, aspect)
            else:
                if detect_walking(kpts, history, w, h):
                    status = "Walking"
                else:
                    status = "Standing"
                details = None

        frame = draw_keypoints_and_status(frame, kpts, confs, status, details)
        out.write(frame)

        cv2.imshow('老年人跌倒检测', frame)

        frame_count += 1
        total_frames += 1

        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
            if detector.debug_info:
                latest = detector.debug_info[-1]
                if latest["improvement"] > 0:
                    print(f"  补全效果: {latest['invalid_before']} -> {latest['invalid_after']} 无效点")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if detector.debug_info:
        print(f"\n补全效果统计:")
        improvements = [info["improvement"] for info in detector.debug_info]
        avg_improvement = sum(improvements) / len(improvements)
        print(f"平均每帧补全 {avg_improvement:.1f} 个关键点")

    print(f"\n处理完成！")
    print(f"总帧数: {total_frames}")
    print(f"检测到姿态的帧数: {detected_frames}")
    print(f"检测率: {detected_frames / total_frames * 100:.1f}%")
    print(f"输出保存至: {output_path}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = 'video/fall_hhl.mp4'
    output_video = 'results/fall_hhl_result.mp4'

    if not os.path.exists(input_video):
        print(f"请放入测试视频: {input_video}")
    else:
        process_video(input_video, output_video)