import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch

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

skeleton = [
    (5,6),(5,7),(6,8),(7,9),(8,10),
    (5,11),(6,12),(11,12),(11,13),(12,14),
    (13,15),(14,16)
]

class EnhancedPoseDetector:
    def __init__(self):
        self.pose_history = []
        self.max_history = 5
        self.last_keypoints = None
        self.last_confidences = None

    def interpolate_missing_keypoints(self, keypoints, confidences):
        usable = np.where(confidences > 0.1)[0]
        if len(usable) < 3:
            return keypoints, confidences

        def fill_chain(chain):
            valid = [i for i in chain if confidences[i] > 0.1]
            if not valid:
                return
            for miss in set(chain) - set(valid):
                dists = sorted([(abs(miss - v), v) for v in valid])
                if len(dists) < 2:
                    continue
                (_, i1), (_, i2) = dists[:2]
                p1, p2 = keypoints[i1], keypoints[i2]
                if p1[0] <= 0 or p2[0] <= 0:
                    continue
                ratio = abs(miss - i1) / (abs(i2 - i1) + 1e-6)
                nx = p1[0] + (p2[0] - p1[0]) * ratio
                ny = p1[1] + (p2[1] - p1[1]) * ratio
                if 0 < nx < 2000 and 0 < ny < 2000:
                    keypoints[miss] = [nx, ny]
                    confidences[miss] = max(confidences[miss], 0.25)

        fill_chain([0,1,2,3,4,5,6,7,8,9,10])  # 上身
        fill_chain([5,11,13,15])               # 左腿
        fill_chain([6,12,14,16])               # 右腿
        return keypoints, confidences

    def temporal_smoothing(self, current_keypoints, current_confidences):
        smoothed_kpts = current_keypoints.copy()
        smoothed_conf = current_confidences.copy()

        if len(self.pose_history) == 0:
            self.pose_history.append((current_keypoints.copy(), current_confidences.copy()))
            return smoothed_kpts, smoothed_conf

        prev_kpts, prev_conf = self.pose_history[-1]
        valid_ratio = np.mean(current_confidences > 0.15)

        if valid_ratio < 0.4:  # 严重遮挡，强拉历史
            for i in range(17):
                if current_confidences[i] < 0.2:
                    if prev_conf[i] > 0.3:
                        smoothed_kpts[i] = prev_kpts[i] * 0.85 + current_keypoints[i] * 0.15
                        smoothed_conf[i] = 0.38
                    elif prev_conf[i] > 0.1:
                        smoothed_kpts[i] = prev_kpts[i] * 0.7 + current_keypoints[i] * 0.3
                        smoothed_conf[i] = 0.25
        else:
            alpha = 0.6
            for i in range(17):
                if current_confidences[i] > 0.1:
                    smoothed_kpts[i] = alpha * current_keypoints[i] + (1 - alpha) * prev_kpts[i]

        if valid_ratio > 0.25:
            self.pose_history.append((smoothed_kpts.copy(), smoothed_conf.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

        return smoothed_kpts, smoothed_conf

    def detect_pose(self, image):
        global frame_count
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
                confs = confs * 0.8
            else:
                return None, None

        kpts, confs = self.interpolate_missing_keypoints(kpts, confs)
        kpts, confs = self.temporal_smoothing(kpts, confs)
        return kpts, confs

    def process_yolo_results(self, results, image_shape):
        if not results or len(results[0].keypoints) == 0:
            return None, None

        boxes = results[0].boxes
        max_area, idx = 0, 0
        for i, box in enumerate(boxes):
            area = box.xywh[0][2].item() * box.xywh[0][3].item()
            if area > max_area:
                max_area = area
                idx = i

        data = results[0].keypoints
        xy = data.xy[idx].cpu().numpy()
        conf = data.conf[idx].cpu().numpy() if data.has_visible else np.ones(17) * 0.3

        kpts = np.full((17, 2), -1.0, dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)
        for i in range(17):
            x, y = xy[i]
            if x > 0 and y > 0 and x < image_shape[1] and y < image_shape[0]:
                kpts[i] = [float(x), float(y)]
                confs[i] = float(conf[i])
        return kpts, confs


class PoseHistory:
    def __init__(self, max_frames=10):
        self.max_frames = max_frames
        self.ankle_positions = []   # 存 tuple: ((x,y), (x,y)) 或 (None, None)
        self.fall_flags = []
        self.fall_confirmed_frames = 0

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
            self.fall_confirmed_frames = max(self.fall_confirmed_frames - 2, 0)

    def get_movement_magnitude(self):
        if len(self.ankle_positions) < 2:
            return 0
        total = 0
        count = 0
        for i in range(1, len(self.ankle_positions)):
            prev_l, prev_r = self.ankle_positions[i-1]
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
    valid = sum(1 for i in [13,14,15,16] if keypoints[i][0] > 0 and keypoints[i][1] > 0)
    if valid < 3:
        return False

    left = (keypoints[15][0], keypoints[15][1]) if keypoints[15][0] > 0 else None
    right = (keypoints[16][0], keypoints[16][1]) if keypoints[16][0] > 0 else None
    history.add_ankle_positions(left, right)

    if len(history.ankle_positions) < 3:
        return False
    return history.get_movement_magnitude() > 3.0


def detect_falling(keypoints, width, height):
    try:
        for i in [5,6,11,12]:
            if not (0 < keypoints[i][0] < width and 0 < keypoints[i][1] < height):
                return False, 0, False, 0.0

        chest_x = (keypoints[5][0] + keypoints[6][0]) / 2
        chest_y = (keypoints[5][1] + keypoints[6][1]) / 2
        waist_x = (keypoints[11][0] + keypoints[12][0]) / 2
        waist_y = (keypoints[11][1] + keypoints[12][1]) / 2

        v_up = np.array([0, -1])
        v_body = np.array([waist_x - chest_x, waist_y - chest_y])
        if np.linalg.norm(v_body) > 0:
            v_body /= np.linalg.norm(v_body)
        angle = np.degrees(np.arccos(np.clip(np.dot(v_up, v_body), -1.0, 1.0)))

        pts = [keypoints[i] for i in [5,6,11,12] if 0 < keypoints[i][0] < width]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        aspect = (max(xs) - min(xs)) / (max(ys) - min(ys)) if len(ys) >= 2 and max(ys) > min(ys) else 0

        angle_cond = angle > 45
        y_cond = chest_y > waist_y
        aspect_cond = aspect > 0.8

        candidate = sum([angle_cond, y_cond, aspect_cond]) >= 2
        return candidate, angle, y_cond, aspect
    except:
        return False, 0, False, 0.0


def draw_keypoints_and_status(frame, keypoints, confidences, status, fall_details=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for (i, j) in skeleton:
        if (keypoints[i][0] > 0 and keypoints[j][0] > 0):
            conf = (confidences[i] + confidences[j]) / 2
            cv2.line(overlay, (int(keypoints[i][0]), int(keypoints[i][1])),
                            (int(keypoints[j][0]), int(keypoints[j][1])),
                            (0, int(255 * conf), 0), 2)

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            c = confidences[i]
            color = (0,0,255) if c > 0.7 else (0,165,255) if c > 0.4 else (0,255,255)
            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)

    color = (0,0,255) if status == "Falling" else (255,0,0) if status == "Walking" else (0,255,0) if status == "Standing" else (128,128,128)
    cv2.putText(overlay, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    y = 60
    if status == "Falling" and fall_details:
        angle, ycomp, aspect = fall_details
        cv2.putText(overlay, f"Angle: {angle:.1f}deg", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1); y += 20
        cv2.putText(overlay, f"Chest>Waist: {ycomp}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1); y += 20
        cv2.putText(overlay, f"Aspect: {aspect:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    valid_cnt = sum(1 for x,y in keypoints if x > 0 and y > 0)
    cv2.putText(overlay, f"Keypoints: {valid_cnt}/17", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kpts, confs = detector.detect_pose(frame)
        if kpts is None:
            kpts = np.full((17,2), -1.0)
            confs = np.zeros(17)

        status = "Not detected"
        details = None

        candidate, angle, ycomp, aspect = detect_falling(kpts, w, h)
        history.add_fall_flag(candidate)

        if history.is_fall_confirmed():
            status = "Falling"
            details = (angle, ycomp, aspect)
        else:
            if detect_walking(kpts, history, w, h):
                status = "Walking"
            else:
                status = "Standing"

        frame = draw_keypoints_and_status(frame, kpts, confs, status, details)
        out.write(frame)
        cv2.imshow('老年人跌倒检测', frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

        if cv2.waitKey(1) == ord('q'):
            break

    print(f"处理完成！输出保存至: {output_path}")
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