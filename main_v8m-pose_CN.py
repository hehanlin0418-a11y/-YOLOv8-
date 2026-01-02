import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont  # 添加PIL库用于中文显示

model = YOLO('yolov8m-pose.pt')

# 关键点索引 (COCO格式)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# 骨架连接
skeleton = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 上半身
    (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),  # 躯干和大腿
    (13, 15), (14, 16)  # 小腿
]


# 存储历史关键点用于运动分析
class PoseHistory:
    def __init__(self, max_frames=10):
        self.max_frames = max_frames
        self.ankle_positions = []  # 存储脚踝位置历史
        self.fall_flags = []  # 存储跌倒状态历史
        self.fall_confirmed_frames = 0  # 跌倒确认持续帧数

    def add_ankle_positions(self, left_ankle, right_ankle):
        """添加脚踝位置到历史记录"""
        self.ankle_positions.append((left_ankle, right_ankle))
        if len(self.ankle_positions) > self.max_frames:
            self.ankle_positions.pop(0)

    def add_fall_flag(self, is_falling):
        """添加跌倒标记到历史记录"""
        self.fall_flags.append(1 if is_falling else 0)
        if len(self.fall_flags) > 5:  # 保留最近5帧用于更稳定的判断
            self.fall_flags.pop(0)

        # 更新跌倒确认帧数
        if is_falling:
            self.fall_confirmed_frames = min(self.fall_confirmed_frames + 1, 10)
        else:
            self.fall_confirmed_frames = max(self.fall_confirmed_frames - 2, 0)

    def get_movement_magnitude(self):
        """计算脚踝移动幅度"""
        if len(self.ankle_positions) < 2:
            return 0

        total_movement = 0
        valid_pairs = 0

        for i in range(1, len(self.ankle_positions)):
            # 左踝移动
            left_diff = 0
            if (self.ankle_positions[i][0] and self.ankle_positions[i - 1][0] and
                    self.ankle_positions[i][0] != (None, None) and self.ankle_positions[i - 1][0] != (None, None)):
                left_diff = np.linalg.norm(
                    np.array(self.ankle_positions[i][0]) -
                    np.array(self.ankle_positions[i - 1][0])
                )
                valid_pairs += 1

            # 右踝移动
            right_diff = 0
            if (self.ankle_positions[i][1] and self.ankle_positions[i - 1][1] and
                    self.ankle_positions[i][1] != (None, None) and self.ankle_positions[i - 1][1] != (None, None)):
                right_diff = np.linalg.norm(
                    np.array(self.ankle_positions[i][1]) -
                    np.array(self.ankle_positions[i - 1][1])
                )
                valid_pairs += 1

            total_movement += (left_diff + right_diff)

        if valid_pairs == 0:
            return 0
        return total_movement / valid_pairs

    def is_fall_confirmed(self):
        """验证跌倒状态：使用累积确认机制"""
        if len(self.fall_flags) < 3:
            return False

        # 方法1：最近5帧中至少3帧为跌倒
        recent_fall = sum(self.fall_flags[-3:]) >= 2

        # 方法2：累积确认帧数达到阈值
        accumulated_fall = self.fall_confirmed_frames >= 3

        return recent_fall or accumulated_fall


def valid_keypoint(x, y, width, height):
    """检查关键点是否有效"""
    return x is not None and y is not None and 0 <= x <= width and 0 <= y <= height


def calculate_angle(v1, v2):
    """计算两个向量的夹角（度）"""
    try:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0

        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    except Exception as e:
        return 0


def detect_walking(keypoints, history, width, height):
    """检测行走状态"""
    # 检查腿部关键点
    required_leg_points = [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    valid_leg_points = 0

    for idx in required_leg_points:
        if idx < len(keypoints):
            x, y = keypoints[idx]
            if valid_keypoint(x, y, width, height):
                valid_leg_points += 1

    # 至少需要3个有效的腿部关键点
    if valid_leg_points < 3:
        return False

    # 获取脚踝位置
    left_ankle = (keypoints[LEFT_ANKLE][0], keypoints[LEFT_ANKLE][1]) if valid_keypoint(keypoints[LEFT_ANKLE][0],
                                                                                        keypoints[LEFT_ANKLE][1], width,
                                                                                        height) else (None, None)
    right_ankle = (keypoints[RIGHT_ANKLE][0], keypoints[RIGHT_ANKLE][1]) if valid_keypoint(keypoints[RIGHT_ANKLE][0],
                                                                                           keypoints[RIGHT_ANKLE][1],
                                                                                           width, height) else (
        None, None)

    # 添加到历史记录
    history.add_ankle_positions(left_ankle, right_ankle)

    # 计算平均移动幅度
    if len(history.ankle_positions) < 3:
        return False

    movement = history.get_movement_magnitude()

    # 行走阈值
    walking_threshold = 3.0
    return movement > walking_threshold


def detect_falling(keypoints, width, height):
    """检测跌倒状态"""
    try:
        # 检查必要的上半身关键点
        required_points = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        for idx in required_points:
            if idx >= len(keypoints):
                return False, 0, False, 0.0
            x, y = keypoints[idx]
            if not valid_keypoint(x, y, width, height):
                return False, 0, False, 0.0

        # 角度计算：胸部-腰部连线与垂直方向夹角
        chest_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
        chest_y = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        waist_x = (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2
        waist_y = (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2

        v1 = np.array([0, -1])  # 垂直方向向量（向下为正）
        v2 = np.array([waist_x - chest_x, waist_y - chest_y])  # 胸→腰向量

        # 归一化向量
        if np.linalg.norm(v2) > 0:
            v2 = v2 / np.linalg.norm(v2)

        angle = calculate_angle(v1, v2)
        angle_condition = angle > 45  # 倾斜角度超45度判定为候选

        # 胸部和腰部Y坐标比较
        y_comparison = chest_y > waist_y  # 胸部Y坐标大于腰部，说明上下颠倒

        # 宽高比计算：肩膀和胯部构成的矩形宽高比
        relevant_points = [keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER],
                           keypoints[LEFT_HIP], keypoints[RIGHT_HIP]]

        # 筛选有效关键点
        valid_relevant = [p for p in relevant_points if valid_keypoint(p[0], p[1], width, height)]
        if len(valid_relevant) >= 3:
            xs = [p[0] for p in valid_relevant]
            ys = [p[1] for p in valid_relevant]
            width_rect = max(xs) - min(xs)
            height_rect = max(ys) - min(ys)
            aspect_ratio = width_rect / height_rect if height_rect != 0 else 0
            aspect_condition = aspect_ratio > 0.8  # 宽高比超0.8判定为候选
        else:
            aspect_condition = False
            aspect_ratio = 0.0

        # 综合判断：3个条件满足2个以上则为跌倒候选
        fall_conditions = sum([angle_condition, y_comparison, aspect_condition])
        is_fall_candidate = fall_conditions >= 2

        return is_fall_candidate, angle, y_comparison, aspect_ratio

    except Exception as e:
        print(f"跌倒检测错误: {e}")
        return False, 0, False, 0.0


def process_results(results, width, height):
    """处理关键点结果"""
    try:
        if len(results) == 0 or not hasattr(results[0], 'keypoints') or len(results[0].keypoints) == 0:
            return None

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        if len(keypoints) != 17:
            return None

        # 过滤无效关键点
        valid_points = []
        for (x, y) in keypoints:
            if valid_keypoint(x, y, width, height):
                valid_points.append((x, y))
            else:
                valid_points.append((None, None))
        return np.array(valid_points)
    except Exception as e:
        print(f"关键点处理错误: {e}")
        return None


def draw_keypoints_and_status(frame, keypoints, status, fall_details=None):
    """绘制关键点、骨架和状态信息 - 修改为显示中文"""
    try:
        frame_copy = frame.copy()
        height, width = frame.shape[:2]

        # 绘制骨架
        for (i, j) in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
                if valid_keypoint(x1, y1, width, height) and valid_keypoint(x2, y2, width, height):
                    cv2.line(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 绘制关键点 - 所有有效关键点都用红色
        for (x, y) in keypoints:
            if valid_keypoint(x, y, width, height):
                cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 状态映射到中文
        status_map = {
            "Falling": "跌倒",
            "Walking": "行走",
            "Standing": "站立",
            "Not detected": "未检测到",
            "Insufficient points": "关键点不足"
        }

        status_chinese = status_map.get(status, status)

        # 状态颜色
        if status_chinese == "跌倒":
            color = (0, 0, 255)  # 红色：跌倒
        elif status_chinese == "行走":
            color = (255, 0, 0)  # 蓝色：行走
        elif status_chinese == "站立":
            color = (0, 255, 0)  # 绿色：站立
        else:
            color = (128, 128, 128)  # 灰色：未检测到

        # 使用PIL绘制中文字符
        # 将OpenCV图像转换为PIL图像
        overlay_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(overlay_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体
        try:
            # 常见中文字体路径
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
                # 如果找不到中文字体，使用默认字体
                font = ImageFont.load_default()
                print("警告: 未找到中文字体，使用默认字体")
        except:
            font = ImageFont.load_default()
            print("警告: 字体加载失败，使用默认字体")

        # 绘制状态文本
        status_text = f"状态: {status_chinese}"
        draw.text((10, 10), status_text, font=font, fill=(color[2], color[1], color[0]))

        # 绘制跌倒判断详情（中文）
        # if fall_details and status == "Falling":
        #     angle, y_comp, aspect = fall_details
        #
        #     # 创建详情文本
        #     detail_texts = [
        #         f"角度: {angle:.1f}度",
        #         f"胸部低于腰部: {'是' if y_comp else '否'}",
        #         f"宽高比: {aspect:.2f}"
        #     ]
        #
        #     y_offset = 50
        #     for text in detail_texts:
        #         draw.text((10, y_offset), text, font=font, fill=(color[2], color[1], color[0]))
        #         y_offset += 35

        # 将PIL图像转换回OpenCV图像
        frame_copy = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame_copy
    except Exception as e:
        print(f"绘制错误: {e}")
        return frame


def process_video(input_path, output_path='results/posture_detection.mp4'):
    """处理视频并输出结果"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {input_path}")
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 初始化历史记录器
        pose_history = PoseHistory(max_frames=10)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 模型推理
            results = model(
                frame,
                imgsz=640,
                conf=0.5,
                device='0'  # 使用CPU，如需GPU改为'0'
            )

            # 处理关键点
            keypoints = process_results(results, width, height)
            status = "Not detected"
            fall_details = None

            if keypoints is not None:
                # 1. 最高优先级：检测跌倒
                is_fall_candidate, angle, y_comp, aspect = detect_falling(keypoints, width, height)
                pose_history.add_fall_flag(is_fall_candidate)

                # 跌倒确认机制
                if pose_history.is_fall_confirmed():
                    status = "Falling"
                    fall_details = (angle, y_comp, aspect)
                else:
                    # 2. 非跌倒状态下的判断
                    # 检查是否有足够的关键点进行状态判断
                    valid_points_count = sum(1 for (x, y) in keypoints if valid_keypoint(x, y, width, height))

                    if valid_points_count >= 6:  # 至少有6个有效关键点
                        # 检测行走
                        if detect_walking(keypoints, pose_history, width, height):
                            status = "Walking"
                        else:
                            status = "Standing"
                    else:
                        status = "Insufficient points"

            # 绘制结果
            frame = draw_keypoints_and_status(frame, keypoints, status, fall_details)

            # 写入输出视频
            out.write(frame)
            frame_count += 1

            # 显示窗口（按'q'退出）
            cv2.imshow('人体姿态检测', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        print(f"处理完成. 总帧数: {frame_count}，输出路径: {output_path}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"视频处理错误: {e}")
        try:
            cap.release()
            out.release()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = 'video/fall_hhl.mp4'  # 输入视频路径
    output_video = 'results/test_result.mp4'  # 输出视频路径
    process_video(input_video, output_video)