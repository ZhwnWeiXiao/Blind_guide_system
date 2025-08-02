import os
import cv2
import torch
import numpy as np
import time
import pyttsx3
import threading
from PIL import Image, ImageDraw, ImageFont

# Ensure sort.py is in the same directory
from sort import Sort
from ultralytics import YOLO

# Temporal Transformer for sequence modeling
from temporal_transformer import TemporalTransformer

# Initialize transformer (moved to device inside main)
transformer = TemporalTransformer()

# Global variables for the speech engine and lock
_engine = None
_lock = threading.Lock() # Lock for speech synthesis to ensure only one voice plays at a time

def init_tts_engine():
    """Initializes the speech engine and sets a Chinese voice."""
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        voices = _engine.getProperty('voices')
        chinese_voice_found = False
        for voice in voices:
            if "taiwan" in voice.name.lower() or "zh-tw" in voice.id.lower():
                _engine.setProperty('voice', voice.id)
                chinese_voice_found = True
                print(f"Set Taiwan Chinese voice: {voice.name} (ID: {voice.id})")
                break
            elif "chinese" in voice.name.lower() or "zh" in voice.id.lower():
                _engine.setProperty('voice', voice.id)
                chinese_voice_found = True
                print(f"Set Chinese voice: {voice.name} (ID: {voice.id})")
                break
        if not chinese_voice_found:
            print("Warning: No Chinese voice found. Voice prompts may not work correctly. Please check if your system has a Chinese language pack installed.")
            if voices:
                _engine.setProperty('voice', voices[0].id)
                print(f"Falling back to default voice: {voices[0].name}")

        _engine.setProperty('rate', 180)
        _engine.setProperty('volume', 0.9)
    return _engine

def speak_async(message):
    """Executes speech synthesis in a separate thread without blocking the main program."""
    global _engine, _lock
    if _engine is None:
        _engine = init_tts_engine()

    print(f"DEBUG: speak_async called with message: '{message}'")

    def _speak():
        with _lock:
            _engine.say(message)
            _engine.runAndWait()
            print(f"DEBUG: _engine.runAndWait() finished for message: '{message}'")

    speaker_thread = threading.Thread(target=_speak)
    speaker_thread.daemon = True
    speaker_thread.start()

def yolov12_detect(model, img, conf_threshold, iou_threshold, target_classes=None, imgsz=640):
    """
    Detects objects using the YOLOv12 model.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, imgsz=img_rgb.shape[1] if imgsz == 'auto' else imgsz, conf=conf_threshold, iou=iou_threshold, classes=target_classes, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, confidence, class_id])
    if detections:
        return np.array(detections, dtype=np.float32)
    else:
        return np.empty((0, 6), dtype=np.float32)

def load_yolov12_model(weights_path):
    """
    Loads the YOLOv12 model.
    """
    print(f"Loading YOLOv12 model: {weights_path}...")
    try:
        model = YOLO(weights_path)
        print("YOLOv12 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error: Could not load YOLOv12 model. Check weights path or if the file is corrupted.\nError message: {e}")
        return None

def midas_value_to_distance(midas_value, K, C, max_valid_midas_value, min_valid_midas_value, return_raw=False):
    """
    Converts raw MiDaS depth values to approximate distance (meters).
    """
    if return_raw:
        return midas_value

    clamped_midas_value = np.clip(midas_value, min_valid_midas_value, max_valid_midas_value)

    if clamped_midas_value <= C:
        return float('inf')

    denominator = clamped_midas_value - C
    distance = K / denominator
    return distance


def normalize_brightness(img_rgb):
    """Apply CLAHE to the V channel of an RGB image to reduce lighting variance."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# --- Main Function (now accepts pre-loaded models and device) ---
def main(video_path, yolo_model, midas_model, midas_transform, device, output_video_path=None,
           conf_threshold=0.4,
           iou_threshold=0.5,
           yolo_imgsz=640,
           sort_max_age=5,
           sort_min_hits=3,
           sort_iou_threshold=0.3,
           target_classes=None,
           detection_interval=1,
           midas_output_scale_factor=1.0,
           # MiDaS depth value to distance conversion parameters
           MIDAS_K_CONVERT=1600,
           MIDAS_C_CONVERT=393,
           MIDAS_MAX_VALID_VALUE=3000,
           MIDAS_MIN_VALID_VALUE=460,
           display_raw_midas_value=False,
           # Obstacle Alert Parameter
           OBSTACLE_ALERT_DISTANCE=5.0,
           # New: Percentage of frame width considered as the "main path" for alerts
           MAIN_PATH_CENTER_WIDTH_PERCENTAGE=0.4, # e.g., 0.4 for middle 40%
           # Danger Assessment Parameters (simplified, only distance weight will be used effectively)
           DANGER_MAX_DISTANCE=15.0,
           DANGER_MIN_SAFE_DISTANCE=2.0,
           WEIGHT_DISTANCE=100,
           WEIGHT_VELOCITY=0,
           WEIGHT_ACCELERATION=0,
           # Voice alert cooldown time (seconds)
           VOICE_ALERT_COOLDOWN_OBSTACLE=3.0,
           VOICE_ALERT_COOLDOWN_LIGHT=5.0,
           DANGER_RESET_COOLDOWN=2.0,
           # Screen display parameters
           screen_width=1920,
           screen_height=1080
           ):

    # Initialize SORT tracker (moved inside main as it's stateful per video)
    mot_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_threshold)

    last_alert_time = 0
    last_spoken_alert_message = ""
    last_effective_danger_level = "Safe"
    global transformer
    transformer = transformer.to(device)
    transformer.eval()
    FRAME_BUFFER_SIZE = 8
    frame_buffer = []
    fast_approach_flag = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    if fps_video == 0:
        fps_video = 30
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if original_width == 0 or original_height == 0:
        print("Warning: Video properties not read correctly. Defaulting to 640x480.")
        original_width = 640
        original_height = 480

    # Calculate main path boundaries for alerts
    main_path_left_bound = original_width * (0.5 - MAIN_PATH_CENTER_WIDTH_PERCENTAGE / 2)
    main_path_right_bound = original_width * (0.5 + MAIN_PATH_CENTER_WIDTH_PERCENTAGE / 2)
    print(f"Main path for alerts: X from {main_path_left_bound:.0f} to {main_path_right_bound:.0f} (total width {original_width})")


    out = None
    output_video_width = original_width
    output_video_height = original_height
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps_video, (output_video_width, output_video_height))

    print(f"Starting to process video: {video_path}...")
    frame_count = 0

    MODEL_CLASSES = ['bicycle', 'bus', 'car', 'dog', 'Pedestrian Green Light',
                     'Pedestrian Red Light', 'person', 'scooter',
                     'Traffic Light Green', 'Traffic Light Red', 'Traffic Light Yellow',
                     'transformer', 'truck']

    PEDESTRIAN_GREEN_LIGHT_ID = MODEL_CLASSES.index('Pedestrian Green Light') if 'Pedestrian Green Light' in MODEL_CLASSES else -1
    PEDESTRIAN_RED_LIGHT_ID = MODEL_CLASSES.index('Pedestrian Red Light') if 'Pedestrian Red Light' in MODEL_CLASSES else -1
    TRAFFIC_LIGHT_GREEN_ID = MODEL_CLASSES.index('Traffic Light Green') if 'Traffic Light Green' in MODEL_CLASSES else -1
    TRAFFIC_LIGHT_RED_ID = MODEL_CLASSES.index('Traffic Light Red') if 'Traffic Light Red' in MODEL_CLASSES else -1
    TRAFFIC_LIGHT_YELLOW_ID = MODEL_CLASSES.index('Traffic Light Yellow') if 'Traffic Light Yellow' in MODEL_CLASSES else -1
    
    STATIC_OBSTACLE_CLASSES = ['transformer', 'bicycle', 'scooter', 'car', 'truck', 'bus'] # Added truck and bus

    depth_map_cached = None
    output_display_depth_cached = None

    tracked_object_distance_history = {}
    DISTANCE_HISTORY_LENGTH = 8

    tracked_object_midas_history = {}
    MIDAS_HISTORY_LENGTH = 15
    MIDAS_TRIM_PERCENTAGE = 0.1

    CHINESE_FONT_PATH = 'C:/Windows/Fonts/simhei.ttf' # Default Windows Chinese font

    try:
        font_size = 90
        font_pil = ImageFont.truetype(CHINESE_FONT_PATH, font_size)
    except IOError:
        print(f"Warning: Could not load Chinese font from {CHINESE_FONT_PATH}. Chinese characters may not display correctly.")
        font_pil = ImageFont.load_default()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Maintain temporal frame buffer
        frame_tensor_buffer = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        frame_buffer.append(frame_tensor_buffer)
        if len(frame_buffer) > FRAME_BUFFER_SIZE:
            frame_buffer.pop(0)

        current_detections_for_sort = np.empty((0, 6), dtype=np.float32)

        if frame_count == 1 or (frame_count % detection_interval == 0):
            detections_yolo = yolov12_detect(yolo_model, frame, conf_threshold, iou_threshold, target_classes, yolo_imgsz)
            current_detections_for_sort = detections_yolo

            img_rgb_midas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb_midas = normalize_brightness(img_rgb_midas)

            input_batch_midas = midas_transform(img_rgb_midas).to(device)

            with torch.no_grad():
                prediction_midas = midas_model(input_batch_midas)

                target_height_midas = int(img_rgb_midas.shape[0] * midas_output_scale_factor)
                target_width_midas = int(img_rgb_midas.shape[1] * midas_output_scale_factor)

                prediction_midas = torch.nn.functional.interpolate(
                    prediction_midas.unsqueeze(1),
                    size=(target_height_midas, target_width_midas),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map_cached = prediction_midas.cpu().numpy()

            grayscale_depth_output = cv2.normalize(depth_map_cached, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            output_display_depth_cached = cv2.cvtColor(grayscale_depth_output, cv2.COLOR_GRAY2BGR)

        if len(frame_buffer) == FRAME_BUFFER_SIZE:
            with torch.no_grad():
                seq_tensor = torch.stack(frame_buffer).to(device)
                temporal_feats = transformer(seq_tensor)
                if temporal_feats.size(0) >= 2:
                    motion_score = (temporal_feats[-1] - temporal_feats[-2]).pow(2).mean().sqrt().item()
                    fast_approach_flag = motion_score > 0.5
                else:
                    fast_approach_flag = False
        else:
            fast_approach_flag = False

        tracks = mot_tracker.update(current_detections_for_sort)

        if output_display_depth_cached is not None:
            raw_midas_map = depth_map_cached
        else:
            raw_midas_map = np.zeros(frame.shape[:2], dtype=np.float32)

        all_detected_objects_info = [] # (class_name, distance, track_id, direction_text, is_static_obstacle, in_main_alert_path, fast_approach_flag)

        pedestrian_red_light_detected = False
        pedestrian_green_light_detected = False
        traffic_light_red_detected = False
        traffic_light_yellow_detected = False
        traffic_light_green_detected = False

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for track in tracks:
            if len(track) < 7: # Ensure class_id is present
                continue

            x1_trk, y1_trk, x2_trk, y2_trk = map(int, track[:4])
            track_id = int(track[4])
            class_id = int(track[5])

            class_name = "Unknown"
            if 0 <= class_id < len(MODEL_CLASSES):
                class_name = MODEL_CLASSES[class_id]
            
            is_static_obstacle = class_name in STATIC_OBSTACLE_CLASSES

            if class_id == PEDESTRIAN_RED_LIGHT_ID:
                pedestrian_red_light_detected = True
            elif class_id == PEDESTRIAN_GREEN_LIGHT_ID:
                pedestrian_green_light_detected = True
            elif class_id == TRAFFIC_LIGHT_RED_ID:
                traffic_light_red_detected = True
            elif class_id == TRAFFIC_LIGHT_YELLOW_ID:
                traffic_light_yellow_detected = True
            elif class_id == TRAFFIC_LIGHT_GREEN_ID:
                traffic_light_green_detected = True

            object_region_x1 = max(0, x1_trk)
            object_region_y1 = max(0, y1_trk)
            object_region_x2 = min(raw_midas_map.shape[1], x2_trk)
            object_region_y2 = min(raw_midas_map.shape[0], y2_trk)

            current_avg_midas_value = None
            if object_region_x2 > object_region_x1 and \
               object_region_y2 > object_region_y1:
                object_midas_region = raw_midas_map[object_region_y1:object_region_y2,
                                                    object_region_x1:object_region_x2]

                if object_midas_region.size > 0:
                    sorted_midas_values = np.sort(object_midas_region.flatten())
                    start_idx = int(len(sorted_midas_values) * MIDAS_TRIM_PERCENTAGE)
                    end_idx = int(len(sorted_midas_values) * (1 - MIDAS_TRIM_PERCENTAGE))

                    if end_idx > start_idx:
                        trimmed_midas_values = sorted_midas_values[start_idx:end_idx]
                        current_avg_midas_value = np.median(trimmed_midas_values)
                    else:
                        current_avg_midas_value = np.median(object_midas_region)

            smoothed_midas_value = None
            if current_avg_midas_value is not None:
                if track_id not in tracked_object_midas_history:
                    tracked_object_midas_history[track_id] = []

                tracked_object_midas_history[track_id].append((current_avg_midas_value, frame_count))

                if len(tracked_object_midas_history[track_id]) > MIDAS_HISTORY_LENGTH:
                    tracked_object_midas_history[track_id].pop(0)

                if len(tracked_object_midas_history[track_id]) > 0:
                    midas_values_only = [val[0] for val in tracked_object_midas_history[track_id]]
                    smoothed_midas_value = np.median(midas_values_only)

            current_distance = None
            distance_text = "D:N/A"

            if smoothed_midas_value is not None:
                if display_raw_midas_value:
                    current_value_to_display = midas_value_to_distance(smoothed_midas_value,
                                                                         K=MIDAS_K_CONVERT, C=MIDAS_C_CONVERT,
                                                                         max_valid_midas_value=MIDAS_MAX_VALID_VALUE,
                                                                         min_valid_midas_value=MIDAS_MIN_VALID_VALUE,
                                                                         return_raw=True)
                    distance_text = f"D_raw:{current_value_to_display:.2f}"
                    current_distance = float('inf')
                else:
                    current_distance = midas_value_to_distance(smoothed_midas_value,
                                                                 K=MIDAS_K_CONVERT, C=MIDAS_C_CONVERT,
                                                                 max_valid_midas_value=MIDAS_MAX_VALID_VALUE,
                                                                 min_valid_midas_value=MIDAS_MIN_VALID_VALUE)
                    display_max_dist = DANGER_MAX_DISTANCE * 2
                    if current_distance is not None and current_distance != float('inf'):
                        if current_distance <= display_max_dist:
                            distance_text = f"D:{current_distance:.2f}m"
                        else:
                            distance_text = f"D:>{int(display_max_dist)}m"
                    else:
                        distance_text = "D:N/A"

            is_object_too_far = False
            if current_distance is None or \
               current_distance == float('inf') or \
               (not display_raw_midas_value and current_distance > DANGER_MAX_DISTANCE * 1.5):
                is_object_too_far = True
                if track_id in tracked_object_distance_history:
                    del tracked_object_distance_history[track_id]

            box_color_rgb = (255, 165, 0) # Orange
            text_color_individual = (255, 255, 255)

            center_x_bbox = (x1_trk + x2_trk) / 2
            left_zone_end_x = original_width * 0.35
            right_zone_start_x = original_width * 0.65
            
            direction_text = "前方"
            if center_x_bbox < left_zone_end_x:
                direction_text = "左前"
            elif center_x_bbox > right_zone_start_x:
                direction_text = "右前"

            # Determine if the object is in the "main path" for voice alerts
            in_main_alert_path = False
            if main_path_left_bound <= center_x_bbox <= main_path_right_bound:
                in_main_alert_path = True


            if not is_object_too_far and current_distance is not None and \
               not display_raw_midas_value and \
               class_id not in [PEDESTRIAN_GREEN_LIGHT_ID, PEDESTRIAN_RED_LIGHT_ID,
                                 TRAFFIC_LIGHT_GREEN_ID, TRAFFIC_LIGHT_RED_ID, TRAFFIC_LIGHT_YELLOW_ID]:
                # Store all relevant information, including whether it's in the main alert path
                all_detected_objects_info.append((class_name, current_distance, track_id, direction_text, is_static_obstacle, in_main_alert_path, fast_approach_flag))

            draw.rectangle([(x1_trk, y1_trk), (x2_trk, y2_trk)], outline=box_color_rgb, width=2)

            text_lines = [
                f"ID:{track_id} {class_name}",
            ]

            if not display_raw_midas_value:
                text_lines.append(distance_text)
            
            text_lines.append(f"方向:{direction_text}")

            y_offset = y1_trk
            for i, line in enumerate(text_lines):
                bbox = draw.textbbox((0, 0), line, font=font_pil)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x_bg = x1_trk
                text_y_bg = y_offset - (text_height) - 5

                if text_y_bg < 0:
                    text_y_bg = y1_trk + (i * (text_height + 5)) + 5
                    if y_offset < y1_trk:
                        y_offset = text_y_bg

                draw.rectangle([(text_x_bg, text_y_bg), (text_x_bg + text_width, text_y_bg + text_height)], fill=(0, 0, 0))
                draw.text((text_x_bg, text_y_bg), line, font=font_pil, fill=text_color_individual)

                if text_y_bg < y1_trk:
                    y_offset = text_y_bg - 5
                else:
                    y_offset = text_y_bg + text_height + 5

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if all_detected_objects_info:
            # Sort by distance for general printing, not necessarily for voice alert logic
            all_detected_objects_info.sort(key=lambda x: x[1]) 
            print(f"\n--- Frame {frame_count}: Detected Objects ---")
            for i, (class_name, distance, track_id, direction, is_static_obstacle, in_main_path, fast_approach) in enumerate(all_detected_objects_info[:5]):
                static_status = "(靜態)" if is_static_obstacle else "(移動)"
                main_path_status = "(主路徑)" if in_main_path else "(側邊)"
                approach_status = "(快速接近)" if fast_approach else ""
                print(f"  {i+1}. ID: {track_id}, 物體: {class_name}{static_status}{main_path_status}{approach_status}, 距離: {distance:.2f}m, 方向: {direction}")
        else:
            if frame_count % 30 == 0:
                print(f"\n--- Frame {frame_count}: 未檢測到任何物體 ---")

        current_time = time.time()
        alert_message = ""
        current_frame_effective_level = "Safe"

        if not display_raw_midas_value:
            alert_priority_order = {
                "Safe": 0,
                "Obstacle Alert": 1,
                "Traffic Green": 2,
                "Pedestrian Green": 3,
                "Traffic Yellow": 4,
                "Traffic Red": 5,
                "Pedestrian Red": 6
            }

            if pedestrian_red_light_detected:
                alert_message = "行人紅燈，請等待。"
                current_frame_effective_level = "Pedestrian Red"
            elif traffic_light_red_detected:
                alert_message = "紅燈，請等待。"
                current_frame_effective_level = "Traffic Red"
            # NEW LOGIC for Obstacle Alert: Only consider static obstacles in the main path
            elif True: # Use a placeholder True to simplify the nested conditions
                closest_main_path_static_obstacle = None
                min_obstacle_distance = float('inf')
                
                # Filter for static obstacles that are in the main path and within alert distance
                eligible_obstacles = [
                    obj for obj in all_detected_objects_info
                    if obj[4] and obj[5] and obj[1] <= OBSTACLE_ALERT_DISTANCE # is_static_obstacle AND in_main_alert_path AND within OBSTACLE_ALERT_DISTANCE
                ]

                if eligible_obstacles:
                    # Sort by distance and pick the closest one
                    closest_main_path_static_obstacle = min(eligible_obstacles, key=lambda x: x[1])
                    
                if closest_main_path_static_obstacle:
                    class_name, distance, _, direction, _, _, fast_approach = closest_main_path_static_obstacle
                    # Use the specific direction for the main path obstacle and transformer-based motion
                    if fast_approach:
                        alert_message = f"{direction} {distance:.1f}公尺有障礙物快速接近：{class_name}，請注意！"
                    else:
                        alert_message = f"{direction} {distance:.1f}公尺有障礙物：{class_name}，請注意！"
                    current_frame_effective_level = "Obstacle Alert"
            
            # Continue with other light alerts if no critical obstacle alert
            if not alert_message: # Only check other conditions if no alert has been set yet
                if traffic_light_yellow_detected:
                    alert_message = "黃燈，請注意！"
                    current_frame_effective_level = "Traffic Yellow"
                elif pedestrian_green_light_detected:
                    alert_message = "行人綠燈，請通行。"
                    current_frame_effective_level = "Pedestrian Green"
                elif traffic_light_green_detected:
                    alert_message = "綠燈，請通行。"
                    current_frame_effective_level = "Traffic Green"


            should_speak = False

            if alert_message:
                current_level_priority = alert_priority_order.get(current_frame_effective_level, 0)
                last_level_priority = alert_priority_order.get(last_effective_danger_level, 0)

                if current_level_priority > last_level_priority:
                    should_speak = True
                elif current_level_priority == last_level_priority and alert_message != last_spoken_alert_message:
                    should_speak = True
                elif current_level_priority == last_level_priority and alert_message == last_spoken_alert_message:
                    # ✅ 修正：以語音播報所需時間作為冷卻間隔，防止過度重複播報
                    cooldown_to_use = 3.5

                    if (current_time - last_alert_time > cooldown_to_use):
                        should_speak = True

                if should_speak:
                    speak_async(alert_message)
                    last_alert_time = current_time
                    last_spoken_alert_message = alert_message
                    last_effective_danger_level = current_frame_effective_level
                    print(f"  ACTION: Speaking: '{alert_message}'")
            else:
                # 若無警告訊息，且冷卻時間已過，則重置語音狀態以便下次播報
                if last_spoken_alert_message and (current_time - last_alert_time > DANGER_RESET_COOLDOWN):
                    last_spoken_alert_message = ""
                    last_effective_danger_level = "Safe"

        combined_output_frame = frame.copy()

        display_width_current = combined_output_frame.shape[1]
        display_height_current = combined_output_frame.shape[0]

        scale_factor_w = 1.0
        scale_factor_h = 1.0
        if display_width_current > screen_width:
            scale_factor_w = screen_width / display_width_current
        if display_height_current > screen_height:
            scale_factor_h = screen_height / display_height_current

        scale = min(scale_factor_w, scale_factor_h)

        display_width_scaled = int(display_width_current * scale * 0.4) # Adjusted for better view on common screens
        display_height_scaled = int(display_height_current * scale * 0.4)

        window_name = "Obstacle Detection and Alert for Visually Impaired"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width_scaled, display_height_scaled)

        cv2.imshow(window_name, combined_output_frame)

        if out:
            if combined_output_frame.shape[1] != output_video_width or \
               combined_output_frame.shape[0] != output_video_height:
                resized_for_output = cv2.resize(combined_output_frame, (output_video_width, output_video_height))
                out.write(resized_for_output)
            else:
                out.write(combined_output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video: {video_path} processing complete.")


if __name__ == '__main__':
    # Set inference device (GPU preferred)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Detected compute device: {device}")

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # --- Configuration Parameters ---
    # 請根據您的實際文件路徑和需求修改以下參數！
    # VVVVVV 請將此路徑替換為您 YOLOv12 模型權重檔案的實際路徑 VVVVVV
    YOUR_YOLOV12_WEIGHTS = 'D:/yolov12/runs/detect/blind_guide_yolov12_s_run_1/weights/best.pt'
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # --- 影片偵測設定 ---
    # VVVVVV 請將此路徑替換為您要偵測影片的資料夾路徑 VVVVVV
    VIDEO_FOLDER_PATH = 'D:/yolov12/blind guide.v5i.yolov12/test/video'
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # 預期的影片檔案副檔名
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 不保存輸出影片，因此設置為 None
    OUTPUT_VIDEO_PATH = None 

    MIDAS_MODEL_TYPE = "DPT_Hybrid" # Options: "DPT_Hybrid", "DPT_Large", "MiDaS_small"

    # YOLO and SORT Parameters
    CONF_THRESHOLD = 0.25           # Detection confidence threshold
    IOU_THRESHOLD = 0.5             # IoU threshold for Non-Maximum Suppression (NMS)
    YOLO_IMGSZ = 640                # YOLO input image size (can be int or 'auto')
    SORT_MAX_AGE = 10               # SORT tracker: Maximum number of frames an object can be missing
    SORT_MIN_HITS = 2               # SORT tracker: Minimum number of detections required for an object to be considered a valid track
    SORT_IOU_THRESHOLD = 0.3        # SORT tracker: IoU threshold for object matching

    TARGET_CLASSES = None           # List of specific classes to detect (None for all classes)

    DETECTION_INTERVAL = 1          # Detect and estimate depth every N frames (1 means every frame)

    MIDAS_OUTPUT_SCALE_FACTOR = 1.0 # Scaling factor for MiDaS output depth map (1.0 means same size as input frame)

    # --- MiDaS Distance Conversion K and C Parameters ---
    MIDAS_K_CONVERT = 3000
    MIDAS_C_CONVERT = 450
    MIDAS_MAX_VALID_VALUE = 3000
    MIDAS_MIN_VALID_VALUE = 460 # Make sure this is greater than C_CONVERT

    display_raw_midas_value=False

    # --- OBSTACLE ALERT DISTANCE ---
    OBSTACLE_ALERT_DISTANCE = 5.0 # 靜態障礙物提醒距離 (公尺)

    # NEW: 主行徑路徑中央寬度百分比 (用於語音警報篩選)
    # 例如：0.4 表示只對畫面中央 40% 寬度內的障礙物發出語音警報
    MAIN_PATH_CENTER_WIDTH_PERCENTAGE = 0.4 

    # Danger Assessment Parameters (內部計算用，不影響語音提醒分類)
    DANGER_MAX_DISTANCE = 15.0 # 最大考慮距離
    DANGER_MIN_SAFE_DISTANCE = 0.5 # 最小安全距離 (例如，如果物體距離小於此，則視為極近)
    WEIGHT_DISTANCE = 100 # 距離權重為100%，表示只考慮距離
    WEIGHT_VELOCITY = 0   # 速度權重設為0
    WEIGHT_ACCELERATION = 0 # 加速度權重設為0

    # --- VOICE ALERT COOLDOWN TIMES ---
    VOICE_ALERT_COOLDOWN_OBSTACLE=3.0 # 靜態障礙物語音提示冷卻時間
    VOICE_ALERT_COOLDOWN_LIGHT=5.0 # 交通燈語音提示冷卻時間
    DANGER_RESET_COOLDOWN = 2.0 # 當沒有危險時，語音提示重置的冷卻時間

    # Your screen resolution (please modify these values according to your actual setup!)
    YOUR_SCREEN_WIDTH = 3840
    YOUR_SCREEN_HEIGHT = 2160

    print("\n--- 載入模型和初始化語音引擎 ---")
    # 載入 YOLO 模型
    yolo_model_loaded = load_yolov12_model(YOUR_YOLOV12_WEIGHTS)
    if yolo_model_loaded is None:
        print("無法載入 YOLO 模型，程式終止。")
        exit()

    # 載入 MiDaS 模型和轉換器
    print(f"Loading MiDaS model ({MIDAS_MODEL_TYPE})... This might take some time to download (only first time).")
    try:
        midas_model_loaded = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
        midas_model_loaded.to(device)
        midas_model_loaded.eval()
        print("MiDaS model loaded successfully.")
    except Exception as e:
        print(f"Error: Could not load MiDaS model '{MIDAS_MODEL_TYPE}'. Check network connection or model name.\nError message: {e}")
        exit()

    print("Loading MiDaS image transforms...")
    try:
        midas_transforms_loaded = torch.hub.load("intel-isl/MiDaS", "transforms")
        if MIDAS_MODEL_TYPE in ["DPT_Large", "DPT_Hybrid"]:
            transform_midas_loaded = midas_transforms_loaded.dpt_transform
        elif MIDAS_MODEL_TYPE == "MiDaS_small":
            transform_midas_loaded = midas_transforms_loaded.small_transform
        else:
            print(f"Warning: Unknown MiDaS model type '{MIDAS_MODEL_TYPE}'. Defaulting to DPT transform.")
            transform_midas_loaded = midas_transforms_loaded.dpt_transform
        print("MiDaS image transforms loaded successfully.")
    except Exception as e:
        print(f"Error: Could not load MiDaS image transforms.\nError message: {e}")
        exit()

    # 初始化語音引擎 (確保只初始化一次)
    init_tts_engine()
    print("語音引擎初始化完成。")

    print(f"\n--- 開始處理資料夾中的影片: {VIDEO_FOLDER_PATH} ---")

    video_files_to_process = []
    if os.path.isdir(VIDEO_FOLDER_PATH):
        for filename in os.listdir(VIDEO_FOLDER_PATH):
            file_path = os.path.join(VIDEO_FOLDER_PATH, filename)
            if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files_to_process.append(file_path)
    else:
        print(f"錯誤：指定的影片資料夾 '{VIDEO_FOLDER_PATH}' 不存在。請檢查路徑。")
        exit()

    if not video_files_to_process:
        print(f"在資料夾 '{VIDEO_FOLDER_PATH}' 中未找到任何支援的影片檔案。請檢查資料夾內容和 VIDEO_EXTENSIONS。")
    else:
        for video_file in video_files_to_process:
            print(f"\n>>>>>>>>>>>> 正在處理影片: {video_file} <<<<<<<<<<<<")
            main(video_path=video_file,
                 yolo_model=yolo_model_loaded,
                 midas_model=midas_model_loaded,
                 midas_transform=transform_midas_loaded,
                 device=device,
                 output_video_path=OUTPUT_VIDEO_PATH,
                 conf_threshold=CONF_THRESHOLD,
                 iou_threshold=IOU_THRESHOLD,
                 yolo_imgsz=YOLO_IMGSZ,
                 sort_max_age=SORT_MAX_AGE,
                 sort_min_hits=SORT_MIN_HITS,
                 sort_iou_threshold=SORT_IOU_THRESHOLD,
                 target_classes=TARGET_CLASSES,
                 detection_interval=DETECTION_INTERVAL,
                 midas_output_scale_factor=MIDAS_OUTPUT_SCALE_FACTOR,
                 MIDAS_K_CONVERT=MIDAS_K_CONVERT,
                 MIDAS_C_CONVERT=MIDAS_C_CONVERT,
                 MIDAS_MAX_VALID_VALUE=MIDAS_MAX_VALID_VALUE,
                 MIDAS_MIN_VALID_VALUE=MIDAS_MIN_VALID_VALUE,
                 display_raw_midas_value=display_raw_midas_value,
                 OBSTACLE_ALERT_DISTANCE=OBSTACLE_ALERT_DISTANCE,
                 MAIN_PATH_CENTER_WIDTH_PERCENTAGE=MAIN_PATH_CENTER_WIDTH_PERCENTAGE, # Pass new parameter
                 DANGER_MAX_DISTANCE=DANGER_MAX_DISTANCE,
                 DANGER_MIN_SAFE_DISTANCE=DANGER_MIN_SAFE_DISTANCE,
                 WEIGHT_DISTANCE=WEIGHT_DISTANCE,
                 WEIGHT_VELOCITY=WEIGHT_VELOCITY,
                 WEIGHT_ACCELERATION=WEIGHT_ACCELERATION,
                 VOICE_ALERT_COOLDOWN_OBSTACLE=VOICE_ALERT_COOLDOWN_OBSTACLE,
                 VOICE_ALERT_COOLDOWN_LIGHT=VOICE_ALERT_COOLDOWN_LIGHT,
                 DANGER_RESET_COOLDOWN=DANGER_RESET_COOLDOWN,
                 screen_width=YOUR_SCREEN_WIDTH,
                 screen_height=YOUR_SCREEN_HEIGHT
            )
            print(f">>>>>>>>>>>> 影片 {video_file} 處理完畢 <<<<<<<<<<<<\n")

    print("\n所有影片偵測已停止。")