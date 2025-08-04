# Blind Guide System

This project demonstrates a video‑based assistance pipeline for visually impaired users. It detects obstacles and traffic lights with **YOLOv12**, estimates their distance with **MiDaS**, tracks movement using **SORT**, and provides spoken guidance through `pyttsx3`.

## Installation

1. Create and activate a Python 3.8+ environment.
2. Install required packages:

   ```bash
   pip install torch ultralytics opencv-python pyttsx3 pillow numpy
   ```

## Usage

1. **Prepare resources**
   - Download YOLOv12 weights and set the path in `yolov12_tracker.py` at `YOUR_YOLOV12_WEIGHTS`.
   - Place the videos you want to analyze in a folder and update `VIDEO_FOLDER_PATH`.

2. **Run detection**

   ```bash
   python yolov12_tracker.py
   ```

   The script loads the models, processes each video in the folder, tracks objects, estimates depth, and speaks alerts for approaching obstacles and traffic light signals.

3. **Optional parameters**
   - `OUTPUT_VIDEO_PATH`: save annotated output (set a file path instead of `None`).
   - `CONF_THRESHOLD`, `IOU_THRESHOLD`, `PROCESS_FPS`, etc. can be adjusted in the configuration section near the end of `yolov12_tracker.py`.

## Repository Structure

| File | Description |
|------|-------------|
| `yolov12_tracker.py` | Main pipeline: detection, tracking, depth estimation, and speech alerts. |
| `sort.py` | Implementation of the SORT multi‑object tracker. |
| `temporal_transformer.py` | Temporal feature extractor used for motion cues. |
| `speak_queue_manager.py` | Manages a queue for `pyttsx3` speech output. |

## Notes

- MiDaS models are downloaded via `torch.hub` on first run.
- Ensure audio output is available for voice alerts.

