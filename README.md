# 盲人導引系統

本專案展示一個以影片為基礎的輔助流程，透過 **YOLOv12** 偵測障礙與交通號誌、使用 **MiDaS** 估計距離，並搭配 **SORT** 追蹤物體，最後由 `pyttsx3` 發出語音提醒。

## 安裝步驟

1. 建立並啟用 Python 3.8 以上的虛擬環境。
2. 安裝必要套件：

   ```bash
   pip install torch ultralytics opencv-python pyttsx3 pillow numpy
   ```

## 使用方式

1. **準備資源**
   - 下載並將 YOLOv12 權重路徑填入 `yolov12_tracker.py` 內的 `YOUR_YOLOV12_WEIGHTS`。
   - 將待分析的影片放在資料夾中，並設定 `VIDEO_FOLDER_PATH` 為該路徑。

2. **執行程式**

   ```bash
   python yolov12_tracker.py
   ```

   程式會載入模型、處理資料夾內的每支影片、追蹤物體、估計深度並透過語音提示障礙物及交通號誌。

## 參數調整

所有參數都集中在 `yolov12_tracker.py` 末端的「設定區塊」，可依需求修改：

- **路徑相關**
  - `YOUR_YOLOV12_WEIGHTS`：YOLOv12 權重檔路徑。
  - `VIDEO_FOLDER_PATH`：欲處理的影片資料夾。
  - `OUTPUT_VIDEO_PATH`：若要儲存加上標註的輸出影片，填入檔案路徑；預設為 `None` 表示不儲存。

- **偵測與追蹤**
  - `CONF_THRESHOLD`：偵測信心閾值，介於 0~1，值越高偵測越嚴格。
  - `IOU_THRESHOLD`：NMS 的 IoU 閾值，用來過濾重疊框。
  - `YOLO_IMGSZ`：YOLO 的輸入尺寸，可設為 640 或 `"auto"`。
  - `SORT_MAX_AGE` / `SORT_MIN_HITS` / `SORT_IOU_THRESHOLD`：SORT 追蹤器的追蹤條件與壽命控制。
  - `PROCESS_FPS`：將輸入影片降採樣到指定 FPS 進行運算。

- **深度估測**
  - `MIDAS_MODEL_TYPE`：MiDaS 模型類型，如 `"DPT_Hybrid"`、`"DPT_Large"`、`"MiDaS_small"`。
  - `MIDAS_OUTPUT_SCALE_FACTOR`：調整深度圖輸出大小的比例。
  - `MIDAS_K_CONVERT` / `MIDAS_C_CONVERT`：將 MiDaS 深度值轉換成實際距離時使用的參數，可依實測調整校正。

- **警示相關**
  - `OBSTACLE_ALERT_DISTANCE`：靜態障礙物的語音提醒距離（公尺）。
  - `MAIN_PATH_CENTER_WIDTH_PERCENTAGE`：只對畫面中央一定寬度範圍內的物體發出語音提醒，0.4 代表中間 40%。
  - `VOICE_ALERT_COOLDOWN_OBSTACLE` / `VOICE_ALERT_COOLDOWN_LIGHT`：語音提示的冷卻時間（秒）。

## 專案架構

| 檔案 | 說明 |
|------|------|
| `yolov12_tracker.py` | 主流程：偵測、追蹤、深度估測與語音提示。 |
| `sort.py` | SORT 多目標追蹤演算法。 |
| `temporal_transformer.py` | 取得影格的時序特徵。 |
| `speak_queue_manager.py` | 管理語音佇列與播報。 |

## 注意事項

- 首次執行時 `torch.hub` 會下載 MiDaS 模型，需要網路連線。
- 若需語音提示請確認系統有可用的音訊輸出裝置。

