# sort.py

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment

def iou_batch(bb_test, bb_gt):
  """
  計算單個檢測框與一組 ground truth 框之間的 IoU。
  
  參數:
    bb_test: (ndarray) 測試邊界框，形狀為 (N, 4)，N 為檢測框數量。
    bb_gt: (ndarray) 真實邊界框，形狀為 (M, 4)，M 為真實框數量。
    
  返回:
    (ndarray) IoU 矩陣，形狀為 (N, M)。
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h # 交集面積

  # 計算 IoU
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) \
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                  
  return(o)  

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    將檢測與現有追蹤器進行關聯。

    參數:
      detections: (ndarray) 檢測框的 NumPy 陣列，格式為 [[x1,y1,x2,y2],...]。
      trackers: (ndarray) 追蹤器預測框的 NumPy 陣列，格式為 [[x1,y1,x2,y2],...]。
      iou_threshold: (float) IoU 匹配閾值。
    返回:
      matched_indices: (ndarray) 一個數組，其中每一行代表 (檢測索引, 追蹤器索引) 的匹配對。
      unmatched_detections: (ndarray) 未匹配到的檢測索引數組。
      unmatched_trackers: (ndarray) 未匹配到的追蹤器索引數組。
    """
    if(len(trackers)==0):
        # 如果沒有追蹤器，所有檢測都是未匹配的
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,0),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        # 使用匈牙利演算法進行線性指派
        # linear_assignment 尋找的是最小成本匹配，所以我們需要將 IoU 轉換為成本 (成本 = 1 - IoU)
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_assignment(cost_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1) # 組合為 (det_idx, trk_idx) 格式

        # 過濾掉不符合 IoU 閾值的匹配
        matched_indices = matched_indices[iou_matrix[matched_indices[:, 0], matched_indices[:, 1]] > iou_threshold]
    else:
        matched_indices = np.empty(shape=(0,2))

    # 找出未匹配的檢測
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    # 找出未匹配的追蹤器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)
    
    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
  """
  這是一個單目標追蹤器，它維護一個卡爾曼濾波器來估計目標的狀態。
  """
  count = 0 # 靜態變數，用於生成唯一的追蹤器 ID
  def __init__(self, bbox, class_id=None, confidence=None):
    """
    初始化一個追蹤器。
    
    參數:
      bbox: (ndarray) 初始檢測框 [x1, y1, x2, y2]。
      class_id: (int) 檢測到的類別 ID。
      confidence: (float) 檢測的置信度。
    """
    # 定義卡爾曼濾波器
    # dim_x=7: 狀態向量 [cx, cy, s, r, v_cx, v_cy, v_s]
    #   cx, cy: 邊界框中心座標
    #   s: 邊界框面積 (scale)
    #   r: 邊界框長寬比 (aspect ratio)
    #   v_cx, v_cy, v_s: cx, cy, s 的速度
    # dim_z=4: 測量向量 [cx, cy, s, r]
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    
    # 狀態轉換矩陣 (F): 線性運動模型 (勻速模型)
    self.kf.F = np.array([[1,0,0,0,1,0,0], # cx = cx + v_cx
                          [0,1,0,0,0,1,0], # cy = cy + v_cy
                          [0,0,1,0,0,0,1], # s = s + v_s
                          [0,0,0,1,0,0,0], # r = r (長寬比假設不變)
                          [0,0,0,0,1,0,0], # v_cx = v_cx
                          [0,0,0,0,0,1,0], # v_cy = v_cy
                          [0,0,0,0,0,0,1]]) # v_s = v_s

    # 測量函數矩陣 (H): 將狀態轉換為測量值
    # 我們測量 cx, cy, s, r
    self.kf.H = np.array([[1,0,0,0,0,0,0], # 測量 cx
                          [0,1,0,0,0,0,0], # 測量 cy
                          [0,0,1,0,0,0,0], # 測量 s
                          [0,0,0,1,0,0,0]]) # 測量 r

    # 測量噪音協方差矩陣 (R): 測量值的不確定性
    self.kf.R[2:,2:] *= 10. # 給予面積和長寬比更大的測量不確定性

    # 初始狀態協方差矩陣 (P): 初始狀態估計的不確定性
    self.kf.P[4:,4:] *= 1000. # 給予速度分量更大的不確定性 (初始速度未知)
    self.kf.P *= 10.          # 整體放大初始不確定性

    # 過程噪音協方差矩陣 (Q): 狀態轉換模型的不確定性 (模型誤差)
    self.kf.Q[-1,-1] *= 0.01 # 給予面積速度很小的噪音
    self.kf.Q[4:,4:] *= 0.01 # 給予 cx, cy 速度較小的噪音

    # 設置卡爾曼濾波器的初始狀態
    self.kf.x[:4] = self.convert_bbox_to_z(bbox) # 將初始檢測框轉換為狀態空間中的測量值

    self.time_since_update = 0 # 距離上次更新的時間
    self.id = KalmanBoxTracker.count # 分配唯一的追蹤器 ID
    KalmanBoxTracker.count += 1
    self.history = [] # 追蹤歷史 (可選，通常用於可視化或軌跡平滑)
    self.hits = 0 # 總擊中次數 (被檢測匹配到的次數)
    self.hit_streak = 0 # 連續擊中次數 (用於判斷追蹤器是否穩定)
    self.age = 0 # 追蹤器生命週期 (已存在的幀數)
    
    # 儲存類別 ID 和置信度
    self.class_id = class_id 
    self.confidence = confidence 

  def update(self, bbox, class_id=None, confidence=None):
    """
    更新目標的狀態。
    
    參數:
      bbox: (ndarray) 新的檢測框 [x1, y1, x2, y2] NumPy 陣列；
            如果為 None 或空陣列，則表示沒有新的檢測，只進行預測。
      class_id: (int) 檢測到的類別 ID。
      confidence: (float) 檢測的置信度。
    """
    self.time_since_update = 0
    self.history = [] # 清空歷史，因為狀態已更新
    self.hits += 1
    self.hit_streak += 1 # 連續擊中次數增加
    
    # 如果有新的檢測，則更新卡爾曼濾波器並儲存類別/置信度
    if bbox is not None and bbox.size > 0:
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.class_id = class_id # 更新類別 ID
        self.confidence = confidence # 更新置信度
    else: 
        # 如果沒有新的檢測，只進行預測 (用於跳幀)
        self.kf.predict()
        self.hit_streak = 0 # 沒有新的擊中，重置連擊數

  def predict(self):
    """
    將卡爾曼濾波器的狀態向前推進一步（預測下一個時間步的位置）。
    """
    # 如果預測的面積變為非正值，重置速度，防止卡爾曼濾波器發散
    # 狀態向量中的 s 是面積 (索引 2)，v_s 是面積速度 (索引 6)
    if((self.kf.x[2] + self.kf.x[6]) <= 0): 
      self.kf.x[6] *= 0.0 # 將面積速度設為 0
    
    self.kf.predict() # 執行卡爾曼濾波器預測步驟
    self.age += 1 # 追蹤器生命週期增加
    
    if(self.time_since_update > 0): 
      self.hit_streak = 0 # 如果這幀沒有更新，重置連擊數

    self.time_since_update += 1 # 未更新時間增加
    
    self.history.append(self.convert_x_to_bbox(self.kf.x)) # 將預測結果加入歷史 (用於回溯或平滑)
    return self.history[-1] # 返回最新的預測邊界框

  def get_state(self):
    """
    返回卡爾曼濾波器當前估計的邊界框狀態。
    """
    return self.convert_x_to_bbox(self.kf.x)

  def convert_bbox_to_z(self, bbox):
    """
    將邊界框 [x1,y1,x2,y2] 轉換為卡爾曼濾波器測量值 z。
    z 的格式為 [cx, cy, s, r]，其中 cx, cy 是中心點，s 是面積，r 是長寬比。
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    # scale (面積)
    r = w/float(h) # 長寬比
    return np.array([x,y,s,r]).reshape((4,1))

  def convert_x_to_bbox(self, x_state):
    """
    將卡爾曼濾波器狀態 x_state 轉換回邊界框 [x1,y1,x2,y2] 格式。
    x_state 的格式為 [cx, cy, s, r, v_cx, v_cy, v_s] (7x1 矩陣或 1D 陣列)。
    """
    # 確保 x_state 是一個 1D NumPy 陣列
    if isinstance(x_state, np.ndarray) and x_state.ndim == 2 and x_state.shape[1] == 1:
        x_state = x_state.flatten() # 將 (7,1) 變成 (7,)

    cx = x_state[0] # 中心 x 座標
    cy = x_state[1] # 中心 y 座標
    s = x_state[2]  # 面積 (scale)
    r = x_state[3]  # 長寬比 (aspect ratio)

    # 防止長寬比或面積為非正數，確保數學運算有效
    if s <= 0: s = 1.0 
    if r <= 0: r = 1.0 

    # 從面積和長寬比反推出寬和高
    w = np.sqrt(s * r)
    h = s / w
    
    # 計算邊界框的四個角點座標
    return np.array([cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.]).reshape((1, 4))


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    初始化 SORT 追蹤器。

    參數:
      max_age: (int) 追蹤器在沒有檢測更新的情況下可以保留的最大幀數。
      min_hits: (int) 追蹤器在被確認為有效軌跡之前需要被連續擊中的最小次數。
      iou_threshold: (float) 檢測與追蹤器匹配的 IoU 閾值。
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = [] # 當前活躍的 KalmanBoxTracker 物件列表
    self.frame_count = 0 # 已處理的幀數

  def update(self, detections):
    """
    更新追蹤器狀態。
    
    參數:
      detections: (ndarray) 檢測框的 NumPy 陣列，格式為 [[x1,y1,x2,y2,score, class_id],...]。
                  如果當前幀沒有檢測，則傳入 np.empty((0,6), dtype=np.float32)。
                  
    返回:
      (ndarray) 追蹤物件的 NumPy 陣列，格式為 [x1,y1,x2,y2,id,class_id,score]。
      如果沒有活躍追蹤器，則返回 np.empty((0,7))。
    """
    self.frame_count += 1

    # 1. 預測所有現有追蹤器的下一個位置
    # 這裡的 trks 將只包含預測的邊界框 [x1,y1,x2,y2]
    trks = np.zeros((len(self.trackers), 4)) # 只需要 bbox，不需要佔位符 0
    to_del = [] # 記錄需要刪除的追蹤器索引
    for t, trk in enumerate(self.trackers):
      pos = trk.predict()[0] # 獲取預測的邊界框 (flattened array)
      trks[t, :] = pos
      if np.any(np.isnan(pos)): # 如果預測結果出現 NaN (發散)，則標記為待刪除
        to_del.append(t)
    
    # 刪除發散的追蹤器
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # 壓縮掉 NaN 行
    for t in reversed(to_del): # 從後往前刪除，避免索引錯亂
      self.trackers.pop(t)
    
    # 2. 準備檢測結果：分離 bbox、class_id 和 confidence
    det_boxes = detections[:, :4] if detections.size > 0 else np.empty((0, 4)) 
    det_confidences = detections[:, 4] if detections.size > 0 else np.empty(0)
    det_class_ids = detections[:, 5].astype(int) if detections.size > 0 else np.empty(0, dtype=int)
    
    # 3. 進行檢測與追蹤器的匹配
    # detections_for_association 傳入的是 bbox (x1,y1,x2,y2)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(det_boxes, trks, self.iou_threshold)

    # 4. 更新已匹配的追蹤器
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks: # 如果追蹤器被匹配到了
        d_indices = matched[np.where(matched[:,1]==t)[0],0] # 找到與當前追蹤器匹配的檢測索引
        if d_indices.size > 0:
            det_idx = d_indices[0] # 取第一個匹配的檢測
            # 更新追蹤器狀態，並傳遞新的類別 ID 和置信度
            trk.update(det_boxes[det_idx], det_class_ids[det_idx], det_confidences[det_idx])
        else:
            # 理論上已匹配的追蹤器應該總能找到對應的檢測
            # 如果出現，可能是匹配邏輯問題，這裡做個防禦性處理
            trk.update(None, None, None) # 讓 KalmanBoxTracker 只做預測

    # 5. 創建並初始化新的追蹤器 (針對未匹配到的檢測)
    for i in unmatched_dets:
        # 創建新的追蹤器時，帶上 class_id 和 confidence
        new_tracker = KalmanBoxTracker(detections[i, :4], detections[i, 5].astype(int), detections[i, 4])
        self.trackers.append(new_tracker)
        
    # 6. 組裝返回結果並刪除不再活躍的追蹤器
    ret = []
    i = len(self.trackers)
    for trk in reversed(self.trackers): # 從後往前遍歷，方便刪除
        # 判斷追蹤器是否為活躍狀態 (未更新時間小於最大值，且**累積**擊中次數滿足最小要求或在初始幀內)
        # 原邏輯使用 hit_streak(連續擊中次數)，在跳幀偵測時會因為未連續偵測而導致追蹤器過早消失。
        # 改為使用 hits(累積擊中次數) 以確保即使不是每幀都偵測，追蹤器仍能持續回傳預測位置。
        if (trk.time_since_update < self.max_age) and \
                (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
            
            bbox_state = trk.get_state()[0] # 追蹤器預測的邊界框狀態

            # 組裝返回格式: [x1,y1,x2,y2,id,class_id,score]
            ret.append(np.concatenate((bbox_state, 
                                       np.array([trk.id + 1]), # ID 通常從 1 開始
                                       np.array([trk.class_id]), 
                                       np.array([trk.confidence]))).reshape(1,-1))
        # 刪除已死的追蹤器（長時間未更新）
        i -= 1 # 確保在 pop 之前更新索引
        if trk.time_since_update >= self.max_age:
          self.trackers.pop(i)
    
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,7)) # 返回空的 7 列陣列