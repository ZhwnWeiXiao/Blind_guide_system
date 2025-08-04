import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import torch

# Import detection pipeline
from yolov12_tracker import main as run_pipeline, load_yolov12_model

# Default model settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')
MIDAS_MODEL_TYPE = 'DPT_Hybrid'


def load_midas(model_type, device):
    """Load MiDaS model and corresponding transforms."""
    midas = torch.hub.load('intel-isl/MiDaS', model_type)
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    if model_type in ['DPT_Large', 'DPT_Hybrid']:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform


class BlindGuideApp:
    def __init__(self, master):
        self.master = master
        master.title('盲人導盲系統')

        self.mode = tk.StringVar(value='video')
        self.video_path = ''

        mode_frame = tk.LabelFrame(master, text='選擇模式')
        mode_frame.pack(padx=10, pady=10, fill='x')

        tk.Radiobutton(mode_frame, text='影片偵測', variable=self.mode, value='video',
                       command=self._toggle_video_button).pack(anchor='w')
        tk.Radiobutton(mode_frame, text='即時偵測', variable=self.mode, value='realtime',
                       command=self._toggle_video_button).pack(anchor='w')

        file_frame = tk.Frame(master)
        file_frame.pack(padx=10, pady=5, fill='x')
        self.select_btn = tk.Button(file_frame, text='選擇影片', command=self._choose_video)
        self.select_btn.pack(side='left')
        self.file_label = tk.Label(file_frame, text='未選擇任何檔案', anchor='w')
        self.file_label.pack(side='left', padx=5)

        action_frame = tk.Frame(master)
        action_frame.pack(padx=10, pady=10)
        tk.Button(action_frame, text='開始偵測', command=self.start_detection).pack(side='left', padx=5)
        tk.Button(action_frame, text='離開', command=master.quit).pack(side='left', padx=5)

        self._toggle_video_button()

    def _toggle_video_button(self):
        if self.mode.get() == 'video':
            self.select_btn.config(state='normal')
        else:
            self.select_btn.config(state='disabled')

    def _choose_video(self):
        path = filedialog.askopenfilename(title='選擇影片',
                                          filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv')])
        if path:
            self.video_path = path
            self.file_label.config(text=os.path.basename(path))

    def start_detection(self):
        mode = self.mode.get()
        if mode == 'video' and not self.video_path:
            messagebox.showerror('錯誤', '請先選擇影片檔案')
            return
        threading.Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolo_model = load_yolov12_model(YOLO_WEIGHTS_PATH)
        if yolo_model is None:
            messagebox.showerror('錯誤', f'無法載入 YOLO 權重: {YOLO_WEIGHTS_PATH}')
            return
        midas_model, midas_transform = load_midas(MIDAS_MODEL_TYPE, device)
        source = 0 if self.mode.get() == 'realtime' else self.video_path
        run_pipeline(
            video_path=source,
            yolo_model=yolo_model,
            midas_model=midas_model,
            midas_transform=midas_transform,
            device=device,
        )


if __name__ == '__main__':
    root = tk.Tk()
    app = BlindGuideApp(root)
    root.mainloop()
