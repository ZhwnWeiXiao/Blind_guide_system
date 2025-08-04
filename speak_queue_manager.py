import threading
import time
import pyttsx3
from collections import deque

class SpeechItem:
    def __init__(self, message, obj_id=None, timestamp=None):
        self.message = message
        self.obj_id = obj_id
        self.timestamp = timestamp or time.time()

class SpeechQueueManager:
    def __init__(self, max_age_seconds=5.0):
        self.lock = threading.Lock()
        self.queue = deque()
        self.obj_id_map = {}
        self.max_age = max_age_seconds

        # ✅ 初始化一次 engine，並設定中文語音
        self.engine = pyttsx3.init()
        self._setup_voice(self.engine)
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

        # ✅ 啟動背景播報執行緒
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

    def _setup_voice(self, engine):
        voices = engine.getProperty('voices')
        chinese_voice_found = False
        for voice in voices:
            name = voice.name.lower()
            vid = voice.id.lower()
            if "taiwan" in name or "zh-tw" in vid:
                engine.setProperty('voice', voice.id)
                chinese_voice_found = True
                print(f"✅ 使用台灣中文語音: {voice.name}")
                break
            elif "chinese" in name or "zh" in vid:
                engine.setProperty('voice', voice.id)
                chinese_voice_found = True
                print(f"✅ 使用中文語音: {voice.name}")
                break
        if not chinese_voice_found:
            print("⚠️ 找不到中文語音，使用預設語音")
            if voices:
                engine.setProperty('voice', voices[0].id)

    def enqueue(self, message, obj_id=None):
        now = time.time()
        with self.lock:
            if obj_id and obj_id in self.obj_id_map:
                existing = self.obj_id_map[obj_id]
                if existing.message != message:
                    existing.message = message
                    existing.timestamp = now
            else:
                item = SpeechItem(message, obj_id, now)
                self.queue.append(item)
                if obj_id:
                    self.obj_id_map[obj_id] = item

    def play_next_if_available(self):
        """Pop and speak the oldest item if present."""
        item = None
        with self.lock:
            if self.queue:
                item = self.queue.popleft()
                if item.obj_id in self.obj_id_map:
                    del self.obj_id_map[item.obj_id]

        if item:
            try:
                self.engine.say(item.message)
                self.engine.runAndWait()
                print(f"[✅ Speech] {item.message}")
            except Exception as e:
                print(f"[Speech Error] {e}")

    def _process_queue(self):
        while True:
            item = None
            with self.lock:
                now = time.time()
                while self.queue and (now - self.queue[0].timestamp > self.max_age):
                    old = self.queue.popleft()
                    if old.obj_id in self.obj_id_map:
                        del self.obj_id_map[old.obj_id]
                if self.queue:
                    item = self.queue.popleft()
                    if item.obj_id in self.obj_id_map:
                        del self.obj_id_map[item.obj_id]

            if item:
                try:
                    self.engine.say(item.message)
                    self.engine.runAndWait()
                    print(f"[✅ Speech] {item.message}")
                except Exception as e:
                    print(f"[Speech Error] {e}")
            else:
                time.sleep(0.1)
