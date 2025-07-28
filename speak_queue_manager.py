import threading
import time
import pyttsx3
from collections import deque

class SpeechItem:
    def __init__(self, message, obj_id=None, timestamp=None):
        self.message = message
        self.obj_id = obj_id  # Optional: Track object by ID
        self.timestamp = timestamp or time.time()

class SpeechQueueManager:
    def __init__(self, max_age_seconds=5.0):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.queue = deque()
        self.obj_id_map = {}  # obj_id -> SpeechItem
        self.max_age = max_age_seconds
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()

        # Set Chinese voice if available
        for voice in self.engine.getProperty('voices'):
            if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

    def enqueue(self, message, obj_id=None):
        now = time.time()
        with self.lock:
            if obj_id and obj_id in self.obj_id_map:
                # Update existing item (only if message changed or newer)
                existing = self.obj_id_map[obj_id]
                if existing.message != message:
                    existing.message = message
                    existing.timestamp = now
            else:
                item = SpeechItem(message, obj_id, now)
                self.queue.append(item)
                if obj_id:
                    self.obj_id_map[obj_id] = item

    def _process_queue(self):
        while True:
            with self.lock:
                # 清除過時項目
                now = time.time()
                while self.queue and (now - self.queue[0].timestamp > self.max_age):
                    old_item = self.queue.popleft()
                    if old_item.obj_id and old_item.obj_id in self.obj_id_map:
                        del self.obj_id_map[old_item.obj_id]

                if self.queue:
                    item = self.queue.popleft()
                    if item.obj_id in self.obj_id_map:
                        del self.obj_id_map[item.obj_id]
                else:
                    item = None

            if item:
                try:
                    self.engine.say(item.message)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"[Speech Error] {e}")
            else:
                time.sleep(0.1)  # Avoid busy-wait