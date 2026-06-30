import cv2
import threading
import time


class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

        # Reduce internal OpenCV buffering
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Optional: lower resolution directly at capture stage
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Optional: reduce FPS to lower CPU/network load
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        self.ret, self.frame = self.cap.read()

        self.stopped = False
        self.lock = threading.Lock()

        # Frame timing
        self.last_frame_time = time.time()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:

            # IMPORTANT:
            # Grab newest frame and discard old buffered frames
            self.cap.grab()

            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame
                self.last_frame_time = time.time()

    def read(self):
        with self.lock:

            if self.frame is None:
                return False, None

            # Return newest available frame only
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True

        # Small delay for safe thread shutdown
        time.sleep(0.1)

        self.cap.release()

