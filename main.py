import cv2
import time
import threading
import queue
import numpy as np
from config import *
from detection import ObjectDetector
from depth_estimation import DepthEstimator
from geometry import bbox_center, pixel_to_3d, calculate_distance
from exporter import export_to_pickle, export_to_json
from utils import draw_bbox
from video_stream import VideoStream
from blueprint import BlueprintMapper

# ─────────────────────────────────────────────
# TUNING CONSTANTS  (adjust to taste)
# ─────────────────────────────────────────────
SCALE_FACTOR        = 0.37
DEPTH_SCALE         = 0.01
CALIBRATION_FACTOR  = 0.5
DETECT_EVERY_N      = 6          # run YOLO every N frames
DEPTH_EVERY_N       = 18          # run depth model every N frames (heavier)
MAX_SPATIAL_HISTORY = 300        # keep only last 300 frames in memory
MOTION_THRESH       = 25.0       # mean-abs-diff that counts as "camera moved fast"
TABLE_CLASSES       = {63, 66}


# ─────────────────────────────────────────────
# FIX 1 ── VideoStream that ALWAYS gives the
#           LATEST frame (drops stale frames)
# ─────────────────────────────────────────────
class LatestFrameStream:
    """
    Drops every buffered frame and always returns the newest one.
    This is the core fix for the 20-second pan lag.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # OS buffer = 1 frame only
        self._frame = None
        self._ret   = False
        self._lock  = threading.Lock()
        self._stop  = False

    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(0.3)
        return self

    def _reader(self):
        while not self._stop:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._stop = True
        self.cap.release()


# ─────────────────────────────────────────────
# FIX 2 ── Async depth estimator
#           Runs depth in a background thread;
#           main loop never blocks waiting for it
# ─────────────────────────────────────────────
class AsyncDepthEstimator:
    def __init__(self, estimator):
        self._estimator  = estimator
        self._in_q       = queue.Queue(maxsize=1)   # only 1 pending job
        self._result     = None
        self._lock       = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            frame = self._in_q.get()
            depth = self._estimator.estimate_depth(frame)
            with self._lock:
                self._result = depth

    def submit(self, frame):
        """Non-blocking: drop the job if the worker is still busy."""
        try:
            self._in_q.put_nowait(frame)
        except queue.Full:
            pass   # worker is busy; we'll submit next frame instead

    def get_latest(self):
        with self._lock:
            return self._result


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────
def count_persons(detections):
    return sum(1 for d in detections if d["class_id"] == 0)


def remove_duplicate_detections(detections, iou_thresh=0.5):
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        keep = True
        for f in filtered:
            fx1, fy1, fx2, fy2 = f["bbox"]
            ix1, iy1 = max(x1, fx1), max(y1, fy1)
            ix2, iy2 = min(x2, fx2), min(y2, fy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou   = inter / ((x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - inter + 1e-6)
            if iou > iou_thresh:
                keep = False
                break
        if keep:
            filtered.append(det)
    return filtered


def detect_fast_motion(prev_gray, curr_gray):
    """Returns True when the camera has panned quickly."""
    if prev_gray is None:
        return False
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) > MOTION_THRESH


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    detector       = ObjectDetector(YOLO_MODEL_PATH)
    depth_est_raw  = DepthEstimator()
    async_depth    = AsyncDepthEstimator(depth_est_raw)   # FIX 2
    mapper         = BlueprintMapper()

    # FIX 1 — use the always-latest stream
    stream = LatestFrameStream(VIDEO_PATH).start()

    spatial_data      = []
    frame_count       = 0
    last_detections   = []
    last_depth_map    = None
    previous_depths   = {}
    prev_gray         = None
    distance_log      = {}   # FIX 4 — avoid print spam; store last distances

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # ── resize once ──────────────────────────────────────────────────
        if frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        # ── motion detection (grayscale, cheap) ──────────────────────────
        curr_gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fast_motion = detect_fast_motion(prev_gray, curr_gray)
        prev_gray   = curr_gray

        # FIX 3 — on fast pan, immediately clear stale depth so we don't
        #          blend a wrong old map with 80 % weight for many frames
        if fast_motion:
            last_depth_map  = None
            last_detections = []
            previous_depths = {}

        frame_objects = []

        # ── YOLO detection ───────────────────────────────────────────────
        if frame_count % DETECT_EVERY_N == 0:
            try:
                detections      = detector.detect(frame)
                last_detections = detections
            except Exception:
                detections = last_detections
        else:
            detections = last_detections

        detections = remove_duplicate_detections(detections)

        # ── async depth: submit a job, grab whatever is ready ────────────
        if frame_count % DEPTH_EVERY_N == 0 and len(detections) > 0:
            async_depth.submit(frame)           # non-blocking

        fresh_depth = async_depth.get_latest()  # non-blocking

        if fresh_depth is not None:
            if last_depth_map is None or fast_motion:
                # After a pan, accept the new map immediately (no blending)
                last_depth_map = fresh_depth
            else:
                # Gentle temporal smoothing only when scene is stable
                last_depth_map = 0.6 * last_depth_map + 0.4 * fresh_depth

        depth_map = last_depth_map

        # ── per-object depth + 3-D position ─────────────────────────────
        if depth_map is not None:
            for det in detections:
                bbox              = det["bbox"]
                x_center, y_center = bbox_center(bbox)
                x_center = int(np.clip(x_center, 0, depth_map.shape[1] - 1))
                y_center = int(np.clip(y_center, 0, depth_map.shape[0] - 1))

                patch = depth_map[
                    max(0, y_center - 2):y_center + 3,
                    max(0, x_center - 2):x_center + 3
                ]
                depth = float(np.median(patch)) * DEPTH_SCALE * CALIBRATION_FACTOR
                # soft depth limiting
                depth = max(depth, 0.2)

                # compress unrealistic spikes smoothly
                if depth > 2.5:
                    depth = 2.5 + (depth - 2.5) * 0.2

                track_id = det.get("track_id", None)
                if track_id is not None:
                    if track_id in previous_depths:
                        depth = 0.7 * previous_depths[track_id] + 0.3 * depth
                    previous_depths[track_id] = depth

                X, Y, Z = pixel_to_3d(x_center, y_center, depth, FX, FY, CX, CY)

                frame_width = frame.shape[1]
                if (x_center / frame_width) < 0.05 or (x_center / frame_width) > 0.95:
                    continue

                object_info = {
                    "class_id":    det["class_id"],
                    "confidence":  det["confidence"],
                    "position_3d": [float(X), float(Y), float(depth)]
                }
                frame_objects.append(object_info)

        # ── table grouping ───────────────────────────────────────────────
        table_objects = [o for o in frame_objects if o["class_id"] in TABLE_CLASSES]
        if len(table_objects) > 1:
            zs = [o["position_3d"][2] for o in table_objects]
            if max(zs) - min(zs) < 1.0:
                avg_z = float(np.median(zs))
                for o in table_objects:
                    o["position_3d"][2] = avg_z

        # FINAL LABEL DRAWING (AFTER ALL DEPTH FIXES)

        for idx, obj in enumerate(frame_objects):

            final_z = obj["position_3d"][2]

            if idx < len(detections):

                bbox = detections[idx]["bbox"]
                track_id = detections[idx].get("track_id", idx)

                label = f"{final_z:.2f}m"

                draw_bbox(frame, bbox, label, track_id)

        # ── distance calculation  (FIX 4: no print spam) ─────────────────
        for i in range(len(frame_objects)):
            for j in range(i + 1, len(frame_objects)):
                d = calculate_distance(
                    frame_objects[i]["position_3d"],
                    frame_objects[j]["position_3d"]
                )
                distance_log[(i, j)] = d   # stored, not printed every frame

        # Print distances only every 30 frames to avoid I/O bottleneck
        if frame_count % 30 == 0 and distance_log:
            for key, d in distance_log.items():
                print(f"Distance {key[0]}-{key[1]}: {d:.2f} m")
            distance_log.clear()

        # ── memory guard (FIX 5) ─────────────────────────────────────────
        spatial_data.append(frame_objects)
        if len(spatial_data) > MAX_SPATIAL_HISTORY:
            spatial_data.pop(0)

        # ── render ───────────────────────────────────────────────────────
        blueprint_map = mapper.create_map(frame_objects)
        cv2.imshow("Drone Vision System", frame)
        cv2.imshow("3D Blueprint", blueprint_map)

        # FIX 6 — removed time.sleep(0.015); let the loop run as fast as it can
        if cv2.waitKey(1) == 27:
            break

    stream.stop()
    cv2.destroyAllWindows()
    export_to_pickle(spatial_data, OUTPUT_PICKLE)
    export_to_json(spatial_data, OUTPUT_JSON)


if __name__ == "__main__":
    main()