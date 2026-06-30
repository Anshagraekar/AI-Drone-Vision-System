# utils.py

import cv2

def draw_bbox(frame, bbox, label, track_id=None):
    x1, y1, x2, y2 = map(int, bbox)

    text = label
    if track_id is not None:
        text = f"ID {track_id} | {label}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, text, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)