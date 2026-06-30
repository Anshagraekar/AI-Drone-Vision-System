# depth_estimation.py

import torch
import cv2

class DepthEstimator:
    def __init__(self):
        self.model_type = "MiDaS_small"
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS","transforms")
        self.transform = midas_transforms.small_transform

    def estimate_depth(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb)

        with torch.no_grad():
            prediction = self.model(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        return depth_map
