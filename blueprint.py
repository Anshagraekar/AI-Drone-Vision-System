import numpy as np
import cv2


class BlueprintMapper:

    def __init__(self, size=600, scale=55):

        self.size = size
        self.scale = scale

        # indoor realistic visualization range
        self.display_depth = 2.5

    def create_map(self, objects):

        canvas = np.ones(
            (self.size, self.size, 3),
            dtype=np.uint8
        ) * 255

        center_x = self.size // 2
        origin_y = self.size - 50

        # ====================================
        # DRAW DRONE
        # ====================================
        cv2.circle(
            canvas,
            (center_x, origin_y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            canvas,
            "Drone",
            (center_x - 30, origin_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # ====================================
        # DEPTH GRID
        # ====================================
        for d in range(1, 3):

            y = int(
                origin_y
                - (d / self.display_depth) * (self.size - 120)
            )

            cv2.line(
                canvas,
                (60, y),
                (self.size - 60, y),
                (220, 220, 220),
                1
            )

            cv2.putText(
                canvas,
                f"{d}m",
                (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 100, 100),
                1
            )

        # ====================================
        # NO OBJECTS
        # ====================================
        if len(objects) == 0:
            return canvas

        # ====================================
        # DRAW OBJECTS
        # ====================================
        for obj in objects:

            X, Y, Z = obj["position_3d"]

            # IMPORTANT FIX:
            # limit unrealistic MiDaS spikes
            Z = min(Z, self.display_depth)

            # stable ratio
            depth_ratio = Z / self.display_depth

            # map coordinates
            map_x = int(center_x + X * self.scale)

            map_y = int(
                origin_y
                - depth_ratio * (self.size - 120)
            )

            # keep inside window
            map_x = np.clip(map_x, 20, self.size - 20)
            map_y = np.clip(map_y, 20, self.size - 20)

            # ====================================
            # COLOR
            # ====================================
            green = int(255 * (1 - depth_ratio))
            red = int(255 * depth_ratio)

            color = (0, green, red)

            # draw object
            cv2.circle(
                canvas,
                (map_x, map_y),
                7,
                color,
                -1
            )

            cv2.circle(
                canvas,
                (map_x, map_y),
                7,
                (0, 0, 0),
                1
            )

            # ====================================
            # LABEL
            # ====================================
            label = f"{Z:.2f}m"

            cv2.putText(
                canvas,
                label,
                (map_x + 10, map_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1
            )

        return canvas