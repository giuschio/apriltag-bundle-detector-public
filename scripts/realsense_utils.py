import numpy as np
from time import time

import pyrealsense2 as rs

class SceneCamera:
    def __init__(self):
        self.pipeline = None
        self.color_intrinsics = None
        self.color_intrinsics_matrix = None
        self.dist_coeffs = None
        self.latest_color = None
        self.initialize()

    @property
    def width(self):
        return self.color_intrinsics["width"]
    
    @property
    def height(self):
        return self.color_intrinsics["height"]

    def initialize(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        color_sensor = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = {
            'fx': color_sensor.fx,
            'fy': color_sensor.fy,
            'cx': color_sensor.ppx,
            'cy': color_sensor.ppy,
            'width': color_sensor.width,
            'height': color_sensor.height,
        }
        self.color_intrinsics_matrix = np.array(
            [
                [color_sensor.fx, 0.0, color_sensor.ppx],
                [0.0, color_sensor.fy, color_sensor.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.asarray(color_sensor.coeffs, dtype=np.float64)

    def capture(self, debug=False):
        start = time()
        frames = self.pipeline.poll_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            if color_frame:
                self.latest_color = np.asanyarray(color_frame.get_data())
        if self.latest_color is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if debug: print(f"Frame aquisition done in {((time()-start)*1000):.2f}ms")
        return np.copy(self.latest_color)

    def finalize(self):
        if self.pipeline:
            self.pipeline.stop()
