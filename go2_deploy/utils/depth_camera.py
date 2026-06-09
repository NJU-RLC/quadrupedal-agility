import threading
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch


class DepthCamera(threading.Thread):
    """RealSense depth stream preprocessor for the high-level policy."""

    def __init__(self, depth_width=848, depth_height=480, depth_fps=60, record_video=True, device="cuda"):
        super().__init__(daemon=True)
        import pyrealsense2 as rs

        DepthCleanerCPP = self._load_depth_cleaner()

        self.rs = rs
        self.device = torch.device(device)
        self.pipeline = rs.pipeline()
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 8)
        self.depth_cleaner = DepthCleanerCPP.DepthProcess(108, 60)
        self.depth_fps = depth_fps
        self.record_video = record_video
        self.running = True
        self.video_recording = False
        self.depth_torch = torch.zeros((1, 58, 87), device=self.device, dtype=torch.float16)
        self.camera_time = None
        self.far_clip = 4.0
        self.low_clip = 0.3

        config = rs.config()
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
        profile = self.pipeline.start(config)
        profile.get_device().first_depth_sensor().set_option(rs.option.laser_power, 360)
        self.tlast = time.time()
        self.start()

    @staticmethod
    def _load_depth_cleaner():
        try:
            import PyDepthInpaint as depth_cleaner

            return depth_cleaner
        except ModuleNotFoundError:
            candidates = sorted(Path(__file__).resolve().parent.glob("PyDepthInpaint*.so"))
            if not candidates:
                raise
            spec = importlib.util.spec_from_file_location("PyDepthInpaint", candidates[0])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def run(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_frame = self.decimation_filter.process(depth_frame)
            depth_image = np.asarray(depth_frame.get_data(), dtype=np.uint16)
            depth_np = self.process_depth_image(self.depth_cleaner.process(depth_image) / 1000.0)
            if self.record_video and self.video_recording:
                self.video_out.write(np.uint8((depth_np + 0.5) * 255))
            self.depth_torch = torch.tensor(depth_np, device=self.device, dtype=torch.float16).unsqueeze(0)
            now = time.time()
            self.camera_time = now - self.tlast
            self.tlast = now

    def init_video(self, video_path):
        if not self.record_video:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_out = cv2.VideoWriter(str(video_path), fourcc, self.depth_fps, (87, 58), False)
        self.video_recording = True

    def save_video(self):
        if self.record_video and self.video_recording:
            self.video_recording = False
            self.video_out.release()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        self.pipeline.stop()
        self.save_video()

    def process_depth_image(self, depth_image):
        depth_image = depth_image[1:-1, 11:-10]
        depth_image = np.clip(depth_image, self.low_clip, self.far_clip)
        return (depth_image - self.low_clip) / (self.far_clip - self.low_clip) - 0.5
