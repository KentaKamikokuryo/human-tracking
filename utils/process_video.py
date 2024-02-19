import os
import cv2
from pathlib import Path
from decord import VideoReader
from decord import cpu


class ProcessVideoDecord:

    def __init__(self, video_path):

        self.video_path = video_path
        self.video_name = Path(self.video_path).stem

        assert os.path.exists(self.video_path), f"Video not found at {self.video_path}"

        vr = VideoReader(self.video_path, ctx=cpu(0))
        self.n_frame_total = len(vr)
        self.frame_rate = vr.get_avg_fps()
        self.duration = self.n_frame_total / self.frame_rate

    def read_frames(self, frames: list, resize_param: int = None):

        self.data_frame_dict = {}

        print("Reading video with Decord for " + str(len(frames)) + " frame at: " + self.video_path)
        print("Please wait ...")

        self.cap = VideoReader(self.video_path, ctx=cpu(0))

        for frame in frames:
            self.data_frame_dict[frame] = self.cap[frame].asnumpy()

        # Convert to dictionary
        for frame in self.data_frame_dict.keys():

            image = self.data_frame_dict[frame]
            self.image_height, self.image_width, _ = image.shape

            if resize_param:
                image_height, image_width = image.shape[:2]

                size = (image_width * resize_param, image_height * resize_param)
                image = cv2.resize(image, size)

            self.data_frame_dict[frame] = image

        print("Reading video finished ...")

        return self.data_frame_dict

    def read(self, n_frame: int = 100, resize_param: int = None):

        if n_frame > self.n_frame_total:
            self.n_frame = self.n_frame_total
        else:
            self.n_frame = n_frame

        frames = [i for i in range(self.n_frame)]

        self.data_frame_dict = self.read_frames(frames=frames, resize_param=resize_param)

        return self.data_frame_dict