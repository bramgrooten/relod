import os
import cv2
import random
import time
from screeninfo import get_monitors
import numpy as np


class VideoPlayerWithDot:
    def __init__(self, video_paths=None, mode='video_hard'):
        self._mode = mode
        if video_paths is None:
            self._get_video_paths()
        else:
            self.video_paths = video_paths
        self.current_video_idx = -1  # Will be incremented to 0 in reset_env()
        self.dot_position = None
        self.radius = 80
        self.margin = 30
        self.color = (0, 0, 255)  # Red in BGR format

        # Setting window properties for fullscreen mode
        cv2.namedWindow('Video with Dot', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video with Dot', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get screen dimensions
        monitor = get_monitors()[0]  # Assuming you want the primary monitor
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        print(f'Screen resolution (w x h): {self.screen_width}x{self.screen_height}')

        # Set first environment
        self.reset_env()

    def _get_video_paths(self):
        video_dir = os.path.join('datasets_augmentation', self._mode)
        if 'video_easy' in self._mode:
            self.video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
        elif 'video_hard' in self._mode:
            self.video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def reset_env(self):
        # Cycle to the next video
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_paths)

        # If cap is already open, release it
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # Open the video to get its dimensions
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])
        _, frame = self.cap.read()
        h, w, _ = frame.shape

        # Generate a random position for the dot
        x, y = np.random.random(2)
        self.dot_position = (self.radius + self.margin + x * (w - 2*self.radius - 2*self.margin),
                             self.radius + self.margin + y * (h - 2*self.radius - 2*self.margin))

        # Compute scaled dot position after resizing the frame
        original_video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scaled_dot_x = int(self.dot_position[0] * (self.screen_width / original_video_width))
        scaled_dot_y = int(self.dot_position[1] * (self.screen_height / original_video_height))
        self.dot_position = (scaled_dot_x, scaled_dot_y)

    def play_frame_with_dot(self):
        # Read frame from video
        ret, frame = self.cap.read()

        if not ret:
            # Restart the video if it has ended
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])
            ret, frame = self.cap.read()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        slow_down_factor = 1.0  # Increasing this factor will make video play slower
        delay = int(1000 / fps * slow_down_factor)

        # Resize frame to screen resolution
        frame = cv2.resize(frame, (int(self.screen_width), int(self.screen_height)))

        # Draw a red dot on the frame
        cv2.circle(frame, self.dot_position, self.radius, self.color, -1)  # -1 thickness fills the circle

        # Display the frame
        cv2.imshow('Video with Dot', frame)

        # Use 'q' to quit the video playback and return True if 'q' was pressed
        return cv2.waitKey(delay) & 0xFF == ord('q')


if __name__ == "__main__":
    # videos = ['/mnt/c/Users/s136407/Desktop/video17.mp4',]
    mode = 'video_hard'
    player = VideoPlayerWithDot(mode=mode)

    start_time = time.time()
    quit_pressed = False

    while not quit_pressed:
        quit_pressed = player.play_frame_with_dot()

        # Check if 2 seconds have passed since the last reset
        if time.time() - start_time >= 30:
            player.reset_env()
            start_time = time.time()

    player.cap.release()
    cv2.destroyAllWindows()
