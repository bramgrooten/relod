import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random


class MonitorTargetVideoBackground:
    def __init__(self, video_paths):
        self.radius = 7
        self.width = 160
        self.height = 90
        self.margin = 20

        # Handling multiple videos
        self.video_paths = video_paths
        # random.shuffle(self.video_paths)
        self.current_video_idx = 0

        # Open the video
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])

        # Check if video opened successfully
        if not self.cap.isOpened():
            print("Error: Couldn't open the video.")
            raise FileNotFoundError(self.video_paths[self.current_video_idx])

        # Get the video's frame rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.speed_factor = 0.5    # adjust this to change playback speed
        self.pause_duration = 1 / (fps * self.speed_factor)

        # Create a figure and axis to display the video
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')  # Turn off axis numbers and ticks
        self.im = self.ax.imshow(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), zorder=1)  # zorder ensures the image is behind the dot

        self.target = plt.Circle((0, 0), self.radius, color='red', zorder=2)
        self.ax.add_patch(self.target)

        # Use FuncAnimation to update the figure with video frames
        self.ani = FuncAnimation(self.fig, self.update_video_frame, frames=self.frame_generator, interval=self.pause_duration * 1000, repeat=False)
        plt.show(block=False)

        # # Turn on the interactive mode
        # plt.ion()
        # self.fig.show()

    def reset_plot(self):
        x, y = np.random.random(2)

        self.target.set_center(
            (self.radius + self.margin + x * (self.width - 2 * self.radius - 2 * self.margin),
             self.radius + self.margin + y * (self.height - 2 * self.radius - 2 * self.margin))
        )

        # Switch to the next video
        self.switch_video()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def switch_video(self):
        # Release the current video capture
        self.cap.release()

        # Go to the next video or shuffle and restart from the beginning
        self.current_video_idx += 1
        if self.current_video_idx >= len(self.video_paths):
            # random.shuffle(self.video_paths)
            self.current_video_idx = 0

        # Open the next video
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])

        # Adjust the playback speed if required
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.pause_duration = 1 / (fps * self.speed_factor)

    def frame_generator(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.switch_video()
                continue
            yield frame

    def update_video_frame(self, frame):
        self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.fig.canvas.flush_events()  # Ensure the canvas is updated
        return self.im, self.target


if __name__ == '__main__':
    monitor = MonitorTargetVideoBackground(['/mnt/c/Users/s136407/Desktop/video17.mp4'])
    while True:
        monitor.reset_plot()
        time.sleep(2)
