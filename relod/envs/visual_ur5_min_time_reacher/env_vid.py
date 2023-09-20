import os
import cv2
import time
import multiprocessing
import threading

from screeninfo import get_monitors
import numpy as np


class VideoPlayer:
    def __init__(self, mode='video_easy_5'):
        self._mode = mode
        video_dir = os.path.join('datasets_augmentation', mode)
        self.video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(5)]
        self.current_video_index = 0
        self._video_switch = multiprocessing.Value('i', 0)
        self.video_process = None
        
        # Get screen dimensions
        monitor = get_monitors()[0]  # Assuming you want the primary monitor
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        # Inits
        self.dot_position = None
        self.radius = 84
        self.margin = 30
        self.color = (0, 0, 255)  # Red in BGR format
        self.switch = multiprocessing.Value('i', 0)
        
        # Load all videos
        self.all_videos = []
        for video_path in self.video_paths:
            video = []
            cap = cv2.VideoCapture(video_path)
            while True:
                tic = time.time()
                ret, frame = cap.read()
                # Check if the video has reached the end
                if not ret:
                    break
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                video.append(frame)
            self.all_videos.append(video[:])
            
            
    def play_video(self, video_ind):
        # Create a full-screen window
        cv2.namedWindow('Video Playback', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Video Playback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for frame in self.all_videos[video_ind]:
            # Draw a red dot on the frame
            cv2.circle(frame, self.dot_position, self.radius, self.color, -1)  # -1 thickness fills the circle
            
            cv2.imshow('Video Playback', frame)
            if self._video_switch.value == 1:
                break

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

    def switch_to_next_video(self):
        if self.video_process:
            self._video_switch.value = 1
            self.video_process.join()
            self._video_switch.value = 0

        # Toggle through videos
        self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
        
        # Generate a random position for the dot
        scaled_dot_x = np.random.randint(low=self.radius, high=self.screen_width-self.radius)
        scaled_dot_y = np.random.randint(low=self.radius, high=self.screen_height-self.radius)
        self.dot_position = (scaled_dot_x, scaled_dot_y)

        self.video_process = multiprocessing.Process(target=self.play_video, args=(self.current_video_index,))
        self.video_process.start()

    def run(self):
        while True:
            self.switch_to_next_video()
            while not self.switch.value:
                time.sleep(0.2)

            if self.switch.value == 2:
                self._video_switch.value = 1
                break
            
            self.switch.value = 0
            
        print("Done playing videos on the monitor")
            

if __name__ == "__main__":
    player = VideoPlayer()
    t = threading.Thread(target=player.run)
    t.start()

    tic = start = time.time()
    while time.time() - start < 30:
        if time.time() - tic > 4:
            player.switch.value = 1
            tic = time.time()
