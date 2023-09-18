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
        self.radius = 80
        self.margin = 30
        self.color = (0, 0, 255)  # Red in BGR format
        self.switch = multiprocessing.Value('i', 0)

    def play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        # Create a full-screen window
        cv2.namedWindow('Video Playback', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Video Playback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to fit the screen resolution
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))

             # Draw a red dot on the frame
            cv2.circle(frame, self.dot_position, self.radius, self.color, -1)  # -1 thickness fills the circle


            cv2.imshow('Video Playback', frame)

            if self._video_switch.value == 1:
                break

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def switch_to_next_video(self):
        if self.video_process:
            self._video_switch.value = 1
            self.video_process.join()
            self._video_switch.value = 0

        # Generate a random position for the dot
        scaled_dot_x = np.random.randint(low=self.radius, high=self.screen_width-self.radius)
        scaled_dot_y = np.random.randint(low=self.radius, high=self.screen_height-self.radius)
        self.dot_position = (scaled_dot_x, scaled_dot_y)

        # Toggle through videos
        self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
        video_path = self.video_paths[self.current_video_index]
        self.video_process = multiprocessing.Process(target=self.play_video, args=(video_path,))
        self.video_process.start()

    def run(self):
        while True:
            self.switch_to_next_video()
            while not self.switch.value:
                time.sleep(0.1)
            self.switch.value = 0

            if self.switch.value == 2:
                break
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




# if __name__ == "__main__":
#     # videos = ['/mnt/c/Users/s136407/Desktop/video17.mp4',]
#     mode = 'video_hard'
#     player = VideoPlayerWithDot(mode=mode)

#     start_time = time.time()
#     quit_pressed = False

#     while not quit_pressed:
#         quit_pressed = player.play_frame_with_dot()

#         # Check if 2 seconds have passed since the last reset
#         if time.time() - start_time >= 30:
#             player.reset_env()
#             start_time = time.time()

#     player.cap.release()
#     cv2.destroyAllWindows()
