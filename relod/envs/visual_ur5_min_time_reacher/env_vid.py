import cv2
import random
import time


class VideoPlayerWithDot:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.current_video_idx = 0
        self.dot_position = None
        self.reset_env()

        # Setting window properties for fullscreen mode
        cv2.namedWindow('Video with Dot', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video with Dot', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # # Get screen dimensions
        # self.screen_width = cv2.getDisplayProperty('Video with Dot', cv2.WND_PROP_FULLSCREEN_WIDTH)
        # self.screen_height = cv2.getDisplayProperty('Video with Dot', cv2.WND_PROP_FULLSCREEN_HEIGHT)

    def reset_env(self):
        # If cap is already open, release it
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # Open the video to get its dimensions
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])
        _, frame = self.cap.read()
        h, w, _ = frame.shape

        # Generate a random position for the dot
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        self.dot_position = (x, y)

        # Cycle to the next video
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_paths)

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

        # Draw a red dot on the frame
        radius = 7
        color = (0, 0, 255)  # Red in BGR format
        cv2.circle(frame, self.dot_position, radius, color, -1)  # -1 thickness fills the circle

        # Resize frame to screen resolution
        # frame = cv2.resize(frame, (int(self.screen_width), int(self.screen_height)))

        # Display the frame
        cv2.imshow('Video with Dot', frame)

        # Use 'q' to quit the video playback and return True if 'q' was pressed
        return cv2.waitKey(delay) & 0xFF == ord('q')


if __name__ == "__main__":
    videos = [
        '/mnt/c/Users/s136407/Desktop/video17.mp4',
        # Add more video paths here
    ]
    player = VideoPlayerWithDot(videos)

    start_time = time.time()
    quit_pressed = False

    while not quit_pressed:
        quit_pressed = player.play_frame_with_dot()

        # Check if 2 seconds have passed since the last reset
        if time.time() - start_time >= 2:
            player.reset_env()
            start_time = time.time()

    player.cap.release()
    cv2.destroyAllWindows()
