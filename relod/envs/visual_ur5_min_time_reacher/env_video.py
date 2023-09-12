import cv2
import matplotlib.pyplot as plt

### file to try to play video in the background of the red dot


# Open the video
cap = cv2.VideoCapture('/mnt/c/Users/s136407/Desktop/video17.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Get the video's frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
speed_factor = 1.0  # adjust this to change playback speed
pause_duration = 1 / (fps * speed_factor)

# Create a figure and axis to display the video
fig, ax = plt.subplots()
ax.axis('off')  # Turn off axis numbers and ticks
im = ax.imshow(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

def update(frame):
    ret, frame = cap.read()
    if ret:
        im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return im,

# Use FuncAnimation to update the figure with video frames
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=None, interval=pause_duration*1000, repeat=False)

plt.show(block=True)

# Release the video capture when done
cap.release()
