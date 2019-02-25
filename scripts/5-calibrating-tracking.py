import os
import time

import numpy as np
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker

from poppy_helpers.config import config_dir
from poppy_helpers.controller import ZMQController

ROBOT = False

cam = Camera(color=True)
tracker = Tracker((54, 68, 11), (92, 255, 224))

if ROBOT:
    zmq = ZMQController("flogo3.local")
    zmq.compliant(False)
    zmq.set_max_speed(100)

    # rest
    zmq.goto_pos([0, 0, 0, 0, 0, 0])
    time.sleep(2)

# max forward   posx: 0.224 posy: 0.102,    [1, -1, 0, 0]
# max backward  posx: -0.148 posy: 0.016,   [-1, -1, 0, 0]
# max up        posx: 0.013	posy: 0.25,     [.1, -1, -.1, 0]

calibration = np.zeros((3, 2), dtype=np.uint16)

# full forward

if ROBOT:
    zmq.goto_normalized([0, 1, -1, 0, 0, 0])
print("forward, 2... 1...")
time.sleep(2)

while True:
    center, radius, x, y = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("forward: ", center, x, y)
calibration[0] = center

# full backward
if ROBOT:
    zmq.goto_normalized([0, -1, -1, 0, 0, 0])
print("backward, 2... 1...")
time.sleep(2)

while True:
    center, radius, x, y = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("backward: ", center, x, y)
calibration[1] = center

# upward
if ROBOT:
    zmq.goto_normalized([0, .1, -1, 0, -.1, 0])
print("upward, 2... 1...")
time.sleep(2)

while True:
    center, radius, x, y = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("upward: ", center, x, y)
calibration[2] = center

np.savez(os.path.join(config_dir(), "calib.npz"), calibration=calibration)
