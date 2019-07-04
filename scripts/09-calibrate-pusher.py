import os
import time

import numpy as np
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker, TRACKING_GREEN

from poppy_helpers.config import config_dir
from poppy_helpers.controller import ZMQController

ROBOT = True

cam = Camera(color=True)
tracker = Tracker(TRACKING_GREEN["pusher_lower"],
                  TRACKING_GREEN["pusher_upper"])

if ROBOT:
    zmq = ZMQController("pokey.local")
    zmq.compliant(False)
    zmq.set_max_speed(100)

    # rest
    zmq.goto_pos([0, 0, 0])
    time.sleep(2)

calibration = np.zeros((3, 2), dtype=np.uint16)

# full right

if ROBOT:
    zmq.goto_pos([90, 0, 0])

input("press enter")

while True:
    _, (center, radius, x, y), _ = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("right: ", center, x, y)
calibration[0] = center

# full middle
if ROBOT:
    zmq.goto_pos([0, 0, 0])

input("press enter")

while True:
    _, (center, radius, x, y), _ = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("middle: ", center, x, y)
calibration[1] = center

# left
if ROBOT:
    zmq.goto_pos([-90, 0, 0])

input("press enter")

while True:
    _, (center, radius, x, y), _ = tracker.get_frame_and_track(cam)
    if center is not None and radius > 10:
        break

print("right: ", center, x, y)
calibration[2] = center

np.savez(os.path.join(config_dir(), "calibration-pusher-real.npz"), calibration)
