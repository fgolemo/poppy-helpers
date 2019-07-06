import os
import time

import numpy as np
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker, TRACKING_GREEN

from poppy_helpers.config import config_dir
from poppy_helpers.controller import ZMQController


zmq = ZMQController("pokey.local")
zmq.compliant(False)
zmq.set_max_speed(100)
zmq.goto_pos([0, 0, 0])
print (zmq.get_posvel())
