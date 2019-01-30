import time

import numpy as np

from poppy_helpers.controller import ZMQController

zmq = ZMQController("flogo3.local")
zmq.compliant(False)
zmq.set_max_speed(100)
zmq.goto_pos([0, -30, 0, 0, -40, 0])

        