import time

import numpy as np

from poppy_helpers.controller import ZMQController

time.sleep(5)

zmq = ZMQController("pokey.local")
zmq.compliant(False)
# zmq.compliant(True)
zmq.set_max_speed(100)
zmq.goto_pos([45, -90, -45])
time.sleep(2)
zmq.set_max_speed(100)
time.sleep(.5)
zmq.goto_pos([-45, -45, 0])
