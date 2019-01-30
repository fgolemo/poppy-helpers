import time

import numpy as np

from poppy_helpers.controller import ZMQController

zmq = ZMQController("flogo3.local")
zmq.compliant(False)
zmq.set_max_speed(100)


zmq.goto_pos([0, -70, 50, 0, 15, 80])

time.sleep(2)
zmq.set_max_speed(500)
zmq.goto_pos([0, 80, -90, 0, -10, 90])
time.sleep(1)
zmq.goto_pos([0, 80, -90, 45, -10, 90])
time.sleep(.5)
zmq.goto_pos([0, 80, -90, -45, -10, 90])
time.sleep(1)
zmq.set_max_speed(100)
zmq.goto_pos([0, -70, 50, 0, 15, 80])


