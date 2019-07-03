import time

import numpy as np

from poppy_helpers.controller import ZMQController

zmq = ZMQController("flogo3.local")
zmq.compliant(False) # make robot stiff
zmq.set_max_speed(100)
zmq.goto_pos([0, 0, 0, 0, 0, 0])

for i in range(6):
    action_val = 30
    if i == 5:
        action_val = 20
    action = [0] * (i) + [action_val] + [0] * (6 - i - 1)
    print(action)
    zmq.goto_pos(action)
    time.sleep(1)
    action = (np.array(action) * -1).tolist()
    print(action)
    zmq.goto_pos(action)

    print ("current pos", zmq.get_pos())

    time.sleep(1)

zmq.goto_pos([0, 0, 0, 0, 0, 0])
time.sleep(.5)
