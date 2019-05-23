import time

import numpy as np

from poppy_helpers.controller import ZMQController

zmq = ZMQController("flogo3.local")
# zmq.compliant(True)
zmq.compliant(False)
zmq.set_max_speed(100)
# zmq.goto_pos([0, -30, 0, 0, -40, 0]) # no gripper
# zmq.goto_pos([0, -30, 0, 0, -40, 30]) # gripper open
# zmq.goto_pos([0, 20, -30, 0, -30, -60]) # gripper closed
zmq.goto_pos([0, 0, 0, 0, 0, 0]) # rest
# for i in range(6):
#     zmq.set_color(i+1,"green")



# Duckie-tracker
# python pytorch_a2c_ppo_acktr/enjoy.py --env-name ErgoReacher-Tracking-Live-v1 --custom-gym poppy_helpers --model ./trained_models/ppo/ErgoReacher-Headless-Gripper-MobileGoal-v1-190404011143-983040.pt