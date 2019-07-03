import time
import matplotlib.pyplot as plt
import numpy as np
import zmq
from tqdm import trange, tqdm

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://{}:{}".format("flogo3.local", 5757))

req = {"robot": {"set_compliant": {"trueorfalse": False}}}
socket.send_json(req)
print(socket.recv_json())

req = {"robot": {"set_max_speed": {"max_speed": 100}}}
socket.send_json(req)
print(socket.recv_json())

# req = {"robot": {"set_pos": {"positions": [-90, -45, 90, 0, 0, -45]}}}
req = {"robot": {"set_pos": {"positions": [0, 45, 0, 0, 0, 45]}}}
socket.send_json(req)
print(socket.recv_json())
#
# time.sleep(1)
#
# # ==========
#
# req = {"robot": {"set_max_speed": {"max_speed": 10}}}
# socket.send_json(req)
# print(socket.recv_json())
#
# loads = []
# TRIALS = 3000
#
# for i in trange(TRIALS):
#
#     req = {"robot": {"set_pos": {"positions": [80, 45, 0, 0, 0, 45]}}}
#     socket.send_json(req)
#     _ = socket.recv_json()
#
#     #
#     req = {"robot": {"get_load": {}}}
#     socket.send_json(req)
#     loads.append(socket.recv_json())
#
# loads = np.array(loads)
#
# x= np.arange(TRIALS)
#
# for i in range(6):
#     plt.plot(x, loads[:,i], label=f"m{i+1}")
#
# plt.legend()
# plt.show()
