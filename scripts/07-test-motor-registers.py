import time

import numpy as np
import zmq
from tqdm import trange, tqdm

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://{}:{}".format("flogo3.local", 5757))

req = {"robot": {"get_motor_registers_list": {"motor": "m1"}}}
socket.send_json(req)
a = socket.recv_json()

print(a)

registers = [
    'present_load', 'present_temperature', 'moving_speed', 'present_speed',
    'present_voltage', 'present_position'
]

TRIALS = 1000

out = []

for r in registers:
    req = {"robot": {"get_register_value": {"motor": "m1", "register": r}}}

    start = time.time()

    for t in trange(TRIALS):
        socket.send_json(req)
        a = socket.recv_json()

    diff = (time.time() - start) / TRIALS

    out.append(f"Register '{r}': {np.around(1/diff/6,2)}Hz")

tqdm.write("\n")
print()

for o in out:
    print (o)

# Register 'present_load': 206.8Hz
# Register 'present_temperature': 75.82Hz
# Register 'moving_speed': 201.78Hz
# Register 'present_speed': 229.05Hz
# Register 'present_voltage': 86.51Hz
# Register 'present_position': 162.9Hz


