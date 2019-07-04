import numpy as np
import time

import zmq
from poppy_helpers.constants import REST_POS, SWORDFIGHT_REST_ATT, SWORDFIGHT_REST_DEF, REST_HALF_POS


class Controller(object):
    def __init__(self, robot):
        self.robot = robot
        self.last_pos = []
        self.last_vel = []

    def goto_pos(self, pos):
        for i in range(6):
            self.robot.motors[i].goal_position = pos[i]

    def get_pos(self):
        pos = [m.present_position for m in self.robot.motors]
        self.last_pos = pos
        return pos

    def rest(self):
        self.goto_pos(REST_POS)

    def safe_rest(self):
        self.goto_pos(REST_HALF_POS)
        time.sleep(1)
        self.goto_pos(REST_POS)

    def set_max_speed(self, speed):
        for i in range(6):
            self.robot.motors[i].moving_speed = speed

    def get_max_speed(self):
        return [m.moving_speed for m in self.robot.motors]  # maximum is 150

    def get_current_speed(self):
        vel = [m.present_speed for m in self.robot.motors]
        self.last_vel = vel
        return vel

    def compliant(self, compliant):
        for m in self.robot.motors:
            m.compliant = compliant

    def act(self, action, scaling=0.1, cached=True):
        last_pos = self.last_pos
        if not cached or len(self.last_pos) == 0:
            last_pos = self.get_pos()

        pos = [last_pos[i] + scaling*(action[i] - last_pos[i]) for i in range(6)]
        self.goto_pos(pos)


class SwordFightController(Controller):
    def __init__(self, robot, mode):
        assert (mode in ["att", "def"])

        super().__init__(robot)
        self.mode = mode
        if mode == "att":
            self.rest_pos = SWORDFIGHT_REST_ATT
        else:
            self.rest_pos = SWORDFIGHT_REST_DEF

    def get_pos_comp(self): # compensated for different starting pos
        return [p - self.rest_pos[i] for i, p in enumerate(super().get_pos())]

    def compensate_pos(self, pos):
        return [self.rest_pos[i] + pos[i] for i in range(6)]

    def goto_pos(self, pos):
        super().goto_pos(self.compensate_pos(pos))

class SwordFightZMQController(SwordFightController):
    def __init__(self, mode, host, port=5757):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        print("Connecting to robot...")
        self.socket.connect("tcp://{}:{}".format(host, port))
        print("Connected.")

        super().__init__(None, mode)

    def _check_answer(self, answer, function):
        if answer: # answer is dict, empty dicts are False
            print ("ERROR: in ",function,answer)

    def goto_pos(self, pos):
        pos = self.compensate_pos(pos)

        req = {"robot": {"set_pos": {"positions": pos}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "goto_pos")

    def get_pos(self):
        req = {"robot": {"get_pos_speed": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer[:6] # the other 6 values in this list are the angular velocities

    def get_posvel(self):
        req = {"robot": {"get_pos_speed": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer

    def compliant(self, trueorfalse):
        req = {"robot": {"set_compliant": {"trueorfalse": trueorfalse}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "compliant")

    def set_max_speed(self, max_speed):
        req = {"robot": {"set_max_speed": {"max_speed": max_speed}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "set_max_speed")

    def get_keys(self):
        req = {"robot": {"get_keys": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer


class ZMQController():
    def __init__(self, host, port=5757):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        print("Connecting to robot...")
        self.socket.connect("tcp://{}:{}".format(host, port))
        print("Connected.")

    def _check_answer(self, answer, function):
        if answer and "unicode not allowed, use send_string" != answer["error"]: # answer is dict, empty dicts are False
            print ("ERROR: in ",function,answer)

    def goto_pos(self, pos):
        req = {"robot": {"set_pos": {"positions": pos}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "goto_pos")

    def goto_normalized(self, pos):
        pos = np.array(pos)
        pos[2] = np.clip(pos[2], a_min=max(-1-pos[1], -1), a_max=1) # to prevent robot from hitting the case

        self.goto_pos((pos*90).tolist())

    def get_pos(self):
        req = {"robot": {"get_pos_speed": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer[:6] # the other 6 values in this list are the angular velocities

    def get_posvel(self):
        req = {"robot": {"get_pos_speed": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer

    def compliant(self, trueorfalse):
        req = {"robot": {"set_compliant": {"trueorfalse": trueorfalse}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "compliant")

    def set_max_speed(self, max_speed):
        req = {"robot": {"set_max_speed": {"max_speed": max_speed}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "set_max_speed")

    def set_color(self, motor_id, color):
        req = {"robot": {"set_register_value": {"motor": "m{}".format(motor_id), "register": "led", "value": color}}}
        self.socket.send_json(req)
        self._check_answer(self.socket.recv_json(), "set_color")

    def rest(self):
        self.goto_pos([0]*6)

    def get_keys(self):
        req = {"robot": {"get_keys": {}}}
        self.socket.send_json(req)
        answer = self.socket.recv_json()
        return answer
    
    def safe_rest(self):
        self.goto_pos(REST_HALF_POS)
        time.sleep(1)
        self.goto_pos(REST_POS)



