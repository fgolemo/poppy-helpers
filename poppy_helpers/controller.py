import numpy as np
import time

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
