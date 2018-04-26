import os
import time
import gym
import numpy as np
from gym import spaces
import logging

from poppy_helpers.constants import JOINT_LIMITS, REST_POS, SWORDFIGHT_RANDOM_NOISE, MOVE_EVERY_N_STEPS
from pytorch_a2c_ppo_acktr.inference import Inference
from skimage.transform import resize

class ErgoFightLiveEnv(gym.Env):
    def __init__(self, no_move=False, scaling=1, shield=True):
        self.no_move = no_move
        self.scaling = scaling
        self.shield = shield

        self.step_in_episode = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        joint_boxes = spaces.Box(low=-1, high=1, shape=(6,))

        # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
        all_joints = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6,))
        self.observation_space = all_joints

        self.action_space = joint_boxes

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _startEnv(self, headless):
        #TODO: launch windows

    def _move_robot(self, configuration, robot=0):
        req = {"robot": {"set_pos": {"pos": [0, 0, 0, 0, 0, 0]}}}
        socket.send_json(req)
        answer = socket.recv_json()
        #TODO: test response

    def _restPos(self):
        self.done = False

        self._move_robot(REST_POS, robot=0)
        self._move_robot(REST_POS, robot=1)

        #TODO: potentially sleep here for a second until target reached

        self.randomize(robot=1, scaling=self.scaling)

        #TODO: potentially sleep here for a second until in random pos

    def randomize(self, robot=1, scaling=1.0):
        new_pos = [REST_POS[i] + scaling * np.random.randint(
                low=SWORDFIGHT_RANDOM_NOISE[i][0],
                high=SWORDFIGHT_RANDOM_NOISE[i][1],
                size=1)[0] for i in range(6)]
        self._move_robot(new_pos, robot=robot)

    def _reset(self):
        self.step_in_episode = 0
        self._restPos()
        self._self_observe()
        return self.observation

    def _get_reward(self):

        #TODO: implement the collision checking here
        reward = 0
        if check_collision():
            reward = 1
            if not self.fencing_mode:
                self.frames_after_hit = 0
            else:
                self._restPos()  # if fencing mode then reset pos on each hit

        # the following bit is for making sure the robot doen't just hit repeatedly
        # ...so the invulnerability countdown only start when the collision is released
        else:  # if it's not hitting anything right now
            if self.frames_after_hit >= 0:
                self.frames_after_hit += 1
            if self.frames_after_hit >= INVULNERABILITY_AFTER_HIT:
                self.frames_after_hit = -1

        if self.defence:
            reward *= -1

        return reward

    def _get_robot_posvel(self, robot_id):
        req = {"robot": {"get_pos_speed": {}}}
        socket.send_json(req)
        answer = socket.recv_json()

        #TODO: check return value/format

        return answer

    def _self_observe(self):
        joint_vel_att = self._get_robot_posvel(0)
        joint_vel_def = self._get_robot_posvel(1)
        self.observation = np.hstack((joint_vel_att, joint_vel_def)).astype('float32')


    def _normalize(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        return out

    def _denormalize(self, actions):
        out = []
        for i in range(6):
            shifted = (actions[i] + 1) / 2  # now it's within [0,1]
            denorm = shifted * self.diffs[i] + JOINT_LIMITS[i][0]
            out.append(denorm)
        return out

    def prep_actions(self, actions):
        actions = np.clip(actions, -1, 1)  # first make sure actions are normalized
        actions = self._denormalize(actions)  # then scale them to the actual joint angles
        return actions

    def _step(self, actions):
        self.step_in_episode += 1
        actions = self.prep_actions(actions)

        self._move_robot(actions, robot=0)

        if not self.no_move:
            if self.step_in_episode % MOVE_EVERY_N_STEPS == 0:
                self.randomize(1, scaling=self.scaling)

        # observe again
        self._self_observe()
        reward = self._get_reward()

        #TODO: here make sure it's not going faster than max framerate

        return self.observation, reward, self.done, {}

    def _close(self):
        self.venv.stop_simulation()
        self.venv.end()

    def _render(self, mode='human', close=False):
        # This intentionally does nothing and is only here for wrapper functions.
        # if you want graphical output, use the environments
        # "ErgoBallThrowAirtime-Graphical-Normalized-v0"
        # or
        # "ErgoBallThrowAirtime-Graphical-v0"
        # ... not the ones with "...-Headless-..."
        pass

    if __name__ == '__main__':
        pass
        #TODO

