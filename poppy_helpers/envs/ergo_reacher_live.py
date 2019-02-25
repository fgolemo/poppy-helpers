import os
import time
import gym
import numpy as np
from gym import spaces
from gym_ergojr.utils.math import RandomPointInHalfSphere
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker

from poppy_helpers.config import config_dir
from poppy_helpers.constants import JOINT_LIMITS, MOVE_EVERY_N_STEPS, MAX_REFRESHRATE, JOINT_LIMITS_SPEED, \
    SIM_VELOCITY_SCALING
from poppy_helpers.controller import SwordFightZMQController, ZMQController

from poppy_helpers.randomizer import Randomizer


class ErgoReacherLiveEnv(gym.Env):
    def __init__(self):
        self.rand = Randomizer()

        # self.step_in_episode = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        self.rhis = RandomPointInHalfSphere(0.0, 0.0369, 0.0437,
                                            radius=0.2022, height=0.2610,
                                            min_dist=0.1, halfsphere=True)

        # observation = 4 joints + 4 velocities + 2 coordinates for target
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4 + 4 + 2,), dtype=np.float32)  #

        # action = 4 joint angles
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  #

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]

        self.cam = Camera(color=True)
        self.tracker = Tracker((54, 68, 11), (92, 255, 224))
        self.calibration = np.load(os.path.join(config_dir(), "calib.npz"))

        self._init_robot()

        super().__init__()

    def _init_robot(self):
        self.controller = ZMQController(host="flogo3.local")
        self._setSpeedCompliance()

    def _setSpeedCompliance(self):
        self.controller.compliant(False)
        self.controller.set_max_speed(100)

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.goal = self.rhis.sampleSimplePoint()

        qpos = np.random.uniform(low=-0.2, high=0.2, size=6)
        qpos[[0, 3]] = 0

        self.controller.rest()
        time.sleep(.5)
        self.controller.goto_normalized(qpos)

        return self._get_obs()

    def _get_obs(self):
        pv = self.controller.get_posvel()
        self.observation = self._normalize(pv)[[1, 2, 4, 5, 7, 8, 10, 11]]  # leave out joint0/3
        self.observation = np.hstack((self.observation, self.rhis.normalize(self.goal)))
        return self.observation

    def _normalize(self, pos):
        pos = np.array(pos).astype('float32')
        pos[:6] = (pos[:6] + 90 / 180) * 2 - 1  # positions
        pos[6:] = (pos[6:] + 300 / 600) * 2 - 1  # velocities
        return pos

    def _transform_cam(self, camcenter):
        pos = np.array(camcenter)
        x_n = (pos[0] - self.calibration[0, 0]) / (self.calibration[1, 0] - self.calibration[0, 0])
        x_d = .224 - (x_n * (.224 + .148))
        y_n = (pos[1] - self.calibration[2, 1]) / (self.calibration[1, 1] - self.calibration[2, 1])
        y_d = .25 - (y_n * (.25 - .016))
        return np.array([x_d, y_d])

    def _get_reward(self):
        done = False

        while True:
            center, radius, x, y = self.tracker.get_frame_and_track(self.cam)
            if center is not None and radius > 10:
                break

        pos = self._transform_cam(center)
        reward = np.linalg.norm(np.array(self.goal[1:]) - np.array(pos))
        reward *= -1  # the reward is the inverse distance

        if reward > -0.016:  # this is a bit arbitrary, but works well
            done = True
            reward = 1

        return reward, done

    def step(self, action):
        action_ = np.zeros(6, np.float32)
        action_[[1, 2, 4, 5]] = action
        action = np.clip(action_, -1, 1)

        self.controller.goto_normalized(action)

        reward, done = self._get_reward()

        dt = (time.time() - self.last_step_time) * 1000
        if dt < MAX_REFRESHRATE:
            time.sleep((MAX_REFRESHRATE - dt) / 1000)

        self.last_step_time = time.time()
        return self.observation, reward, done, {}

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    env = gym.make("ErgoReacher-Live-v1")
    obs = env.reset()
    print(obs)

    print("forward")

    # forward
    env.unwrapped.goal = [0, .224, .102]
    for i in range(100):
        obs, rew, done, misc = env.step([1, -1, 0, 0])
        print(np.around(obs, 3))

    print("backward")

    # backward
    env.unwrapped.goal = [0, -.148, .016]
    for i in range(100):
        obs, rew, done, misc = env.step([-1, -1, 0, 0])
        print(np.around(obs, 3))

    print("upward")

    # backward
    env.unwrapped.goal = [0, .013, .25]
    for i in range(100):
        obs, rew, done, misc = env.step([.1, -1, -.1, 0])
        print(np.around(obs, 3))
