import os
import time
from collections import deque

import gym
import cv2

import numpy as np
from gym import spaces
from gym_ergojr.utils.math import RandomPointInHalfSphere
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker, TRACKING_GREEN
from poppy_helpers.config import config_dir
from poppy_helpers.constants import JOINT_LIMITS, MOVE_EVERY_N_STEPS, MAX_REFRESHRATE, JOINT_LIMITS_SPEED, \
    SIM_VELOCITY_SCALING
from poppy_helpers.controller import SwordFightZMQController, ZMQController

from poppy_helpers.randomizer import Randomizer
from realsense_tracker.utils import add_text

DETECTION_RADIUS = 6  # minimum pixel radius of the green tracking blob, otherwise it's not detected
MIN_DIST = -0.025  # if the distance is smaller than this, the episode is solved
ITERATIONS_MAX = 10000  # pause the robot for maintenance every N steps


# BUFFER_MAX_SIZE = 100  # write buffer to disk every <- steps

class ErgoReacherLiveEnv(gym.Env):
    def __init__(self, multi=False, multi_no=3):
        self.multi_goal = multi
        self.no_goals = multi_no
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
        self.tracker = Tracker(TRACKING_GREEN["maxime_lower"], TRACKING_GREEN["maxime_upper"])
        self.calibration = np.load(os.path.join(config_dir(), "calib.npz"))["calibration"].astype(np.int16)

        self.last_step_time = time.time()
        self.pts = deque(maxlen=32)

        self.last_frame = np.ones((480, 640, 3), dtype=np.uint8)

        self.pause_counter = 0

        self.last_speed = 100

        self.goals_reached = 0

        # self.goal_states=[]

        # for _ in range(10000):
        #     goal = self.rhis.sampleSimplePoint()
        #     self.goal_states.append(self._goal2pixel(goal))

        self._init_robot()

        super().__init__()

    def _init_robot(self):
        self.controller = ZMQController(host="flogo3.local")
        self._setSpeedCompliance()

    def _setSpeedCompliance(self):
        self.controller.compliant(False)
        self.controller.set_max_speed(500)  # default: 100

    def setSpeed(self, speed):
        assert speed > 0 and speed < 1000
        self.controller.set_max_speed(speed)
        self.last_speed = speed

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.goal = self.rhis.sampleSimplePoint()

        self.setSpeed(100)  # do resetting at a normal speed

        # goals = []

        # for _ in range(100000):
        #     goals.append(self.rhis.sampleSimplePoint())
        #
        # goals = np.array(goals)
        # print ("x min/max",goals[:,1].min(),goals[:,1].max())
        # print ("y min/max",goals[:,2].min(),goals[:,2].max())

        qpos = np.random.uniform(low=-0.2, high=0.2, size=6)
        qpos[[0, 3]] = 0

        self.pts.clear()
        add_text(self.last_frame, "=== RESETTING ===")

        if (self.pause_counter > ITERATIONS_MAX):
            self.pause_counter = 0
            self.controller.compliant(True)
            input("\n\n=== MAINTENANCE: PLEASE CHECK THE ROBOT AND THEN PRESS ENTER TO CONTINUE ===")
            self.controller.compliant(False)
            time.sleep(.5)  # wait briefly to make sure all joints are non-compliant / powered

        # this counts as rest position
        self.controller.goto_normalized(qpos)

        cv2.imshow("Frame", self.last_frame)
        cv2.waitKey(1000)

        self.setSpeed(self.last_speed)

        self.last_step_time = time.time()

        return self._get_obs()

    def _get_obs(self):
        pv = self.controller.get_posvel()
        self.observation = self._normalize(pv)[[1, 2, 4, 5, 7, 8, 10, 11]]  # leave out joint0/3
        self.observation = np.hstack((self.observation, self.rhis.normalize(self.goal)[1:]))
        return self.observation

    def _normalize(self, pos):
        pos = np.array(pos).astype('float32')
        pos[:6] = ((pos[:6] + 90) / 180) * 2 - 1  # positions
        pos[6:] = ((pos[6:] + 300) / 600) * 2 - 1  # velocities
        return pos

    def _pixel2goal(self, camcenter):
        pos = np.array(camcenter).astype(np.int16)
        x_n = (pos[0] - self.calibration[0, 0]) / (self.calibration[1, 0] - self.calibration[0, 0])
        x_d = .224 - (x_n * (.224 + .148))
        y_n = (pos[1] - self.calibration[2, 1]) / (self.calibration[1, 1] - self.calibration[2, 1])
        y_d = .25 - (y_n * (.25 - .016))
        return np.array([x_d, y_d])

    def _goal2pixel(self, goal):
        # print (self.goal[1:], self.calibration)

        # x_n = (float(goal[1]) + .0979) / (.2390 + .0979)
        # x_d = x_n * (self.calibration[0, 0] - self.calibration[1, 0]) + self.calibration[1, 0]
        # y_n = (float(goal[2]) - .0437) / (.2458 - .0437)
        # y_d = self.calibration[0, 1] - (y_n * (self.calibration[0, 1] - self.calibration[2, 1]))

        x_n = (float(goal[1]) + .148) / (.224 + .148)
        x_d = x_n * (self.calibration[0, 0] - self.calibration[1, 0]) + self.calibration[1, 0]
        y_n = (float(goal[2]) - .016) / (.25 - .016)
        y_d = self.calibration[0, 1] - (y_n * (self.calibration[0, 1] - self.calibration[2, 1]))
        # print ([x_d, y_d])
        return np.array([int(round(x_d)), int(round(y_d))])

    def _render_img(self, frame, center, radius, x, y):
        g = self._goal2pixel(self.goal)
        cv2.circle(frame, (g[0], g[1]), int(5), (255, 0, 255), 3)

        # circle center
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # center of mass
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        self.pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        self.last_frame = frame.copy()

        return frame

    def _get_reward(self):
        done = False

        while True:
            frame, (center, radius, x, y) = self.tracker.get_frame_and_track(self.cam)

            # grab more frames until the green blob is big enough / visible
            if center is not None and radius > DETECTION_RADIUS:
                break

        frame2 = np.ascontiguousarray(frame, dtype=np.uint8)
        frame3 = self._render_img(frame2, center, radius, x, y)

        pos = self._pixel2goal(center)

        reward = np.linalg.norm(np.array(self.goal[1:]) - np.array(pos))
        distance = reward.copy()
        reward *= -1  # the reward is the inverse distance

        if reward > MIN_DIST:  # this is a bit arbitrary, but works well
            done = True
            reward = 1
            if self.multi_goal:
                self.goals_reached += 1
                self.goal = self.rhis.sampleSimplePoint()
                if not self.goals_reached == self.no_goals:
                    done = False

        return reward, done, distance, frame3.copy()

    def step(self, action):
        action_ = np.zeros(6, np.float32)
        action_[[1, 2, 4, 5]] = action
        action = np.clip(action_, -1, 1)

        self.controller.goto_normalized(action)

        reward, done, distance, frame = self._get_reward()
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        dt = (time.time() - self.last_step_time) * 1000
        if dt < MAX_REFRESHRATE:
            time.sleep((MAX_REFRESHRATE - dt) / 1000)

        self.last_step_time = time.time()

        self.pause_counter += 1

        return self._get_obs(), reward, done, {"distance": distance, "img": frame}

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    env = gym.make("ErgoReacher-Live-v1")
    obs = env.reset()
    print(obs)

    print("forward")

    # forward
    env.unwrapped.goal = [0, .224, .102]
    for i in range(1000):
        obs, rew, done, misc = env.step([1, -1, 0, 0])
        # print(np.around(obs, 3))
        print(rew)

    print("backward")

    # backward
    env.unwrapped.goal = [0, -.148, .016]
    for i in range(1000):
        obs, rew, done, misc = env.step([-1, -1, 0, 0])
        # print(np.around(obs, 3))
        print(rew)

    print("upward")

    # backward
    env.unwrapped.goal = [0, .013, .25]
    for i in range(1000):
        obs, rew, done, misc = env.step([.1, -1, -.1, 0])
        # print(np.around(obs, 3))
        print(rew)
