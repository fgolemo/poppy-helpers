import os
import time
from collections import deque

import gym
import cv2

import numpy as np
from gym import spaces
from gym_ergojr.utils.math import RandomPointInHalfSphere
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker, TRACKING_GREEN, TRACKING_YELLOW
from poppy_helpers.config import config_dir
from poppy_helpers.constants import JOINT_LIMITS, MOVE_EVERY_N_STEPS, MAX_REFRESHRATE, JOINT_LIMITS_SPEED, \
    SIM_VELOCITY_SCALING
from poppy_helpers.controller import SwordFightZMQController, ZMQController

from poppy_helpers.randomizer import Randomizer
from realsense_tracker.utils import add_text

DETECTION_RADIUS = 6  # minimum pixel radius of the green tracking blob, otherwise it's not detected
MIN_DIST = -0.025  # if the distance is smaller than this, the episode is solved
ITERATIONS_MAX = 10000  # pause the robot for maintenance every N steps
GRIPPER_CLOSED_MAX_FRAMES = 100
CLOSING_FRAMES = 7

# BUFFER_MAX_SIZE = 100  # write buffer to disk every <- steps

class ErgoReacherLiveEnv(gym.Env):
    def __init__(self, multi=False, multi_no=3, tracking=False):
        self.multi_goal = multi
        self.no_goals = multi_no
        self.tracking = tracking
        self.rand = Randomizer()

        self.goal = None

        # self.step_in_episode = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        self.rhis = RandomPointInHalfSphere(0.0, 0.0369, 0.0437,
                                            radius=0.2022, height=0.2610,
                                            min_dist=0.1, halfsphere=True)

        # observation = 4 joints + 4 velocities + 2 coordinates for target
        joints_no = 4
        if self.tracking:
            joints_no = 3

        self.observation_space = spaces.Box(low=-1, high=1, shape=(joints_no + joints_no + 2,), dtype=np.float32)

        # action = 4 joint angles
        self.action_space = spaces.Box(low=-1, high=1, shape=(joints_no,), dtype=np.float32)

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]

        self.cam = Camera(color=True)
        self.tracker = Tracker(TRACKING_GREEN["maxime_lower"], TRACKING_GREEN["maxime_upper"])
        # self.tracker = Tracker(TRACKING_GREEN["duct_lower"], TRACKING_GREEN["duct_upper"])
        if self.tracking:
            self.tracker_goal = Tracker(TRACKING_YELLOW["duckie_lower"], TRACKING_YELLOW["duckie_upper"])

        self.calibration = np.load(os.path.join(config_dir(), "calib.npz"))["calibration"].astype(np.int16)

        self.last_step_time = time.time()
        self.pts_tip = deque(maxlen=32)
        self.pts_goal = deque(maxlen=32)

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
        self.setSpeed(100)  # do resetting at a normal speed

        self.gripper_closed = False
        self.gripper_closed_frames = 0
        self.closest_distance = np.inf
        self.ready_to_close = False
        self.tracking_frames = 0

        if not self.tracking:
            self.goal = self.rhis.sampleSimplePoint()

        # goals = []

        # for _ in range(100000):
        #     goals.append(self.rhis.sampleSimplePoint())
        #
        # goals = np.array(goals)
        # print ("x min/max",goals[:,1].min(),goals[:,1].max())
        # print ("y min/max",goals[:,2].min(),goals[:,2].max())

        qpos = np.random.uniform(low=-0.2, high=0.2, size=6)
        qpos[[0, 3]] = 0

        self.pts_tip.clear()

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
        if self.goal is None:
            return np.zeros(8)

        pv = self.controller.get_posvel()
        if not self.tracking:
            self.observation = self._normalize(pv)[[1, 2, 4, 5, 7, 8, 10, 11]]  # leave out joint0/3
        else:
            self.observation = self._normalize(pv)[[1, 2, 4, 7, 8, 10]]  # leave out joint0/3/5
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
        y_d = .25 - (y_n * (.23 - .046))  # .056 has been modified from the .016 below
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

    def _render_img(self, frame, center, radius, x, y, pts, color_a=(0, 255, 255), color_b=(0, 0, 255)):
        g = self._goal2pixel(self.goal)
        cv2.circle(frame, (g[0], g[1]), int(5), (255, 0, 255), 3)

        # circle center
        cv2.circle(frame, (int(x), int(y)), int(radius), color_a, 2)

        # center of mass
        cv2.circle(frame, center, 5, color_b, -1)

        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], color_b, thickness)

        self.last_frame = frame.copy()

        return frame

    def _get_reward(self):
        done = False
        center_goal = None

        while True:
            frame = Camera.to_numpy(self.cam.get_color())[:, :, ::-1]  # RGB to BGR for cv2
            hsv = self.tracker.blur_img(frame)
            mask_tip = self.tracker.prep_image(hsv)

            # center of mass, radius of enclosing circle, x/y of enclosing circle
            center_tip, radius_tip, x_tip, y_tip = self.tracker.track(mask_tip)

            if self.tracking:
                mask_goal = self.tracker_goal.prep_image(hsv)
                center_goal, radius_goal, x_goal, y_goal = self.tracker_goal.track(mask_goal)

            # grab more frames until the green blob is big enough / visible
            if center_tip is not None and radius_tip > DETECTION_RADIUS:
                break

        frame2 = np.ascontiguousarray(frame, dtype=np.uint8)

        if self.tracking and center_goal is not None:
            self.goal = np.zeros(3, dtype=np.float32)
            self.goal[1:] = self._pixel2goal(center_goal)

            frame2 = self._render_img(frame2, center_goal, radius_goal, x_goal, y_goal, pts=self.pts_goal)

        if self.goal is not None:
            frame2 = self._render_img(frame2, center_tip, radius_tip, x_tip, y_tip, pts=self.pts_tip,
                                      color_a=(255, 255, 0), color_b=(255, 0, 0))

        if self.tracking and center_goal is None:
            self.goal = None
            return 0, False, np.inf, frame2.copy()

        pos_tip = self._pixel2goal(center_tip)

        reward = np.linalg.norm(np.array(self.goal[1:]) - np.array(pos_tip))
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

        return reward, done, distance, frame2.copy()

    def step(self, action):
        action_ = np.zeros(6, np.float32)
        if not self.tracking:
            action_[[1, 2, 4, 5]] = action
        else:
            action_[[1, 2, 4]] = action
            action_[5] = 50 / 90

            if self.gripper_closed:
                self.gripper_closed_frames += 1
                action_[5] = -10 / 90

                if self.gripper_closed_frames >= GRIPPER_CLOSED_MAX_FRAMES:
                    self.gripper_closed = False
                    self.gripper_closed_frames = 0

        action = np.clip(action_, -1, 1)

        if self.goal is not None:
            self.controller.goto_normalized(action)

        reward, done, distance, frame = self._get_reward()
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        if self.tracking:
            done = False

        if not self.gripper_closed:
            if distance < self.closest_distance:
                self.closest_distance = distance
                self.ready_to_close += 1
            else:
                if self.ready_to_close > CLOSING_FRAMES:
                    self.ready_to_close = 0
                    self.closest_distance = np.inf
                    self.gripper_closed = True  # only takes effect on next step
                else:
                    self.tracking_frames += 1
                    if self.tracking_frames > 50:
                        self.closest_distance = np.inf

        dt = (time.time() - self.last_step_time) * 1000
        if dt < MAX_REFRESHRATE:
            time.sleep((MAX_REFRESHRATE - dt) / 1000)

        self.last_step_time = time.time()

        self.pause_counter += 1

        return self._get_obs(), reward, done, {"distance": distance, "img": frame}

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    # env = gym.make("ErgoReacher-Live-v1")
    # obs = env.reset()
    # print(obs)
    #
    # print("forward")
    #
    # # forward
    # env.unwrapped.goal = [0, .224, .102]
    # for i in range(1000):
    #     obs, rew, done, misc = env.step([1, -1, 0, 0])
    #     # print(np.around(obs, 3))
    #     print(rew)
    #
    # print("backward")
    #
    # # backward
    # env.unwrapped.goal = [0, -.148, .016]
    # for i in range(1000):
    #     obs, rew, done, misc = env.step([-1, -1, 0, 0])
    #     # print(np.around(obs, 3))
    #     print(rew)
    #
    # print("upward")
    #
    # # backward
    # env.unwrapped.goal = [0, .013, .25]
    # for i in range(1000):
    #     obs, rew, done, misc = env.step([.1, -1, -.1, 0])
    #     # print(np.around(obs, 3))
    #     print(rew)

    env = gym.make("ErgoReacher-Tracking-Live-v1")
    obs = env.reset()
    print(obs)

    # forward
    for i in range(10000):
        obs, rew, done, misc = env.step([0, 0, 0])
        # print(np.around(obs, 3))
        print(rew)
