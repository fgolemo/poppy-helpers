import os
import time
from collections import deque

import gym
import cv2

import numpy as np
from gym import spaces
from gym_ergojr.models import MODEL_PATH
from realsense_tracker.camera import Camera
from realsense_tracker.tracker import Tracker, TRACKING_GREEN, TRACKING_RED
from poppy_helpers.config import config_dir
from poppy_helpers.controller import ZMQController

from poppy_helpers.randomizer import Randomizer
from realsense_tracker.utils import add_text

DETECTION_RADIUS = 6  # minimum pixel radius of the green tracking blob, otherwise it's not detected
MIN_DIST = 0.01  # if the distance is smaller than this, the episode is solved
ITERATIONS_MAX = 10000  # pause the robot for maintenance every N steps

PUSHER_GOAL_X = [-.2, -.1]
PUSHER_GOAL_Y = [-.1, .05]
PUSHER_PUCK_X = [-0.07, -0.10]
PUSHER_PUCK_Y = [0.05, 0.08]

PUSHER_PUCK_X_NORM = [
    min(PUSHER_PUCK_X[0], PUSHER_PUCK_X[0]),
    max(PUSHER_PUCK_X[1], PUSHER_PUCK_X[1])
]
PUSHER_PUCK_Y_NORM = [
    min(PUSHER_PUCK_Y[0], PUSHER_PUCK_Y[0]),
    max(PUSHER_PUCK_Y[1], PUSHER_PUCK_Y[1])
]


class ErgoPusherLiveEnv(gym.Env):
    def __init__(self):
        self.goals_done = 0

        self.metadata = {'render.modes': ['human']}

        # observation = 3 joints + 3 velocities + 2 puck position + 2 coordinates for target
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3 + 3 + 2 + 2,), dtype=np.float32)  #

        # action = 3 joint angles
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32)  #

        self.cam = Camera(color=True)
        self.tracker_puck = Tracker(TRACKING_RED["pusher_lower"], TRACKING_RED["pusher_upper"])
        self.tracker_pusher = Tracker(TRACKING_GREEN["pusher_lower"], TRACKING_GREEN["pusher_upper"])

        self.calib_sim = np.load(os.path.join(MODEL_PATH, "calibration-pusher-sim.npz"))['arr_0']
        self.calib_real = np.load(os.path.join(config_dir(), "calibration-pusher-real.npz"))["arr_0"].astype(np.int16)

        self.x_rr = self.calib_real[0, 0]
        self.x_rl = self.calib_real[2, 0]
        self.x_sr = self.calib_sim[0, 0]
        self.x_sl = self.calib_sim[2, 0]

        self.y_rr = self.calib_real[0, 1]
        self.y_rl = self.calib_real[2, 1]
        self.y_sr = self.calib_sim[0, 1]
        self.y_sl = self.calib_sim[2, 1]

        self.x_rm = self.calib_real[1, 0]
        self.y_rm = self.calib_real[1, 1]
        self.x_sm = self.calib_sim[1, 0]
        self.y_sm = self.calib_sim[1, 1]

        self.x_rc = (self.x_rl + self.x_rr) / 2
        self.y_rc = (self.y_rl + self.y_rr) / 2

        # [0.13473208 0.00849917]
        # [0.00820219 0.1347505]
        # [-0.13471367  0.00878573]

        # [56 266]
        # [226 441]
        # [398 251]

        self.last_step_time = time.time()
        self.pts_tip = deque(maxlen=32)
        self.pts_puck = deque(maxlen=32)

        self.rest_pos = np.array([-.5, 1, .5])

        # self.last_frame = np.ones((480, 640, 3), dtype=np.uint8)

        # DEBUGGING CODE

        # frame = Camera.to_numpy(self.cam.get_color())[:, :, ::-1]
        # frame2 = np.ascontiguousarray(frame, dtype=np.uint8)
        # print (frame.shape)
        # for _ in range(500):
        #     self._sample_goal()
        #
        #     pix = self._sim2pixel(self.goal)
        #     print(self.goal, pix)
        #     cv2.circle(frame2, (int(pix[0]), int(pix[1])), 5, (255,0,0), 4)
        #
        # cv2.circle(frame2, (int(50), int(50)), 10, (0, 0, 0), 4)
        # cv2.circle(frame2, (int(50), int(5)), 10, (0, 255, 0), 4)
        # cv2.circle(frame2, (int(5), int(50)), 10, (0, 255, 255), 4)
        #
        # cv2.imshow("Frame2", frame2)
        # cv2.waitKey(10000)
        #
        # quit()

        self.controller = ZMQController(host="pokey.local")
        self._setSpeedCompliance()

        super().__init__()

    def sim2real(self, act):
        act = np.clip(act, -1, 1)
        return ((np.array(act) + self.rest_pos) * -90).tolist()

    def _setSpeedCompliance(self):
        self.controller.compliant(False)
        self.controller.set_max_speed(100)  # default: 100

    def setSpeed(self, speed):
        assert speed > 0 and speed < 1000
        self.controller.set_max_speed(speed)
        self.last_speed = speed

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        action = self.sim2real(action)
        self.controller.goto_pos(action)

        reward, done, distance, frame, center_puck = self._get_reward()
        cv2.imshow("Live Feed", frame)
        cv2.waitKey(1)

        obs = self._get_obs(center_puck)
        return obs, reward, done, {"distance": distance, "img": frame}

    def _sim2pixel(self, sim_coords):
        x_t = sim_coords[0]
        x = -x_t * (self.x_rl - self.x_rr) / (self.x_sr - self.x_sl) + self.x_rc

        y_t = sim_coords[1]
        y = (y_t - self.y_sr) * (self.y_rm - self.y_rc) / (self.y_sm - self.y_sr) + self.y_rc

        return [int(x), int(y)]

    def _pixel2sim(self, pix_coords):
        x_g = pix_coords[0]
        x = (self.x_rc - x_g) * (self.x_sr - self.x_sl) / (self.x_rl - self.x_rr)

        y_g = pix_coords[1]
        y = (y_g - self.y_rc) * (self.y_sm - self.y_sr) / (self.y_rm - self.y_rc) + self.y_sr

        return [x, y]

    def _img_and_track(self):
        while True:
            frame = Camera.to_numpy(self.cam.get_color())[:, :, ::-1]  # RGB to BGR for cv2
            hsv = self.tracker_pusher.blur_img(frame)
            # mask_tip = self.tracker_pusher.prep_image(hsv)
            #
            # # center of mass, radius of enclosing circle, x/y of enclosing circle
            # center_tip, radius_tip, x_tip, y_tip = self.tracker_pusher.track(mask_tip)

            mask_puck = self.tracker_puck.prep_image(hsv)
            center_puck, radius_puck, x_puck, y_puck = self.tracker_puck.track(mask_puck)

            # grab more frames until the green blob is big enough / visible
            if center_puck is not None and radius_puck > DETECTION_RADIUS:
                break

        frame2 = np.ascontiguousarray(frame, dtype=np.uint8)

        goal = self._sim2pixel(self.goal)

        self._render_tracking(frame2, center_puck, radius_puck, x_puck, y_puck, pts=self.pts_puck)

        cv2.circle(frame2, tuple(goal), 10, (255, 0, 255), 4)

        return frame2, self._pixel2sim(center_puck)

    def reset(self):
        self.setSpeed(100)

        self._sample_goal_and_puck()

        qpos = np.random.uniform(low=-0.1, high=0.1, size=3)

        self.controller.goto_pos(self.sim2real(qpos))

        while True:
            frame, center = self._img_and_track()

            puck = self._sim2pixel(self.puck)

            cv2.circle(frame, tuple(puck), 10, (255, 0, 0), 4)

            add_text(frame, "MOVE PUCK TO BLUE CIRCLE AND PRESS KEY", (0, 0, 0), .5)

            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                break

        return self._get_obs()

    def _get_obs(self, center_puck=None):
        pv = self.controller.get_posvel()

        goal = self.normalize_goal()

        if center_puck is None:
            puck = self.normalize_puck(self.puck)
        else:
            puck = self.normalize_puck(center_puck)

        self.observation = np.hstack((self._normalize(pv), puck, goal))
        return self.observation

    def _normalize(self, pos):
        pos = np.array(pos).astype('float32') * -1 # joints are inverted
        pos[:3] = ((pos[:3] + 90) / 180) * 2 - 1  # positions
        pos[3:] = ((pos[3:] + 300) / 600) * 2 - 1  # velocities
        return pos

    def _render_tracking(self, frame, center, radius, x, y, pts):
        # circle center
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # center of mass
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

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
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        return frame

    def _sample_goal_and_puck(self):
        self.goal = [
            np.random.uniform(PUSHER_GOAL_X[0], PUSHER_GOAL_X[1]),
            np.random.uniform(PUSHER_GOAL_Y[0], PUSHER_GOAL_Y[1])
        ]
        self.puck = [
            np.random.uniform(PUSHER_PUCK_X[0], PUSHER_PUCK_X[1]),
            np.random.uniform(PUSHER_PUCK_Y[0], PUSHER_PUCK_Y[1])
        ]

    def normalize_goal(self):
        x = (self.goal[0] - PUSHER_GOAL_X[0]) / (
                PUSHER_GOAL_X[1] - PUSHER_GOAL_X[0])
        y = (self.goal[1] - PUSHER_GOAL_Y[0]) / (
                PUSHER_GOAL_Y[1] - PUSHER_GOAL_Y[0])
        return np.array([x, y])

    def normalize_puck(self, puck):
        x = (puck[0] - PUSHER_PUCK_X_NORM[0]) / (
                PUSHER_PUCK_X_NORM[1] - PUSHER_PUCK_X_NORM[0])
        y = (puck[1] - PUSHER_PUCK_Y_NORM[0]) / (
                PUSHER_PUCK_Y_NORM[1] - PUSHER_PUCK_Y_NORM[0])
        return np.array([x, y])

    def _get_reward(self):
        done = False

        frame, center_puck = self._img_and_track()

        reward = np.linalg.norm(np.array(self.goal) - np.array(center_puck))
        distance = reward.copy()

        reward *= -1  # the reward is the inverse distance
        if distance < MIN_DIST:  # this is a bit arbitrary, but works well
            done = True
            reward = 1

        return reward, done, distance, frame.copy(), center_puck


if __name__ == '__main__':
    env = gym.make("ErgoPusher-Live-v1")

    for _ in range(3):
        print("=== RESET ===")
        obs = env.reset()
        print(obs, obs.shape)

        start = time.time()

        # forward
        for i in range(101):
            obs, rew, done, misc = env.step([0, 0, 0])
            # print(rew, done, misc)

        diff = time.time() - start
        print("diff", diff)
