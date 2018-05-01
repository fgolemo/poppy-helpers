import time
import gym
import numpy as np
from gym import spaces

from poppy_helpers.constants import JOINT_LIMITS, MOVE_EVERY_N_STEPS, MAX_REFRESHRATE, JOINT_LIMITS_SPEED
from poppy_helpers.controller import SwordFightZMQController

from poppy_helpers.randomizer import Randomizer


class ErgoFightLiveEnv(gym.Env):
    def __init__(self, no_move=False, scaling=1, shield=True):
        self.no_move = no_move
        self.scaling = scaling
        self.shield = shield

        self.rand = Randomizer()
        self.last_step_time = None

        self.step_in_episode = 0
        self.step_in_fight = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6,))

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,))

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]

        self._init_robots()

    def _init_robots(self):

        self.controller_def = SwordFightZMQController(mode="def", host="flogo4.local")
        self.controller_att = SwordFightZMQController(mode="att", host="flogo2.local")

        self.controller_def.compliant(False)
        self.controller_att.compliant(False)

        self.controller_att.set_max_speed(100)
        self.controller_def.set_max_speed(100)

        self.controller_def.safe_rest()
        self.controller_att.safe_rest()

        self.controller_def.get_keys()  # in case there are keys stored

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _restPos(self):
        self.done = False

        self.controller_def.safe_rest()
        self.controller_att.safe_rest()

        time.sleep(1)  # FIXME: run loop, check joints

        self.randomize(robot=1, scaling=self.scaling)

        time.sleep(1)  # FIXME: instead run a loop here and check when
        # the joints are close to the given configuration
        self.controller_def.get_keys()  # clear queue

    def randomize(self, robot=1, scaling=1.0):
        new_pos = self.rand.random_sf(scaling)
        robot_ctrl = self.controller_att
        if robot == 1:
            robot_ctrl = self.controller_def

        robot_ctrl.goto_pos(new_pos)

    def _reset(self):
        self.step_in_episode = 0
        self._restPos()
        self._self_observe()
        self.last_step_time = time.time()
        return self.observation

    def _get_reward(self):

        collisions = self.controller_def.get_keys()

        reward = 0
        if "s" in collisions:
            reward = 1
            self._restPos()

        return reward

    def _self_observe(self):
        joint_vel_att = self.controller_att.get_posvel()
        joint_vel_def = self.controller_def.get_posvel()
        self.observation = self._normalize(
            np.hstack((joint_vel_att, joint_vel_def)).astype('float32')
        )

    def _normalize(self, pos):
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            out.append(norm)
        if len(pos) > 6:
            shifted = (np.array(pos[6:]) + JOINT_LIMITS_SPEED) / (JOINT_LIMITS_SPEED * 2)
            norm = shifted * 2 - 1
            out += list(norm)
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

        self.controller_att.goto_pos(actions)

        if not self.no_move:
            if self.step_in_episode % MOVE_EVERY_N_STEPS == 0:
                # print ("step {}, randomizing".format(self.step_in_episode))
                self.randomize(1, scaling=self.scaling)

        # observe again
        self._self_observe()
        reward = self._get_reward()

        dt = (time.time() - self.last_step_time) * 1000
        if dt < MAX_REFRESHRATE:
            time.sleep((MAX_REFRESHRATE - dt) / 1000)

        self.last_step_time = time.time()
        return self.observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # This intentionally does nothing and is only here for wrapper functions.
        # if you want graphical output, use the environments
        # "ErgoBallThrowAirtime-Graphical-Normalized-v0"
        # or
        # "ErgoBallThrowAirtime-Graphical-v0"
        # ... not the ones with "...-Headless-..."
        pass

    if __name__ == '__main__':
        import poppy_helpers
        from tqdm import tqdm

        env = gym.make("ErgoFight-Live-Shield-Move-ThreequarterRand-v0")

        env.reset()

        for i in tqdm(range(2)):
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            print(i, obs, rew, done)
