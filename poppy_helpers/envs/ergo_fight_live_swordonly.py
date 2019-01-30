import time
import gym
import numpy as np
from gym import spaces

from poppy_helpers.constants import JOINT_LIMITS, MOVE_EVERY_N_STEPS, MAX_REFRESHRATE, JOINT_LIMITS_SPEED, \
    SIM_VELOCITY_SCALING
from poppy_helpers.controller import SwordFightZMQController, ZMQController

from poppy_helpers.randomizer import Randomizer


class ErgoFightLiveEnv(gym.Env):
    def __init__(self, no_move=False, scaling=1, shield=True, sim=False, compensation=True):
        self.no_move = no_move
        self.scaling = scaling
        self.shield = shield
        self.sim = sim
        self.compensation = compensation

        self.rand = Randomizer()
        self.last_step_time = None

        self.step_in_episode = 0
        self.step_in_fight = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        # 6 own joint pos, 6 own joint vel, 6 enemy joint pos, 6 enemy joint vel
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6 + 6 + 6 + 6,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.diffs = [JOINT_LIMITS[i][1] - JOINT_LIMITS[i][0] for i in range(6)]

        self._init_robots()

    def _init_robots(self):

        if self.compensation:
            self.controller_def = SwordFightZMQController(mode="def", host="flogo4.local")
            self.controller_att = SwordFightZMQController(mode="att", host="flogo2.local")
        else:
            self.controller_def = ZMQController(host="flogo4.local")
            self.controller_att = ZMQController(host="flogo2.local")

        self._setSpeedCompliance()

        self.controller_def.get_keys()  # in case there are keys stored

    def _setSpeedCompliance(self):
        self.controller_def.compliant(False)
        self.controller_att.compliant(False)
        self.controller_att.set_max_speed(100)
        self.controller_def.set_max_speed(100)

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _restPos(self):
        self.done = False

        self._setSpeedCompliance()

        self.controller_def.safe_rest()
        self.controller_att.safe_rest()

        time.sleep(.6)  # FIXME: run loop, check joints

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
        # self.observation = ((self._normalize( ### OLD AND WRONG IMPL
        #     np.hstack((joint_vel_att, joint_vel_def)).astype('float32')
        # )), )
        self.observation = ((  # this should be the technically correct solution
                                np.hstack((
                                    self._normalize(joint_vel_att),
                                    self._normalize(joint_vel_def)
                                ))

                            ),)

    def _normalize(self, pos):
        pos = np.array(pos).astype('float32')
        out = []
        for i in range(6):
            shifted = (pos[i] - JOINT_LIMITS[i][0]) / self.diffs[i]  # now it's in [0,1]
            norm = shifted * 2 - 1
            if i == 0 and self.sim:
                norm *= -1  # between sim and real the first joint it inverted
            out.append(norm)
        if len(pos) > 6:
            shifted = (pos[6:] + JOINT_LIMITS_SPEED) / (JOINT_LIMITS_SPEED * 2)
            norm = shifted * 2 - 1
            if self.sim:
                norm[0] *= -1  # between sim and real the first joint it inverted (probably both pos * vel)
            out += list(norm)
        return out

    def _denormalize(self, actions):
        out = []
        if self.sim:
            actions[0] *= -1

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

    def _test_second_robot(self, actions):
        actions = self.prep_actions(actions)
        self.controller_def.goto_pos(actions)

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

    env = gym.make("ErgoFight-Live-Shield-Move-HalfRand-Sim-v0")

    env.reset()


    def test1():
        obs_buf = []

        for episode in range(5):
            for frame in tqdm(range(1000)):
                if frame < 50:
                    action = [1, 0.5, 0.5, 1, -0.5, 0.5]
                elif frame >= 50 and frame < 100:
                    action = [0, 0, 0, 0, 0, 0]
                else:
                    action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                obs_buf.append(obs)
                # print(i, obs, rew, done)
                if done or frame == 999:
                    env.reset()
                    break

        obs = np.array(obs_buf)
        print(obs.shape)

        print("robo 1 pos")
        print(obs[:, 0, :6].max())
        print(obs[:, 0, :6].min())
        print("robo 2 pos")
        print(obs[:, 0, 12:18].max())
        print(obs[:, 0, 12:18].min())

        print("robo 1 vel")
        print(obs[:, 0, 6:12].max())
        print(obs[:, 0, 6:12].min())
        print("robo 2 vel")
        print(obs[:, 0, 18:24].max())
        print(obs[:, 0, 18:25].min())


    def test2():

        for _ in tqdm(range(500)):
            # action = [1,0,0,1,0,0] # bottom turns right, top turns left
            action = [1, 1, -1, 1, 1, -1]  # bottom turns right, top turns left
            obs, rew, done, _ = env.step(action)
        print(obs)

        for _ in tqdm(range(500)):
            # action = [1,0,0,1,0,0] # bottom turns right, top turns left
            action = [0, -1, 1, 0, -1, 1]  # bottom turns right, top turns left
            obs, rew, done, _ = env.step(action)
        print(obs)

        env.reset()


    input = [1, 1, -1, 1, 1, -1]
    out = [1.00522224, -0.94977782, 0.99866664, -1.002, -1.00522224, 0.99866664]
    # all inverted except first

    test2()

### NORMAL
# robo 1 pos
# 0.5914444393581815
# -0.5587777879503038
# robo 2 pos
# 0.1422332525253296
# -0.1422332525253296
# robo 1 vel
# 0.7370400428771973
# -0.586080014705658
# robo 2 vel
# 0.3818399906158447
# -0.41736000776290894

### SIM
