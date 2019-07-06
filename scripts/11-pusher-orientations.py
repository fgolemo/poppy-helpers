from poppy_helpers.controller import ZMQController
import gym
import gym_ergojr
import numpy as np

env = gym.make("ErgoPusher-Graphical-v1")
env.reset()

zmq = ZMQController("pokey.local")
zmq.compliant(False)
zmq.set_max_speed(100)


def sim2real(lst):
    return ((np.array(lst) + np.array([-.5, 1, .5])) * -90).tolist()


def run(act):
    zmq.goto_pos(sim2real(act))

    for _ in range(20):
        env.step(act)

    input("press enter:")


run([.5, 0, 0])
run([0, .5, 0])
run([0, 0, .5])
