from poppy_helpers.controller import ZMQController
import gym
import gym_ergojr
import numpy as np

env_sim = gym.make("ErgoPusher-Graphical-v1")
env_sim.seed(1)
env_sim.reset()

env_real = gym.make("ErgoPusher-Live-v1")
env_real.seed(1)
env_real.reset()


def run(act):
    obs, _, _, _ = env_real.step(act)
    print ("real\t", obs)
    for _ in range(20):
        obs, _, _, _ = env_sim.step(act)

    print ("sim\t", obs)
    input("press enter:")


run([.5, 0, 0])
run([0, .5, 0])
run([0, 0, .5])
