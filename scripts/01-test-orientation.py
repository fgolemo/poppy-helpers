import gym
import poppy_helpers

env = gym.make("ErgoFight-Live-Shield-Move-HalfRand-v0")

env.reset()


for i in range(100):
    obs = env.step([0,0,0,0,0,0])
    env.unwrapped._test_second_robot([0.3,0,0,0,0,0])
    print (obs[:6])

env.reset()


# simplus, sim, real, lstm

