from gym.envs import register

register(
    id='ErgoFight-Live-Shield-Move-ThreequarterRand-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'shield': True, 'no_move': False, 'scaling': 0.75},
)

register(
    id='ErgoFight-Live-Shield-Move-HalfRand-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'shield': True, 'no_move': False, 'scaling': 0.5},
)

register(
    id='ErgoFight-Live-Shield-Move-HalfRand-NoComp-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'shield': True, 'no_move': False, 'scaling': 0.5, 'compensation':False},
)

register(
    id='ErgoFight-Live-Shield-Move-HalfRand-Sim-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'shield': True, 'no_move': False, 'scaling': 0.5, 'sim': True},
)

register(
    id='ErgoReacher-Live-v1',
    entry_point='poppy_helpers.envs:ErgoReacherLiveEnv',
    timestep_limit=1000,
    reward_threshold=1,
    kwargs={},
)

register(
    id='ErgoPusher-Live-v1',
    entry_point='poppy_helpers.envs:ErgoPusherLiveEnv',
    timestep_limit=100,
    reward_threshold=1,
    kwargs={},
)

register(
    id='ErgoReacher-MultiGoal-Live-v1',
    entry_point='poppy_helpers.envs:ErgoReacherLiveEnv',
    timestep_limit=1000,
    reward_threshold=1,
    kwargs={'multi':True},
)

register(
    id='ErgoReacher-Tracking-Live-v1',
    entry_point='poppy_helpers.envs:ErgoReacherLiveEnv',
    timestep_limit=10000000,
    reward_threshold=1,
    kwargs={'tracking':True},
)

