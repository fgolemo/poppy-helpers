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

