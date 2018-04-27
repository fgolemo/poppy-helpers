from gym.envs import register

register(
    id='ErgoFight-Live-Shield-Move-ThreequarterRand-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=1000,
    reward_threshold=150,
    kwargs={'shield': True, 'no_move': False, 'scaling': 0.75},
)