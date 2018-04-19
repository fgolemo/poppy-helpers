from gym.envs import register

register(
    id='ErgoFight-Live-Fencing-Swordonly-v0',
    entry_point='poppy_helpers.envs:ErgoFightLiveEnv',
    timestep_limit=150,
    reward_threshold=150,
    kwargs={'fencing_mode': True, 'with_img': False, 'sword_only': True},
)