from gym.envs.registration import register

register(
    id='strike-v0',
    entry_point='uav_gym.envs:StrikeEnv',
)