import numpy as np
import gym

env = gym.make("uav_gym:strike-v0")
observation = env.reset()
for _ in range(300):
    # env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)
    observation, reward, done, info = env.step(0)
    # print(observation, reward)
    if done:
        observation = env.reset()
        break
env.close()