import numpy as np
import gym

env = gym.make("uav_gym:strike-v0")
observation = env.reset()

for _ in range(10):
    r_e = 0
    action = 0
    for t in range(10000):
        # env.render()
        if t%100==0:
            action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        # print(observation.shape, reward)
        # print(observation.shape)
        r_e += reward
        if done:
            
            print("done in step:", t)
            for t in env.targets:
                print(t.alive)
            observation = env.reset()
            break
    print("episodic reward: ", r_e)
    print("-----------------------------")
env.close()