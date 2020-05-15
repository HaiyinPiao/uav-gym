import numpy as np
import gym

env = gym.make("uav_gym:strike-v0")
observation = env.reset()

for _ in range(1):
    r_e = 0
    action = 0
    for t in range(10000):
        # env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        # print(observation.shape, reward)
        # print(observation.shape)
        r_e += reward
        if done:
            print("done in step:", t)
            for t in env.targets:
                print(t.alive)
            #render
            if env.render:
                env.vis.plot()
            observation = env.reset()
            break
    print("episodic reward: ", r_e)
    print("-----------------------------")
env.close()