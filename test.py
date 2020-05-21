import numpy as np
import gym

env = gym.make("uav_gym:strike-v0")
observation = env.reset()

for _ in range(1):
    r_e = 0
    action = [0] * env.n_agents
    for t in range(1000):
        action = [env.action_space[i].sample() for i in range(env.n_agents)]
        observation, reward, done, info = env.step(action)
        # print(len(observation[0]))
        # print(observation)
        r_e += sum(reward)
        if all(done):
            print("done in step:", t)
            # for t in env.targets:
            #     print(t.alive)
            #render
            if env.render:
                env.vis.plot()
            observation = env.reset()
            break
    print("episodic reward: ", r_e)
    print("-----------------------------")
env.close()