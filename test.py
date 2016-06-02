import gym
env = gym.make('Sphere-v0')
env.reset()
print(env.action_space)
print(env.observation_space)
print("************************")
print(env.action_space.low)
print(env.action_space.high)
for _ in range(5000):
    env.render()
    act = env.action_space.sample()
    #print(act)
    env.step(act)
