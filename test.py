import gym
env = gym.make('Sphere-v0')
env.reset()
for _ in range(5000):
    env.render()
    env.step(env.action_space.sample())
