import gym
import gym_elevator_group

env =  gym.make("gym_elevator_group/ElevatorGroup-v0")
env.action_space.seed(10)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(observation)
    print(info)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
