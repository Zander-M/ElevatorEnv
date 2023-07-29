import gym
import gym_elevator_group
from pprint import pprint

env =  gym.make("gym_elevator_group/ElevatorGroup-v0", num_floor=3, num_elevator=1, num_passenger=3, num_timestep=3, capacity=1)
env.action_space.seed(10)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    pprint(observation)
    # pprint(info)
    # pprint(reward)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
