from gym.envs.registration import register

register(
    id="gym_elevator_group/ElevatorGroup-v0",
    entry_point="gym_elevator_group.envs:ElevatorGroupEnv",
)
