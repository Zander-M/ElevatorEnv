import random
import numpy as np


class PassengerGenerator:
    # Generate Task Sequences
    
    def __init__(self, num_floor, num_passenger, num_timestep) -> None:

        self.num_floor = num_floor
        self.num_passenger = num_passenger
        self.num_timestep = num_timestep
        self.UP = 1
        self.DOWN = 0

    def generate_passenger(self, type="uniform"):
        # return a dict of tasks indexed by start time
        passengers = dict()
        if type=="uniform":
            for i in range(self.num_passenger):
                curr_floor = random.randint(0, self.num_floor-1)
                dest_floor = random.randint(0, self.num_floor-1)
                while curr_floor == dest_floor:
                    dest_floor = random.randint(0, self.num_floor-1)
                if dest_floor > curr_floor:
                    # going up
                    direction = self.UP
                else:
                    # going down
                    direction = self.DOWN
                start_time = random.randint(0, self.num_timestep) # timestep when the user push the button
                if start_time not in passengers:
                    passengers[start_time] = []
                passengers[start_time].append(np.array([curr_floor, dest_floor, direction]))
            return passengers
        else:
            assert False, "Task type not implemented"


