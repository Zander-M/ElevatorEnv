import random
import numpy as np

class PassengerGenerator():
    # Generate Task Sequences
    def __init__(self, num_floor, num_passenger, num_timestep) -> None:

        self.num_floor = num_floor
        self.num_passenger = num_passenger
        self.num_timestep = num_timestep

    def generate_passenger(self, type="uniform"):
        # return a dict of tasks indexed by start time
        passengers = dict()
        if type=="uniform":
            for i in range(self.num_passenger):
                curr_floor = random.randint(1, self.num_floor)
                dest_floor = random.randint(1, self.num_floor)
                while curr_floor == dest_floor:
                    dest_floor = random.randint(1, self.num_floor)
                if dest_floor > curr_floor:
                    # going up
                    direction = 1
                else:
                    # going down
                    direction = 0 
                start_time = random.randint(0, self.num_timestep) # timestep when the user push the button
                if start_time not in passengers:
                    passengers[start_time] = []
                passengers[start_time].append(np.array([curr_floor, dest_floor, direction]))

            return passengers

        else:
            assert False, "Task type not implemented"


