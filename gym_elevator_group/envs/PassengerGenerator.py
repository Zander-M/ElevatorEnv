import random
import numpy as np

class PassengerGenerator():
    # Generate Task Sequences
    def __init__(self, num_passenger, num_timestep, num_floor) -> None:

        self.num_passenger = num_passenger
        self.num_timestep = num_timestep
        self.num_floor = num_floor

    def generate_passenger(self):
        passengers = []
        for i in range(self.num_passenger):
            curr_floor = random.randint(1, self.num_floor)
            dest_floor = random.randint(1, self.num_floor)
            while curr_floor == dest_floor:
                dest_floor = random.randint(1, self.num_floor)
            if dest_floor > curr_floor:
                direction = 1
            else:
                direction = 0 
            start_time = random.randint(0, self.num_timestep) # timestep when the user push the button
            passengers.append(np.array([curr_floor, dest_floor, direction, start_time]))

        return passengers


