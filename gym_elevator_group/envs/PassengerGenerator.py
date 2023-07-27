import random

class PassengerGenerator():
    # Generate Task Sequences
    def __init__(self, num_passenger, num_timestep) -> None:
        self.num_passenger = num_passenger
        self.num_timestep = num_timestep

    def generate_passenger(self):
        passengers = []
        for i in range(self.num_passenger):
            
