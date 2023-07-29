# A Gym Environment for simulating an Elevator Control System

# Author: Zining Mao
# Date : July 2023

# import gymnasium as gym
import gym
from gym import spaces
import pygame
import numpy as np
from gym_elevator_group.envs.PassengerGenerator import PassengerGenerator
from pprint import pprint

MOVE_UP = 0
MOVE_DOWN = 1
STOP = 2

class ElevatorGroupEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, num_floor=18, num_elevator=4, num_passenger=10, num_timestep=50, capacity=10, maximum_timestep=5000, render_mode=None):

        self.num_floor = num_floor
        self.num_elevator = num_elevator

        self.num_passenger = num_passenger
        self.num_timestep = num_timestep

        self.maximum_timestep = maximum_timestep

        self.capacity = capacity # capacity of the elevator

        self.passenger_generator = PassengerGenerator(self.num_floor, self.num_passenger, self.num_timestep)
        self._curr_timestep = 0 # keeping track of the current timestep

        self.window_size = 512  # The size of the PyGame window

        # Observation space for the elevator group control system
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(0, self.num_floor-1, shape=[self.num_elevator, ], dtype=np.int32), # position of 4 elevators, 18 floors each.
                "passenger": spaces.Box(0, self.num_passenger, shape=[self.num_floor, ], dtype=np.int32), # people waiting on each floor
                "capacity": spaces.Box(0, self.capacity, shape=[self.num_elevator,]), # capacity of each elevator 
                "up_call": spaces.MultiBinary([self.num_elevator, self.num_floor]), # Hallway elevator calls for going up
                "down_call": spaces.MultiBinary([self.num_elevator, self.num_floor]), # Hallway elevator calls for going down
                "car_call": spaces.MultiBinary([self.num_elevator, self.num_floor]) # Target floors for the elevators
            }
            )

        # We have 3 actions for each elevator, corresponding to "up", "down", "stop". The action corresponds to index 0, 1, 2.
        # If someone needs to get off at the current floor, the stop action will let the people out. 
        # If someone needs to get on at the the current floor, the stop action will let the people in.
        # Otherwise the elevator will be waiting.
        # The above tasks can be done simultaneously in one timestep.

        self.action_space = spaces.MultiDiscrete(np.ones([self.num_elevator], dtype=np.int32) * 3,)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction the elevator will move in if that action is taken.
        I.e. 0 corresponds to "up", 1 to "down", 2 to "stop".
        """
        self._action_to_direction = {
            0: 1,
            1: -1,
            2: 0
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
                "pos": self._elevator_pos, # position of 4 elevators, 18 floors each.
                "passenger": self._curr_passenger, # people waiting on each floor
                "capacity": self._curr_capacity, # capacity of each elevator 
                "up_call": self._curr_up_call, # Hallway elevator calls for going up
                "down_call": self._curr_down_call, # Hallway elevator calls for going down
                "car_call": self._curr_car_call # Target floors for the elevators
        }

    def _get_info(self):
        return {
            "num_passenger_delivered": self._num_passenger_delivered,
            "cumulative_waiting_time": self._cumulative_waiting_time,
            "waiting_passengers": self._waiting_passengers
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the elevators' start locations uniformly at random
        self._elevator_pos = self.np_random.integers(0, self.num_floor-1, size=self.num_elevator, dtype=np.int32)

        # Generate a random sequence of tasks. The tasks is a dictionary indexed 
        self._tasks = self.passenger_generator.generate_passenger()

        # DEBUG print current tasks
        # pprint(self._tasks)

        # initialize observation variables
        self._curr_timestep = 0
        self._curr_passenger = np.zeros([self.num_floor], dtype=np.int32)
        self._curr_capacity = np.ones([self.num_elevator], dtype=np.int32) * self.capacity
        self._curr_up_call = np.zeros([self.num_floor]) 
        self._curr_down_call = np.zeros([self.num_floor])
        self._curr_car_call = np.zeros([self.num_elevator, self.num_floor])

        self._num_passenger_delivered = 0
        self._cumulative_waiting_time = 0 # time between start task to task complete (the passenger reaches the goal floor)

        self._tasks_queue = []
        for _ in range(self.num_elevator):
            self._tasks_queue.append([])
        self._pending_task = 0 # count the number of ongoing tasks
        self._waiting_passengers = [] # list of passengers that are waiting to be pick up

        # update the observation based on the task sequence
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
            # self._render_frame()

        return observation, info

    def step(self, actions):
        # The car call observation is updated when the passenger enters the elevator
        reward = 0

        # update observation from the environment task list
        if self._curr_timestep in self._tasks:
            for task in self._tasks[self._curr_timestep]:
                # update passenger
                self._curr_passenger[task[0]] += 1

                # update hall calls
                if task[2] == self.passenger_generator.UP:
                    self._curr_up_call[task[0]] = True
                elif task[2] == self.passenger_generator.DOWN:
                    self._curr_down_call[task[0]] = True

                # update pending task
                self._pending_task += 1
                self._waiting_passengers.append(task)

        # update observation based on actions for each elevator
        for idx, a in enumerate(actions):
            if a == STOP:
                # handle arriving tasks first
                # update car call
                self._curr_car_call[idx, self._elevator_pos[idx]] = False

                # count the number of passengers leaving this floor and update reward and pending task count
                task_completed = self._tasks_queue[idx].count(self._elevator_pos[idx]) 
                self._num_passenger_delivered += task_completed
                self._pending_task -= task_completed
                reward += task_completed * 5

                # remove completed tasks and update current elevator capacity
                self._tasks_queue[idx] = [i for i in self._tasks_queue[idx] if i != self._elevator_pos[idx]] 
                self._curr_capacity[idx] += task_completed

                assert 0 <= self._curr_capacity[idx] <= self.capacity

                # update hall call. We do not differentiate up calls and down calls for now 
                # and ignore the elevators operation direction
                if self._curr_up_call[self._elevator_pos[idx]] or self._curr_down_call[self._elevator_pos[idx]]:
                    tasks_this_floor = [t for t in self._waiting_passengers if t[0] == self._elevator_pos[idx]]
                    
                    # This guarantees the capacity constraint is satisfied
                    tasks_to_be_done = min(self._curr_capacity[idx], len(tasks_this_floor))
                    for i in range(tasks_to_be_done):
                        # update target floor car call
                        self._curr_car_call[idx, tasks_this_floor[i][1]] = True
                        for j in range(len(self._waiting_passengers)):
                            if np.array_equal(self._waiting_passengers[j], tasks_this_floor[i]):
                                self._waiting_passengers.pop(j)
                                break
                        self._tasks_queue[idx].append(tasks_this_floor[i][1])
                        self._curr_capacity[idx] -= 1
                    
                    # update number of waiting passengers
                    self._curr_passenger[self._elevator_pos[idx]] -= tasks_to_be_done
                    self._curr_up_call[self._elevator_pos[idx]] = False
                    self._curr_down_call[self._elevator_pos[idx]] = False

                    # update hall call if some passengers remains on the floor
                    if len(tasks_this_floor > self._curr_capacity[idx]):
                        for t in tasks_this_floor[self._curr_capacity[idx]:]:
                            if t[2] == self.passenger_generator.UP:
                                self._curr_up_call[self._elevator_pos[idx]] = True 
                            if t[2] == self.passenger_generator.DOWN:
                                self._curr_down_call[self._elevator_pos[idx]] = True 

                else:
                    # Nothing happens. The elevator just stopped.
                    pass
            else:
                # update the position of the elevators
                direction = self._action_to_direction[a]
                self._elevator_pos[idx] = np.clip(self._elevator_pos[idx] + direction, 0, self.num_floor-1)

        reward += - self._pending_task * 0.1 # 0.1 penalty for every pending task

        self._curr_timestep += 1
        self._cumulative_waiting_time += self._pending_task


        # An episode is done iff all the passengers have reached the target
        terminated = (self._curr_timestep > self.num_timestep and self._pending_task == 0) or self._curr_timestep > self.maximum_timestep
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
'''
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
'''