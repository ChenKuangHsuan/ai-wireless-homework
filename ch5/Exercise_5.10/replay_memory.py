import os
import random
import numpy as np


class ReplayMemory:
    def __init__(self, entry_size):
        self.entry_size = entry_size
        self.memory_size = 400000
        self.action_band = np.empty(self.memory_size, dtype=np.uint8)
        # Continous power control
        self.action_power = np.empty(self.memory_size, dtype=np.float64)
        self.rewards = np.empty(self.memory_size, dtype=np.float64)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.batch_size = 4000
        self.count = 0
        self.current = 0
        

    def add(self, prestate, poststate, reward, action_band, action_power):
        self.action_band[self.current] = action_band
        self.action_power[self.current] = action_power
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):

        if self.count < self.batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0,self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        action_band = self.action_band[indexes]
        action_power = self.action_power[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate, action_band, action_power, rewards
   
