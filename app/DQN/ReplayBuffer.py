from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int32)).to(self.device)
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).to(self.device)
        next_state = torch.tensor(np.stack([x[3] for x in data]), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).to(self.device)
        return state, action, reward, next_state, done