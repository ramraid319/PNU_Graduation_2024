from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        indices = torch.randperm(len(self.buffer))[:self.batch_size]
        data = [self.buffer[idx] for idx in indices]

        state = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in data])
        action = torch.tensor([x[1] for x in data], dtype=torch.uint8)
        reward = torch.tensor([x[2] for x in data], dtype=torch.float32)
        next_state = torch.stack([torch.tensor(x[3], dtype=torch.float32) for x in data])
        done = torch.tensor([x[4] for x in data], dtype=torch.bool)

        return state, action, reward, next_state, done