from collections import deque
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
        indices = torch.randperm(len(self.buffer))[:self.batch_size]
        data = [self.buffer[idx] for idx in indices]

        state = torch.stack([x[0].clone().detach().float() for x in data]).to(self.device)      #  state = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in data])
        action = torch.tensor([x[1] for x in data], dtype=torch.long).to(self.device)
        reward = torch.tensor([x[2] for x in data], dtype=torch.float32).to(self.device)
        next_state = torch.stack([x[3].clone().detach().float() for x in data]).to(self.device)  #  next_state = torch.stack([torch.tensor(x[3], dtype=torch.float32) for x in data]) 
        done = torch.tensor([x[4] for x in data], dtype=torch.long).to(self.device)

        return state, action, reward, next_state, done