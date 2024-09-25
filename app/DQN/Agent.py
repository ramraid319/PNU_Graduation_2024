import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import QNetwork
import ReplayBuffer

class Agent:
    def __init__(self, action_size, device, buffer_size=50000, batch_size=64, gamma=0.98, lr=0.001, epsilon_start=0.8, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = device
        
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNetwork(action_size).to(self.device)
        self.qnet_target = QNetwork(action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        self.qnet.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            qs = self.qnet(state)
        self.qnet.train()
        return qs.argmax().item() if np.random.rand() >= self.epsilon else np.random.choice(self.action_size)

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[torch.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]
        next_q.detach()
        
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay