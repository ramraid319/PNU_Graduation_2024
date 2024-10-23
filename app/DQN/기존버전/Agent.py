import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .QNetwork import *
from .ReplayBuffer import *

class Agent:
    def __init__(self, action_size, device, total_episodes, buffer_size=20000, batch_size=32, gamma=0.98, lr=0.001, epsilon_start=0.9, epsilon_end=0.1):
        self.action_size = action_size  # 현재 sumo와 carla 모두 action_size 는 4가지로 고정됨
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.total_episodes = total_episodes

        self.device = device
        
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        self.qnet = QNetwork(self.action_size).to(self.device)
        self.qnet_target = QNetwork(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        self.qnet.eval()
        with torch.no_grad():
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = state.clone().detach().unsqueeze(0).float().to(self.device)
            qs = self.qnet(state)
            del state

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

        del state
        del next_state
        
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
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / (self.total_episodes * 0.8)


    # Save model
    def save_model(self, path, current_episode):
        torch.save({
            'qnet_state_dict': self.qnet.state_dict(),
            'qnet_target_state_dict': self.qnet_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'current_episode' : current_episode
            # 'replay_buffer': self.replay_buffer  # Optional, add if you want to save buffer state too
        }, path)

        print("Model saved successfully.")
        torch.load


    # Load model
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

        self.qnet.train()
        self.qnet_target.eval()

        print(f"Model loaded with epsilon = {self.epsilon}")

        return checkpoint['current_episode']

    # def save(self, filepath, episodes_count):
    #     torch.save({
    #         'model_state_dict': self.qnet.state_dict(),
    #         'target_state_dict': self.qnet_target.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'epsilon': self.epsilon,
    #         'episodes_count': episodes_count
    #     }, filepath)

    # def load(self, filepath):
    #     checkpoint = torch.load(filepath)
    #     self.qnet.load_state_dict(checkpoint['model_state_dict'])
    #     self.qnet_target.load_state_dict(checkpoint['target_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.epsilon = checkpoint['epsilon']
    #     return checkpoint['episodes_count']