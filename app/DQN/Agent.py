import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .QNetwork import *
from .ReplayBuffer import *

epsilon_start = 0.9
epsilon_end = 0.1
epsilon_decay = 0.997

# cuda_available = torch.cuda.is_available()
# print(f"CUDA available: {cuda_available}")

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# print(device)

class Agent:
    def __init__(self, action_size, device, total_episodes):
        self.gamma = 0.99  # 0.9
        self.lr = 0.0001   # 0.0005
        self.epsilon = epsilon_start   # 0.1
        self.buffer_size = 50000   # 10000
        self.batch_size =  64  #32
        self.action_size = action_size  # <-- 2
        self.device = device
        self.total_episodes = total_episodes

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size, device)
        self.qnet = QNet(self.action_size).float().to(device)
        self.qnet_target = QNet(self.action_size).float().to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        # if np.random.rand() < self.epsilon:
        #     return np.random.choice(self.action_size)
        # else:
        #     state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
            
        #     self.qnet = self.qnet.float()  # ??
        #     qs = self.qnet(state)
        #     return qs.argmax().item()
        
        self.qnet.eval()  # Set to evaluation mode to prevent batch norm errors
        with torch.no_grad():  # Disable gradient calculation for action selection
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(self.device)
            qs = self.qnet(state)
        self.qnet.train()  # Set back to training mode after inference
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
        # Decay epsilon
        if self.epsilon > epsilon_end:
            self.epsilon *= epsilon_decay
            
        print(f"Epsilon: {self.epsilon:.3f}")

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