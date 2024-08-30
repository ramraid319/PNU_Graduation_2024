import matplotlib.pyplot as plt
import copy
from collections import deque
import random
import numpy as np
import sumo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

epsilon_start = 0.3
epsilon_end = 0.05
epsilon_decay = 0.995

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)


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
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]), dtype=torch.float32).to(device)
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int32)).to(device)
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).to(device)
        next_state = torch.tensor(np.stack([x[3] for x in data]), dtype=torch.float32).to(device)
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).to(device)
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # self.l1 = nn.Linear(71, 128)     # input size : 71
        # self.l2 = nn.Linear(128, 128)
        # self.l3 = nn.Linear(128, action_size)
        
        # self.l1 = nn.Linear(295, 1024)   # input size : 295
        # self.l2 = nn.Linear(1024, 1024)
        # self.l3 = nn.Linear(1024, 1024)
        # self.l4 = nn.Linear(1024, 1024)
        # self.l5 = nn.Linear(1024, 1024)
        # self.l6 = nn.Linear(1024, 1024)
        # self.l7 = nn.Linear(1024, 1024)
        # self.l8 = nn.Linear(1024, action_size)
        
        
        
        # Reduced model complexity for better performance and stability
        self.l1 = nn.Linear(295, 512)   # Input size: 295, reduced hidden layer size
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after the first layer
        self.l2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)  # Batch normalization after the second layer
        self.l3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)  # Batch normalization after the third layer
        self.l4 = nn.Linear(256, action_size)  # Output layer

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = self.l3(x)
        
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        # x = F.relu(self.l7(x))
        # x = self.l8(x)
        
        x = F.relu(self.bn1(self.l1(x)))  # ReLU activation + BatchNorm
        x = self.dropout(x)  # Dropout after the first layer
        
        x = F.relu(self.bn2(self.l2(x)))  # ReLU activation + BatchNorm
        x = self.dropout(x)  # Dropout after the second layer
        
        x = F.relu(self.bn3(self.l3(x)))  # ReLU activation + BatchNorm
        x = self.dropout(x)  # Dropout after the third layer
        
        x = self.l4(x)  # Output layer without activation
        
        
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.001   # 0.0005
        self.epsilon = epsilon_start   # 0.1
        self.buffer_size = 50000   # 10000
        self.batch_size =  64  #32
        self.action_size = 4  # <-- 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
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
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)
            qs = self.qnet(state)
        self.qnet.train()  # Set back to training mode after inference
        return qs.argmax().item() if np.random.rand() >= self.epsilon else np.random.choice(self.action_size)

        

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[torch.arange(len(action)).to(device), action]

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
        if agent.epsilon > epsilon_end:
            agent.epsilon *= epsilon_decay
            
        print(f"Epsilon: {agent.epsilon:.3f}")

        


episodes = 1000
sync_interval = 20
# env = gym.make('CartPole-v1')
env = sumo.make('cross.sumocfg', 'sumo')   # <-- [두번째 파라미터] : sumo를 cli버전으로 실행하려면 'sumo'로,  gui버전으로 실행하려면 'sumo-gui'로 설정
agent = DQNAgent()
reward_history = []

##########################▼▼GRAPH▼▼##############################
# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Start with empty data
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Live Updating Graph of Total Reward')
##########################▲▲GRAPH▲▲##############################

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # env.render()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
    # env.close() ??

    if episode % sync_interval == 0:
        agent.sync_qnet()

    if episode % 1 == 0 and episode > 0:
        reward_history.append(total_reward)
        print("episode :{}, total reward : {}".format(episode, total_reward))
         
        ##########################▼▼GRAPH▼▼##############################
        # Update the plot
        line.set_xdata(range(1, len(reward_history)+1))
        line.set_ydata(reward_history)

        # Dynamically adjust the x and y limits
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Apply the recalculated limits

        plt.draw()
        plt.pause(0.1)  # Pause to update the plot
        ##########################▲▲GRAPH▲▲##############################
        
    agent.decay_epsilon()
        


        
# Keep the plot open after the loop ends
plt.ioff()  # Turn off interactive mode
plt.show()