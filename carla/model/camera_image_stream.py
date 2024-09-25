import sys
import os
import glob
import numpy as np
import cv2
import random
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Simulator 
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

SECONDS_PER_EPISODE = 10

# Model
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

REPLAY_BUFFER = 5000
MIN_REPLAY_BUFFER = 1000

MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE

MODEL_NAME = "Sequential"

EPISODES = 12000
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

class IntersectionMonitor:
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Camera settings
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.cameras = []
        self.actor_list = []

        # Store the images from each camera
        self.camera_images = [None, None, None, None]

    def setup_cameras(self, camera_positions):
        """Set up four cameras at the given positions."""
        for i, pos in enumerate(camera_positions):
            cam_bp = self.blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{self.im_width}")
            cam_bp.set_attribute("image_size_y", f"{self.im_height}")
            cam_bp.set_attribute("fov", "110")
            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                                        carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]))
            camera = self.world.spawn_actor(cam_bp, transform)
            self.actor_list.append(camera)
            self.cameras.append(camera)
            camera.listen(lambda data, idx=i: self.process_image(data, idx))

    def process_image(self, image, camera_index):
        """Process the image from the camera and store it."""
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]  # RGB 값 추출
        i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)  # 필요하다면 흑백 이미지로 변환
        i3 = i3 / 255.0  # 정규화
        self.camera_images[camera_index] = i3

    def reset(self):
        """Reset the camera images and set up the environment."""
        self.camera_images = [None, None, None, None]
        # Set up initial state if necessary

    def destroy_actors(self):
        """Destroy all actors."""
        for actor in self.actor_list:
            actor.destroy()

    def get_camera_images(self):
        """Return the latest images from all cameras."""
        return self.camera_images

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

        state = torch.tensor(np.stack([x[0] for x in data]), dtype=torch.float32).to(DEVICE)
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int32)).to(DEVICE)
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).to(DEVICE)
        next_state = torch.tensor(np.stack([x[3] for x in data]), dtype=torch.float32).to(DEVICE)
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).to(DEVICE)
        return state, action, reward, next_state, done

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # Reduced model complexity for better performance and stability
        self.l1 = nn.Linear(114, 512)   # Input size: 38 x 3 = 114, reduced hidden layer size
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after the first layer
        self.l2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)  # Batch normalization after the second layer
        self.l3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)  # Batch normalization after the third layer
        self.l4 = nn.Linear(256, action_size)  # Output layer

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
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
        self.lr = 0.001
        self.epsilon = EPSILON
        self.buffer_size = REPLAY_BUFFER
        self.batch_size =  TRAINING_BATCH_SIZE
        self.action_size = 4  # <-- 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).float().to(DEVICE)
        self.qnet_target = QNet(self.action_size).float().to(DEVICE)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        self.qnet.eval()  # Set to evaluation mode to prevent batch norm errors
        with torch.no_grad():  # Disable gradient calculation for action selection
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            qs = self.qnet(state)
        self.qnet.train()  # Set back to training mode after inference
        return qs.argmax().item() if np.random.rand() >= self.epsilon else np.random.choice(self.action_size)

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) > MIN_REPLAY_BUFFER:
            agent.update(state, action, reward, next_state, done)

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[torch.arange(len(action)).to(DEVICE), action]

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
        if agent.epsilon > MIN_EPSILON:
            agent.epsilon *= EPSILON_DECAY
            
        print(f"Epsilon: {agent.epsilon:.3f}")

# Example usage:
if __name__ == "__main__":
    # Define the camera positions [x, y, z, pitch, yaw, roll] for four cameras
    camera_positions = [
        [-60, 20, 10, -40, 180, 0],  # Camera 1
        [-45, 35, 10, -40, 90, 0],  # Camera 2
        [-45, 20, 10, -40, 0, 0],  # Camera 3
        [-45, 10, 10, -40, -90, 0]   # Camera 4
    ]

    # Create an instance of IntersectionMonitor and set up the cameras
    monitor = IntersectionMonitor()
    monitor.setup_cameras(camera_positions)

    agent = DQNAgent()
    reward_history = []

    for episode in range(EPISODES):
        if episode % 10 == 0:  # 10 에피소드마다 타겟 네트워크 동기화
            agent.sync_qnet()
        images = monitor.get_camera_images()
        
        # 이미지를 하나의 state로 변환
        if all(img is not None for img in images):  # 모든 카메라에서 이미지가 받아진 경우에만
            state = np.concatenate([img.flatten() for img in images])
            action = agent.get_action(state)
            # Action을 환경에 적용하고, 그에 따른 보상을 계산해야 합니다.
        else:
            continue  # 이미지가 제대로 수신되지 않았다면 스킵

    # Run the monitoring for a certain time
    try:
        for frame in range(10):  # Save images for 10 frames
            images = monitor.get_camera_images()
            for idx, img in enumerate(images):
                if img is not None:
                    # Save the image to a file
                    cv2.imwrite(f"_out/camera_{idx+1}_frame_{frame+1}.png", img)
            time.sleep(1)  # Wait for 1 second before capturing the next frame
    finally:
        monitor.destroy_actors()
