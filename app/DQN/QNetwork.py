import torch
import torch.nn as nn
import torch.nn.functional as F

# class QNet(nn.Module):
#     def __init__(self, action_size):
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # Input channels: 4 (color), Output: 32
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # Output: 64
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # Output: 128
#         self.bn3 = nn.BatchNorm2d(128)

        
#         # Adjust size after conv layers based on input dimensions
#         self.fc1 = nn.Linear(128 * 25 * 75, 512)  # 11x8 comes from the conv layers' output size
#         self.fc2 = nn.Linear(512, action_size)  # Output size is the number of actions

#     def forward(self, x):
#         # x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
        
#         # print("Shape before flattening:", x.shape)  # Debug print

#         x = torch.flatten(x, 1)  # Flatten the convolution output
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)  # Output  layer
    
#         return x


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # Convolutional layers to process 200x600x1 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2)  # Output: (32, 50, 150)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: (64, 25, 75)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output: (128, 13, 38)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: (256, 7, 19)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(29184, 1024)  # First fully connected layer
        self.fc2 = nn.Linear(1024, 512)  # Second fully connected layer
        self.fc3 = nn.Linear(512, action_size)  # Output layer for Q-values (4 actions)

    def forward(self, x):
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output Q-values for each action
        x = self.fc3(x)
        
        return x
    

# class QNet(nn.Module):
#     def __init__(self, action_size):
#         super().__init__()

#         # Convolutional layers to process 200x600x1 input
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)

#         self.fc1 = nn.Linear(512 * 12 * 16, 2048)  # First fully connected layer
#         self.fc2 = nn.Linear(2048, action_size)  # Output layer for Q-values (4 actions)

#     def forward(self, x):
#         # Apply convolutional layers with ReLU activations
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))

#         # Flatten the output from convolutional layers
#         x = x.view(x.size(0), -1)

#         # Fully connected layers with ReLU activations
#         x = F.relu(self.fc1(x))

#         # Output Q-values for each action
#         x = self.fc2(x)
        
#         return x

    