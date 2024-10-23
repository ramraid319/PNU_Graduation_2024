import torch.nn as nn
import torch.nn.functional as F

# class QNetwork(nn.Module):
#     def __init__(self, action_size):
#         super(QNetwork, self).__init__()

#         self.layers = nn.ModuleList()

#         # Input channels
#         in_channels = 16
#         out_channels = 16  # Starting output channels

#         # Reduce the number of convolutional layers
#         num_layers = 10  # Reduced from 20 to 10

#         for i in range(num_layers):
#             # Convolutional layer
#             conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             self.layers.append(conv)
#             self.layers.append(nn.BatchNorm2d(out_channels))
#             self.layers.append(nn.ReLU())

#             in_channels = out_channels  # Update input channels for the next layer

#             # Add Max Pooling every 4 layers and double the output channels
#             if (i + 1) % 4 == 0:
#                 self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#                 out_channels *= 2  # Double the output channels

#                 # Limit the doubling to avoid too many channels
#                 if out_channels > 64:  # cap the output channels at 64
#                     out_channels = 64


#         # Calculate the output size after all convolutions and pooling
#         self.fc1 = nn.Linear(118400, 512)  # Reduced fully connected layer size
#         self.fc2 = nn.Linear(512, action_size)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)

#         x = x.flatten(start_dim=1)
#         print("Size before fc1:", x.size())
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        # Input size: (4, 196, 144)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 49 * 36, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
    
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
