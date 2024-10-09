import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()

        # Input channels
        in_channels = 48
        out_channels = 16  # Starting output channels

        # Reduce the number of convolutional layers
        num_layers = 10  # Reduced from 20 to 10

        for i in range(num_layers):
            # Convolutional layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.layers.append(conv)
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())

            in_channels = out_channels  # Update input channels for the next layer

            # Add Max Pooling every 4 layers and double the output channels
            if (i + 1) % 4 == 0:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_channels *= 2  # Double the output channels

                # Limit the doubling to avoid too many channels
                if out_channels > 64:  # cap the output channels at 64
                    out_channels = 64

        # Calculate the output size after all convolutions and pooling
        self.fc1 = nn.Linear(64 * 200 * 150, 512)  # Reduced fully connected layer size
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)