import torch 
import numpy as np
import torch.nn as nn
from .QNetwork import *

class Agent:
    def __init__(self, action_size, device):
        self.action_size = action_size  # <-- Action size
        self.device = device
        self.qnet = QNet(self.action_size).float().to(device)  # Q-Network

    def get_action(self, state):
        """Select the action using the trained Q-network."""
        self.qnet.eval()  # Ensure evaluation mode for inference
        with torch.no_grad():  # Disable gradient calculation
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(self.device)
            qs = self.qnet(state)  # Get Q-values
        return qs.argmax().item()  # Return action with the highest Q-value

    # Load model for inference only
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])  # Load the trained Q-network
        self.qnet.eval()  # Set Q-network to evaluation mode for inference
        print(f"Model loaded for inference from {path}")
