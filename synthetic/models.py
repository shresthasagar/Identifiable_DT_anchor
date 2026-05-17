import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network for GAN training to distinguish between
    true source distributions and extracted sources from the autoencoder.
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.2):
        super(Discriminator, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x) 