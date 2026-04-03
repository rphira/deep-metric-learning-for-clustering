"""
Model definition
"""
import torch
import torch.nn as nn

class TripletModel(nn.Module):
    """
    Triplet-based Deep Clustering model using encoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super(TripletModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent