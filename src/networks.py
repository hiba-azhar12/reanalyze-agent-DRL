
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# MLP 
# =============================================================================

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 activation: str = 'relu', output_activation: str = 'none'):
        super().__init__()

    
        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu':  nn.ELU(),
        }
        act_fn = act_map.get(activation, nn.ReLU())

       
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(type(act_fn)())  
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# DQNNetwork 
# =============================================================================

class DQNNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: list = None,
                 dueling: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.dueling = dueling
        self.action_dim = action_dim

        encoder_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims[:-1]:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, hidden_dims[-1]))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        if dueling:
            self.value_head = nn.Linear(hidden_dims[-1], 1)
            
            self.advantage_head = nn.Linear(hidden_dims[-1], action_dim)
        else:
            
            self.q_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)

        if self.dueling:
            value = self.value_head(features)          
            advantage = self.advantage_head(features)  
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
    
        return self.forward(obs).max(dim=-1, keepdim=True).values

# =============================================================================
# LatentModel : encodeur + model de dynamique (EfficientZero / DreamerV3)
# =============================================================================

class LatentModel(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next_latent(self, latent: torch.Tensor,
                            action_onehot: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.dynamics(x)


# =============================================================================
# consistency_loss
# =============================================================================

def consistency_loss(next_obs: torch.Tensor,
                     latent_t: torch.Tensor,
                     action_onehot: torch.Tensor,
                     latent_model: LatentModel) -> torch.Tensor:
   
    predicted_next = latent_model.predict_next_latent(latent_t, action_onehot)
    with torch.no_grad():
        target_next = latent_model.encode(next_obs)

    return F.mse_loss(predicted_next, target_next)