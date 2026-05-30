"""
networks.py — Architectures des réseaux de neurones

Contenu :
  - MLP           : réseau générique multi-couches
  - DQNNetwork    : réseau Q pour DQN (tête valeur + avantage optionnel)
  - LatentModel   : encodeur + modèle de dynamique (pour variantes)
  - consistency_loss : loss de consistance latente (EfficientZero)

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - MLP : __init__ et forward() entièrement implémentés avec nn.Sequential
  - DQNNetwork : encodeur + tête Q implémentés, dueling DQN supporté
  - DQNNetwork.get_value() retourne V(s) = max Q(s,a) — utilisé dans reanalyze()
  - LatentModel : encode() et predict_next_latent() implémentés
  - consistency_loss() implémentée — pénalise incohérence du modèle de dynamique

Référence : Mnih et al. (2015), Wang et al. (2024) EfficientZero V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# MLP — réseau générique
# =============================================================================

class MLP(nn.Module):
    """
    Réseau fully-connected générique.
    Utilisé comme brique de base pour DQNNetwork et LatentModel.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 activation: str = 'relu', output_activation: str = 'none'):
        super().__init__()

        # Choisir la fonction d'activation
        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu':  nn.ELU(),
        }
        act_fn = act_map.get(activation, nn.ReLU())

        # Construire les couches
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(type(act_fn)())  # nouvelle instance à chaque fois
            in_dim = h_dim

        # Couche de sortie
        layers.append(nn.Linear(in_dim, output_dim))

        # Activation de sortie optionnelle
        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# DQNNetwork — réseau Q principal
# =============================================================================

class DQNNetwork(nn.Module):
    """
    Réseau Q pour DQN et toutes les variantes.

    Architecture : encodeur MLP → tête Q-values (ou dueling : value + advantage)

    Pour LunarLander-v2 :
        obs_dim = 8, action_dim = 4, hidden = [256, 256]
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: list = None,
                 dueling: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.dueling = dueling
        self.action_dim = action_dim

        # Encodeur partagé : obs_dim → dernière couche cachée
        encoder_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims[:-1]:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        # Dernière couche cachée sans activation (la tête l'ajoute)
        encoder_layers.append(nn.Linear(in_dim, hidden_dims[-1]))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        if dueling:
            # Tête Value : un scalaire V(s)
            self.value_head = nn.Linear(hidden_dims[-1], 1)
            # Tête Advantage : un vecteur A(s, a) pour chaque action
            self.advantage_head = nn.Linear(hidden_dims[-1], action_dim)
        else:
            # Tête Q directe
            self.q_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Retourne les Q-values pour toutes les actions. Shape : (batch, action_dim)"""
        features = self.encoder(obs)

        if self.dueling:
            value = self.value_head(features)          # (batch, 1)
            advantage = self.advantage_head(features)  # (batch, action_dim)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            # Soustraire la moyenne stabilise l'entraînement
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Retourne V(s) = max_a Q(s,a). Utilisé dans la réanalyse.
        C'est avec cette fonction qu'on recalcule les cibles fraîches.
        """
        return self.forward(obs).max(dim=-1, keepdim=True).values

# =============================================================================
# LatentModel — encodeur + modèle de dynamique (EfficientZero / DreamerV3)
# =============================================================================

class LatentModel(nn.Module):
    """
    Encodeur d'observations + modèle de dynamique dans l'espace latent.

    Utilisé par :
      - EfficientZero : consistency_loss pour entraîner la cohérence du modèle
      - DreamerV3     : générer des trajectoires imaginées depuis des états réels

    Architecture :
      encode()              : obs  → latent  (MLP)
      predict_next_latent() : (latent, action_onehot) → latent suivant prédit
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encodeur : observation → espace latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Modèle de dynamique : (latent + action_onehot) → latent suivant
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (batch, obs_dim) → latent (batch, latent_dim)"""
        return self.encoder(obs)

    def predict_next_latent(self, latent: torch.Tensor,
                            action_onehot: torch.Tensor) -> torch.Tensor:
        """
        (latent, action_onehot) → latent prédit au pas suivant.

        Args :
            latent       : (batch, latent_dim)
            action_onehot: (batch, action_dim) — vecteur one-hot de l'action
        Returns :
            next_latent  : (batch, latent_dim)
        """
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.dynamics(x)


# =============================================================================
# consistency_loss — pénalise l'incohérence du modèle de dynamique
# =============================================================================

def consistency_loss(next_obs: torch.Tensor,
                     latent_t: torch.Tensor,
                     action_onehot: torch.Tensor,
                     latent_model: LatentModel) -> torch.Tensor:
    """
    Loss de consistance latente (EfficientZero).

    Objectif : le latent prédit par le modèle de dynamique doit correspondre
    au latent réellement encodé depuis l'observation suivante.

        L = MSE( predict_next_latent(z_t, a_t),  sg(encode(o_{t+1})) )

    sg() = stop_gradient — on ne propage pas le gradient à travers la cible.
    Cela évite l'effondrement (mode collapse) où les deux côtés convergent
    vers zéro ensemble.

    Args :
        next_obs     : observations au pas t+1, shape (batch, obs_dim)
        latent_t     : latents encodés au pas t,  shape (batch, latent_dim)
        action_onehot: actions one-hot au pas t,  shape (batch, action_dim)
        latent_model : instance de LatentModel

    Returns :
        loss scalaire
    """
    # Latent prédit par le modèle de dynamique (avec gradient)
    predicted_next = latent_model.predict_next_latent(latent_t, action_onehot)

    # Latent réel encodé depuis o_{t+1} — stop gradient (cible fixe)
    with torch.no_grad():
        target_next = latent_model.encode(next_obs)

    return F.mse_loss(predicted_next, target_next)