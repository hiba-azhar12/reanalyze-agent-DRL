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
# LatentModel — encodeur + modèle de dynamique (pour variante EfficientZero)
# =============================================================================

class LatentModel(nn.Module):
    """
    Encodeur d'état + modèle de transition dans l'espace latent.

    h_t = encoder(o_t)
    h_t+1_pred = dynamics(h_t, a_t)

    La loss de consistance latente pénalise :
        MSE(h(o_t+1), g(h(o_t), a_t))

    Référence : Wang et al. (2024) — EfficientZero V2
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 latent_dim: int = 64, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encodeur : observation → espace latent
        self.encoder = MLP(obs_dim, hidden_dims, latent_dim)

        # Modèle de dynamique : (latent + action one-hot) → latent suivant
        self.dynamics = MLP(latent_dim + action_dim, hidden_dims, latent_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode une observation dans l'espace latent."""
        return self.encoder(obs)

    def predict_next_latent(self, latent: torch.Tensor,
                            action: torch.Tensor) -> torch.Tensor:
        """
        Prédit l'état latent suivant.
        action doit être one-hot, shape (batch, action_dim).
        """
        x = torch.cat([latent, action], dim=-1)
        return self.dynamics(x)


def consistency_loss(obs_t1: torch.Tensor, latent_t: torch.Tensor,
                     action_t: torch.Tensor, model: LatentModel) -> torch.Tensor:
    """
    Loss de consistance latente (EfficientZero V2) :
        L = MSE(h(o_t+1), g(h(o_t), a_t))

    Pénalise l'incohérence entre la dynamique apprise et les vraies observations.
    À ajouter à la loss principale avec un coefficient λ.

    .detach() sur h_real : on ne veut pas backpropager dans l'encodeur
    depuis cette loss — on veut juste que les prédictions de dynamique
    convergent vers les vraies encodings.
    """
    h_real = model.encode(obs_t1)                               # ce qui s'est vraiment passé
    h_pred = model.predict_next_latent(latent_t, action_t)     # ce que le modèle prédit
    return F.mse_loss(h_pred, h_real.detach())
