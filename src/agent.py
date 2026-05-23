"""
agent.py — Agent DQN et variantes

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - __init__() : online_net, target_net, optimizer entièrement initialisés
  - select_action() : politique epsilon-greedy implémentée
  - update() : boucle complète avec gestion PER (IS weights) et ReplayBuffer
               calcule les TD errors et les retourne au buffer pour update_priorities()
  - sync_target() : implémentée
  - soft_reset() : NOUVEAU — réinitialisation partielle pour éviter la plasticité
                   catastrophique quand replay_ratio est élevé
                   Référence : D'Oro et al. (2023) Breaking the Replay Ratio Barrier

Référence : Mnih et al. (2015)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from networks import DQNNetwork


class DQNAgent:
    """
    Agent DQN classique.

    Deux réseaux :
    - online_net  : réseau entraîné à chaque step
    - target_net  : copie gelée, synchronisée toutes les N steps

    La loss est :
        L = E[(r + γ * max_a' Q_target(s', a') - Q_online(s, a))^2]
    pondérée par les IS weights si PER est activé.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update', 1000)
        self.step_count = 0

        hidden_dims = config.get('hidden_dims', [256, 256])
        dueling = config.get('dueling', False)

        # Réseau online — s'entraîne à chaque step
        self.online_net = DQNNetwork(
            obs_dim, action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(self.device)

        # Réseau target — copie gelée, synchronisée périodiquement
        self.target_net = DQNNetwork(
            obs_dim, action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target ne s'entraîne jamais

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.get('lr', 1e-3)
        )

        # Compteur pour soft_reset (plasticité catastrophique)
        self.soft_reset_freq = config.get('soft_reset_freq', 0)  # 0 = désactivé
        self.soft_reset_alpha = config.get('soft_reset_alpha', 0.8)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """
        Politique epsilon-greedy.
        - Avec probabilité epsilon : action aléatoire (exploration)
        - Sinon : argmax Q(s, a) (exploitation)
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
        return q_values.argmax().item()

    def update(self, buffer) -> Optional[float]:
        """
        Met à jour le réseau online à partir d'un batch du buffer.

        Compatible avec ReplayBuffer (pas de IS weights) et
        PrioritizedReplayBuffer (retourne indices + IS weights).

        Retourne la loss pour le logging, ou None si buffer pas assez plein.
        """
        if len(buffer) < self.batch_size:
            return None

        batch = buffer.sample(self.batch_size)

        # Convertir en tensors sur le bon device
        states      = torch.FloatTensor(batch['states']).to(self.device)
        actions     = torch.LongTensor(batch['actions']).to(self.device)
        rewards     = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones       = torch.FloatTensor(batch['dones']).to(self.device)
        is_weights  = torch.FloatTensor(batch['is_weights']).to(self.device)

        # Q(s, a) avec le réseau online — valeurs actuelles
        q_values = self.online_net(states)
        # Sélectionner uniquement les Q-values des actions effectuées
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Cibles : r + γ * max_a' Q_target(s', a')
        # torch.no_grad() : les cibles ne doivent pas avoir de gradients
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            # Si done=True, pas de récompense future
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # TD errors pour mettre à jour les priorités PER
        td_errors = (targets - q_values).detach().cpu().numpy()

        # Loss MSE pondérée par IS weights (correction du biais PER)
        loss = (is_weights * (q_values - targets) ** 2).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilité
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mettre à jour les priorités dans le buffer si PER
        if hasattr(buffer, 'update_priorities'):
            buffer.update_priorities(batch['indices'], np.abs(td_errors))

        self.step_count += 1

        # Synchroniser le target network périodiquement
        if self.step_count % self.target_update_freq == 0:
            self.sync_target()

        # Soft reset périodique si activé (contre plasticité catastrophique)
        # Référence : D'Oro et al. (2023) Breaking the Replay Ratio Barrier
        if self.soft_reset_freq > 0 and self.step_count % self.soft_reset_freq == 0:
            self.soft_reset()

        return loss.item()

    def sync_target(self) -> None:
        """Copie les poids de online_net vers target_net."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_reset(self) -> None:
        """
        Réinitialisation partielle du réseau online.
        Mélange les poids actuels avec une initialisation aléatoire.

        Formule : θ_new = α * θ_current + (1 - α) * θ_random

        Pourquoi : quand le replay ratio est élevé (beaucoup de réanalyse),
        le réseau peut perdre sa plasticité — sa capacité à apprendre
        de nouvelles informations. Le soft reset la restaure.

        NOUVEAU — pas dans le code original.
        Référence : D'Oro et al. (2023) Breaking the Replay Ratio Barrier
        """
        alpha = self.soft_reset_alpha
        # Créer un réseau temporaire avec initialisation aléatoire
        fresh_net = DQNNetwork(
            self.obs_dim, self.action_dim,
            hidden_dims=self.config.get('hidden_dims', [256, 256]),
            dueling=self.config.get('dueling', False)
        ).to(self.device)

        # Mélanger les poids
        with torch.no_grad():
            for param_current, param_fresh in zip(
                self.online_net.parameters(), fresh_net.parameters()
            ):
                param_current.data.copy_(
                    alpha * param_current.data + (1 - alpha) * param_fresh.data
                )

    def get_policy_params(self):
        """Retourne les paramètres de la politique — utilisé par le scheduler TRIGGERED."""
        return [p.data.cpu().numpy().copy()
                for p in self.online_net.parameters()]

    def save(self, path: str) -> None:
        """Sauvegarde le réseau online."""
        torch.save({
            'online_net':  self.online_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'step_count':  self.step_count,
        }, path)

    def load(self, path: str) -> None:
        """Charge un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
