"""
agent.py — Agent DQN et variantes

CORRECTIONS FUSIONNEES des deux versions :
  [C1] gradient clipping  : max_norm configurable depuis yaml via grad_clip
                            defaut 10.0 pour DQN vanilla, 1.0 pour PER/Reanalyze
  [C2] soft update EMA    : target network via EMA si use_soft_update=True
  [C3] soft_reset alpha   : alpha croissant avec le temps
  [C4] valid_mask         : ignore les targets reanalysees a 0.0
  [C5] reanalyze_alpha    : defaut 0.8
  [C6] retour tuple       : retourne (loss, mean_td_error)
  [C7] update_td_errors   : appele directement dans update()

Reference : Mnih et al. (2015), Schaul et al. (2016), D'Oro et al. (2023)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple

from networks import DQNNetwork


class DQNAgent:

    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.config     = config
        self.device     = torch.device(config.get('device', 'cpu'))
        self.gamma      = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update', 1000)
        self.step_count = 0

        hidden_dims = config.get('hidden_dims', [256, 256])
        dueling     = config.get('dueling', False)

        self.online_net = DQNNetwork(
            obs_dim, action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(self.device)

        self.target_net = DQNNetwork(
            obs_dim, action_dim,
            hidden_dims=hidden_dims,
            dueling=dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config.get('lr', 1e-3)
        )

        self.soft_reset_freq  = config.get('soft_reset_freq', 0)
        self.soft_reset_alpha = config.get('soft_reset_alpha', 0.8)

        self.use_soft_update = config.get('use_soft_update', False)
        self.tau             = config.get('tau', 0.005)

        # [C1] grad_clip configurable depuis yaml
        # DQN vanilla : grad_clip=10.0 (original Mnih et al.)
        # PER/Reanalyze : grad_clip=1.0 (plus stable avec IS weights)
        self.grad_clip = config.get('grad_clip', 10.0)
        
        ##here efficientZero
        self.use_consistency = config.get('use_consistency_loss', False)
        self.consistency_weight = config.get('consistency_loss_weight', 0.1)

        ##==================================
        ##here DreamerV3
        self.use_dreamer = config.get('use_dreamer', False)
        self.latent_model = None
        if self.use_consistency or self.use_dreamer:
            from networks import LatentModel
            latent_dim = config.get('latent_dim', 64)
            self.latent_model = LatentModel(
                obs_dim, action_dim, latent_dim=latent_dim
            ).to(self.device)
            # Optimizer inclut toujours latent_model si présent
            self.optimizer = optim.Adam(
                list(self.online_net.parameters()) +
                list(self.latent_model.parameters()),
                lr=config.get('lr', 1e-3)
            )
        ##================================

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
        return q_values.argmax().item()

    def update(self, buffer) -> Optional[Tuple[float, float]]:
        if len(buffer) < self.batch_size:
            return None

        batch = buffer.sample(self.batch_size)

        states      = torch.FloatTensor(batch['states']).to(self.device)
        actions     = torch.LongTensor(batch['actions']).to(self.device)
        rewards     = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones       = torch.FloatTensor(batch['dones']).to(self.device)
        is_weights  = torch.FloatTensor(batch['is_weights']).to(self.device)

        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            targets_fresh = rewards + self.gamma * next_q_values * (1 - dones)

        reanalyze_alpha = self.config.get('reanalyze_alpha', 0.8)
        if 'reanalyzed_targets' in batch and batch['reanalyzed_targets'] is not None:
            reanalyzed = torch.FloatTensor(batch['reanalyzed_targets']).to(self.device)
            valid_mask = (reanalyzed != 0.0).float()
            targets = (valid_mask * (reanalyze_alpha * reanalyzed
                                     + (1 - reanalyze_alpha) * targets_fresh)
                       + (1 - valid_mask) * targets_fresh)
        else:
            targets = targets_fresh

        td_errors     = (targets - q_values).detach().cpu().numpy()
        mean_td_error = float(np.mean(np.abs(td_errors)))

        loss = (is_weights * (q_values - targets) ** 2).mean()

        #here efficientZero
        if self.use_consistency and batch.get('next_states') is not None:
            from networks import consistency_loss
            latent_t = self.latent_model.encode(states)
            actions_onehot = torch.zeros(
                states.shape[0], self.action_dim
            ).to(self.device)
            actions_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
            cons_loss = consistency_loss(
                next_states, latent_t, actions_onehot, self.latent_model
            )
            loss = loss + self.consistency_weight * cons_loss
        #==========================================
        self.optimizer.zero_grad()
        loss.backward()
        # [C1] grad_clip depuis config — pas hardcode
        params_to_clip = list(self.online_net.parameters())
        if self.latent_model is not None:
            params_to_clip += list(self.latent_model.parameters())
        nn.utils.clip_grad_norm_(params_to_clip, max_norm=self.grad_clip)
        self.optimizer.step()

        if hasattr(buffer, 'update_priorities'):
            buffer.update_priorities(batch['indices'], np.abs(td_errors))

        if hasattr(buffer, 'update_td_errors') and hasattr(buffer, '_traj_counter'):
            last_traj_id = buffer._traj_counter - 1
            buffer.update_td_errors(last_traj_id, mean_td_error)

        self.step_count += 1

        if self.use_soft_update:
            self._soft_update_target()
        elif self.step_count % self.target_update_freq == 0:
            self.sync_target()

        if self.soft_reset_freq > 0 and self.step_count % self.soft_reset_freq == 0:
            self.soft_reset()

        return loss.item(), mean_td_error

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for p_online, p_target in zip(
                self.online_net.parameters(),
                self.target_net.parameters()
            ):
                p_target.data.copy_(
                    self.tau * p_online.data + (1 - self.tau) * p_target.data
                )

    def soft_reset(self) -> None:
        total_steps = self.config.get('total_steps', 500000)
        progress    = min(1.0, self.step_count / total_steps)
        alpha       = self.soft_reset_alpha + (1.0 - self.soft_reset_alpha) * progress

        fresh_net = DQNNetwork(
            self.obs_dim, self.action_dim,
            hidden_dims=self.config.get('hidden_dims', [256, 256]),
            dueling=self.config.get('dueling', False)
        ).to(self.device)

        with torch.no_grad():
            for param_current, param_fresh in zip(
                self.online_net.parameters(), fresh_net.parameters()
            ):
                param_current.data.copy_(
                    alpha * param_current.data + (1 - alpha) * param_fresh.data
                )

    def get_policy_params(self):
        return [p.data.cpu().numpy().copy()
                for p in self.online_net.parameters()]

    def save(self, path: str) -> None:
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']