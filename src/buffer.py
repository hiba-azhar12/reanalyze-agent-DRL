"""
buffer.py — Gestion du replay buffer

TOUTES LES CORRECTIONS APPLIQUÉES (fusion Doc35 + Doc36) :
  [Bug 1] SumTree désync     : idx_before sauvegardé AVANT super().add()
  [Bug 2] beta_increment     : 0.001 → 0.0001 (beta=1.0 après 6000 steps au lieu de 600)
  [Bug 3] clamp masquant     : supprimé dans PER.sample() — masquait Bug 1 silencieusement
  [Fix 4] ReanalyzeBuffer.sample() : ajouté — retourne reanalyzed_targets via _state_to_target
  [Fix 5] Limite trajectoires : capacity//200 au lieu de capacity//10
  [Fix 6] update_td_errors() : nouvelle méthode — td_errors mis à jour depuis agent
  [Fix 7] _state_to_target   : index O(1) état→target, nettoyé dans smart_delete()
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# Classe 1 — SumTree
# =============================================================================

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# =============================================================================
# Classe 2 — ReplayBuffer
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action: int, reward: float,
            next_state, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            'states':             np.array(states,      dtype=np.float32),
            'actions':            np.array(actions,     dtype=np.int64),
            'rewards':            np.array(rewards,     dtype=np.float32),
            'next_states':        np.array(next_states, dtype=np.float32),
            'dones':              np.array(dones,       dtype=np.float32),
            'indices':            indices,
            'is_weights':         np.ones(batch_size,   dtype=np.float32),
            'reanalyzed_targets': None,
        }

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Classe 3 — PrioritizedReplayBuffer
# =============================================================================

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    [Bug 1] add() : idx_before sauvegardé AVANT super().add()
    [Bug 2] beta_increment : 0.0001 au lieu de 0.001
    [Bug 3] sample() : clamp supprimé
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.0001,  # [Bug 2] était 0.001
                 epsilon: float = 1e-6):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, state, action: int, reward: float,
            next_state, done: bool) -> None:
        # [Bug 1] sauvegarder AVANT super().add() qui avance self.position
        idx_before = self.position
        super().add(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, idx_before)

    def sample(self, batch_size: int) -> Dict:
        self._update_beta()

        indices = []
        priorities = []
        batch_indices = []

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data_idx = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch_indices.append(data_idx)

        N = len(self.buffer)
        sampling_probs = np.array(priorities) / self.tree.total()
        sampling_probs = np.clip(sampling_probs, 1e-8, 1.0)
        is_weights = (N * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()

        # [Bug 3] suppression du clamp min/max qui masquait Bug 1
        batch_indices = [int(i) for i in batch_indices]
        batch = [self.buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states':             np.array(states,      dtype=np.float32),
            'actions':            np.array(actions,     dtype=np.int64),
            'rewards':            np.array(rewards,     dtype=np.float32),
            'next_states':        np.array(next_states, dtype=np.float32),
            'dones':              np.array(dones,       dtype=np.float32),
            'indices':            np.array(indices),
            'is_weights':         np.array(is_weights,  dtype=np.float32),
            'reanalyzed_targets': None,
        }

    def update_priorities(self, indices: List[int],
                          td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def _update_beta(self) -> None:
        self.beta = min(1.0, self.beta + self.beta_increment)


# =============================================================================
# Classe 4 — ReanalyzeBuffer
# =============================================================================

class ReanalyzeBuffer(PrioritizedReplayBuffer):
    """
    [Fix 4] sample()           : retourne reanalyzed_targets depuis _state_to_target
    [Fix 5] limite trajectoires: capacity//200 au lieu de capacity//10
    [Fix 6] update_td_errors() : nouvelle méthode pour vraies erreurs TD
    [Fix 7] _state_to_target   : index O(1), nettoyé dans smart_delete()
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.0001,
                 w_age: float = 0.4, w_td: float = 0.4, w_redundancy: float = 0.2):
        super().__init__(capacity, alpha, beta, beta_increment)
        self.trajectories = {}
        self.timestamps = {}
        self.td_errors = {}
        self.current_step = 0
        self.w_age = w_age
        self.w_td = w_td
        self.w_redundancy = w_redundancy
        self._traj_counter = 0
        # [Fix 7] index rapide état→target
        self._state_to_target = {}

    def add_trajectory(self, trajectory: List[Dict]) -> None:
        traj_id = self._traj_counter
        self._traj_counter += 1

        steps = []
        for step in trajectory:
            state = np.array(step['state'], dtype=np.float32)
            steps.append({
                'state':      state,
                'action':     int(step['action']),
                'reward':     float(step['reward']),
                'next_state': np.array(step['next_state'], dtype=np.float32),
                'done':       bool(step['done']),
                'target':     0.0,
            })
            super().add(step['state'], step['action'], step['reward'],
                        step['next_state'], step['done'])

        self.trajectories[traj_id] = steps
        self.timestamps[traj_id] = self.current_step
        self.td_errors[traj_id] = 1.0

        # [Fix 5] capacity//200 cohérent avec ~200 steps/épisode LunarLander
        max_trajectories = max(10, self.capacity // 200)
        if len(self.trajectories) > max_trajectories:
            self.smart_delete(n=1)

    def sample(self, batch_size: int) -> Dict:
        # [Fix 4] méthode manquante — retourne les reanalyzed_targets
        batch = super().sample(batch_size)

        if len(self._state_to_target) == 0:
            batch['reanalyzed_targets'] = None
            return batch

        reanalyzed_targets = []
        found_any = False
        for state in batch['states']:
            key = state.tobytes()
            target = self._state_to_target.get(key, None)
            if target is not None and target != 0.0:
                reanalyzed_targets.append(target)
                found_any = True
            else:
                reanalyzed_targets.append(0.0)

        batch['reanalyzed_targets'] = (
            np.array(reanalyzed_targets, dtype=np.float32) if found_any else None
        )
        return batch

    def get_trajectories(self, indices: List[int]) -> List[Dict]:
        result = []
        for idx in indices:
            if idx in self.trajectories:
                result.append({'steps': self.trajectories[idx], 'id': idx})
            else:
                result.append({'steps': [], 'id': idx})
        return result

    def update_targets(self, indices: List[int],
                       new_targets: np.ndarray) -> None:
        for idx, targets in zip(indices, new_targets):
            if idx in self.trajectories and len(targets) > 0:
                for t, step in enumerate(self.trajectories[idx]):
                    if t < len(targets):
                        step['target'] = float(targets[t])
                        # [Fix 7] mettre à jour l'index rapide
                        key = step['state'].tobytes()
                        self._state_to_target[key] = float(targets[t])

    def update_td_errors(self, traj_id: int, td_error: float) -> None:
        # [Fix 6] appelée depuis train.py après chaque update agent
        if traj_id in self.td_errors:
            self.td_errors[traj_id] = float(abs(td_error))

    def compute_staleness_score(self, idx: int) -> float:
        if idx not in self.trajectories:
            return 0.0
        age = self.current_step - self.timestamps.get(idx, 0)
        max_age = max(1, self.current_step)
        age_score = age / max_age
        td_err = self.td_errors.get(idx, 1.0)
        td_score = 1.0 / (td_err + 1e-6)
        all_td = [1.0 / (e + 1e-6) for e in self.td_errors.values()]
        if len(all_td) > 1:
            td_min, td_max = min(all_td), max(all_td)
            if td_max > td_min:
                td_score = (td_score - td_min) / (td_max - td_min)
            else:
                td_score = 0.0
        ages = [self.current_step - self.timestamps.get(i, 0) for i in self.trajectories]
        if len(ages) > 1:
            similar = sum(1 for a in ages if abs(a - age) < max_age * 0.1)
            redundancy_score = similar / len(ages)
        else:
            redundancy_score = 0.0
        return (self.w_age * age_score
                + self.w_td * td_score
                + self.w_redundancy * redundancy_score)

    def smart_delete(self, n: int = 1) -> None:
        if len(self.trajectories) == 0:
            return
        scores = {idx: self.compute_staleness_score(idx) for idx in self.trajectories}
        to_delete = sorted(scores, key=scores.get, reverse=True)[:n]
        for idx in to_delete:
            # [Fix 7] nettoyer aussi _state_to_target
            if idx in self.trajectories:
                for step in self.trajectories[idx]:
                    self._state_to_target.pop(step['state'].tobytes(), None)
            del self.trajectories[idx]
            if idx in self.timestamps:
                del self.timestamps[idx]
            if idx in self.td_errors:
                del self.td_errors[idx]

    def get_all_trajectory_ids(self) -> List[int]:
        return list(self.trajectories.keys())

    def get_staleness_stats(self) -> Dict:
        if not self.timestamps:
            return {'mean_age': 0, 'max_age': 0, 'n_trajectories': 0}
        ages = [self.current_step - t for t in self.timestamps.values()]
        return {
            'mean_age':       float(np.mean(ages)),
            'max_age':        float(np.max(ages)),
            'min_age':        float(np.min(ages)),
            'n_trajectories': len(ages),
        }

    def _increment_step(self) -> None:
        self.current_step += 1