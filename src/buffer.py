"""
buffer.py — Gestion du replay buffer

4 classes dans l'ordre de complexité croissante :
  1. SumTree              — structure d'arbre pour échantillonnage O(log N)
  2. ReplayBuffer         — buffer FIFO basique (DQN vanilla)
  3. PrioritizedReplayBuffer — buffer avec priorités TD (DQN + PER)
  4. ReanalyzeBuffer      — buffer avec réanalyse et suppression intelligente

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - SumTree : entièrement implémenté (_propagate, _retrieve, add, update, get)
  - ReplayBuffer : add() et sample() implémentés avec écrasement FIFO
  - PrioritizedReplayBuffer : add(), sample() avec IS weights, update_priorities()
  - ReanalyzeBuffer : add_trajectory(), get_trajectories(), update_targets(),
                      compute_staleness_score(), smart_delete() implémentés
  - ReanalyzeBuffer stocke aussi les targets TD pour pouvoir les mettre à jour
  - compute_staleness_score() utilise score composite : âge + TD inverse + redondance
  - smart_delete() supprime selon score composite au lieu du FIFO

Référence : Schaul et al. (2016), Schrittwieser et al. (2021)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# Classe 1 — SumTree
# =============================================================================

class SumTree:
    """
    Arbre binaire où chaque nœud = somme de ses enfants.
    Permet un échantillonnage proportionnel aux priorités en O(log N).

    Structure :
        - self.tree  : tableau numpy de taille 2*capacity - 1
        - self.data  : tableau des transitions stockées (feuilles seulement)
        - self.write : pointeur circulaire sur la prochaine feuille à écrire
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propage le changement de priorité vers la racine."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Trouve la feuille correspondant à la valeur s."""
        left = 2 * idx + 1
        right = left + 1
        # Si on est une feuille, on retourne cet index
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Retourne la somme totale des priorités (racine de l'arbre)."""
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        """Ajoute une transition avec sa priorité."""
        # Index dans le tableau tree (les feuilles commencent à capacity-1)
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        """Met à jour la priorité d'une feuille."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Échantillonne une transition pour la valeur s.
        Retourne (idx_dans_tree, priorité, transition).
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# =============================================================================
# Classe 2 — ReplayBuffer
# =============================================================================

class ReplayBuffer:
    """
    Buffer FIFO basique pour DQN vanilla.
    Échantillonnage uniforme.

    Stocke des tuples (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action: int, reward: float,
            next_state, done: bool) -> None:
        """Ajoute une transition. Écrase la plus ancienne si plein."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        """
        Échantillonne batch_size transitions uniformément.
        Retourne un dict avec clés : states, actions, rewards, next_states, dones
        """
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
            'reanalyzed_targets': None,  # pas de réanalyse pour DQN vanilla
        }

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Classe 3 — PrioritizedReplayBuffer
# =============================================================================

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer avec priorités TD (Prioritized Experience Replay).
    Hérite de ReplayBuffer, ajoute SumTree + importance sampling weights.

    Référence : Schaul et al. (2016) — Prioritized Experience Replay
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        alpha : degré de priorisation (0 = uniforme, 1 = plein PER)
        beta  : correction IS weights (monte de beta vers 1 pendant l'entraînement)
        """
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, state, action: int, reward: float,
            next_state, done: bool) -> None:
        """Ajoute avec priorité maximale courante (optimiste)."""
        # Ajouter dans le buffer parent
        super().add(state, action, reward, next_state, done)
        # Ajouter dans le SumTree avec la priorité max (optimiste)
        # On stocke l'index de position pour retrouver la transition
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, self.position - 1)

    def sample(self, batch_size: int) -> Dict:
        """
        Échantillonnage proportionnel aux priorités.
        Retourne aussi indices (pour update_priorities) et is_weights.
        """
        self._update_beta()

        indices = []
        priorities = []
        batch_indices = []

        # Diviser [0, total] en batch_size segments égaux
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data_idx = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch_indices.append(data_idx)

        # Calcul des IS weights
        N = len(self.buffer)
        sampling_probs = np.array(priorities) / self.tree.total()
        # Clamp pour éviter division par zéro
        sampling_probs = np.clip(sampling_probs, 1e-8, 1.0)
        is_weights = (N * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()  # Normaliser

        # Récupérer les transitions
        batch_indices = [min(max(0, int(i)), len(self.buffer) - 1) for i in batch_indices]
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
            'reanalyzed_targets': None,  # sera rempli par ReanalyzeBuffer
        }

    def update_priorities(self, indices: List[int],
                          td_errors: np.ndarray) -> None:
        """Met à jour les priorités après un update du réseau."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def _update_beta(self) -> None:
        """Incrémente beta vers 1 (appelé à chaque sample)."""
        self.beta = min(1.0, self.beta + self.beta_increment)


# =============================================================================
# Classe 4 — ReanalyzeBuffer
# =============================================================================

class ReanalyzeBuffer(PrioritizedReplayBuffer):
    """
    Buffer avec réanalyse et suppression intelligente.
    Hérite de PrioritizedReplayBuffer.

    Fonctionnalités supplémentaires :
    - Stockage des trajectoires complètes (pas juste (s,a,r,s',done))
    - Calcul du staleness (âge × faible erreur TD)
    - Suppression intelligente : score = w1*âge + w2*(1/td_error) + w3*redondance
    - Mise à jour des cibles après réanalyse

    MODIFICATIONS :
    - trajectories stocke chaque step avec son état, action, reward, next_state,
      done ET sa target TD courante (recalculée après réanalyse)
    - timestamps enregistre le step d'ajout pour calculer l'âge
    - td_errors stocke la dernière erreur TD connue pour la priorisation
    - smart_delete() utilise score composite au lieu du FIFO

    Référence : Schrittwieser et al. (2021) — MuZero Reanalyze
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001,
                 w_age: float = 0.4, w_td: float = 0.4, w_redundancy: float = 0.2):
        """
        w_age        : poids du critère âge dans le score de suppression
        w_td         : poids du critère erreur TD (inverse)
        w_redundancy : poids du critère redondance (similarité cosine)
        """
        super().__init__(capacity, alpha, beta, beta_increment)
        self.trajectories = {}       # idx -> liste de steps {state, action, reward, next_state, done, target}
        self.timestamps = {}         # idx -> step d'ajout
        self.td_errors = {}          # idx -> dernière erreur TD connue
        self.current_step = 0
        self.w_age = w_age
        self.w_td = w_td
        self.w_redundancy = w_redundancy
        self._traj_counter = 0       # compteur unique pour les trajectoires

    def add_trajectory(self, trajectory: List[Dict]) -> None:
        """
        Ajoute une trajectoire complète.
        trajectory : liste de dicts {state, action, reward, next_state, done}

        Chaque step est aussi ajouté dans le buffer parent (s,a,r,s',done)
        pour que l'agent puisse faire des updates normaux.
        La trajectoire complète est stockée dans self.trajectories pour la réanalyse.
        """
        traj_id = self._traj_counter
        self._traj_counter += 1

        # Stocker la trajectoire avec targets initiales = 0 (seront calculées après)
        steps = []
        for step in trajectory:
            steps.append({
                'state':      np.array(step['state'],      dtype=np.float32),
                'action':     int(step['action']),
                'reward':     float(step['reward']),
                'next_state': np.array(step['next_state'], dtype=np.float32),
                'done':       bool(step['done']),
                'target':     0.0,  # sera recalculé par reanalyze()
            })
            # Aussi ajouter chaque transition dans le buffer parent
            super().add(step['state'], step['action'], step['reward'],
                        step['next_state'], step['done'])

        self.trajectories[traj_id] = steps
        self.timestamps[traj_id] = self.current_step
        self.td_errors[traj_id] = 1.0  # priorité initiale optimiste

        # Supprimer les plus vieilles trajectoires si buffer trop plein
        if len(self.trajectories) > self.capacity // 10:
            self.smart_delete(n=1)

    def get_trajectories(self, indices: List[int]) -> List[Dict]:
        """
        Retourne les trajectoires complètes pour les indices donnés.
        indices : liste d'IDs de trajectoires (pas indices SumTree)
        """
        result = []
        for idx in indices:
            if idx in self.trajectories:
                result.append({'steps': self.trajectories[idx], 'id': idx})
            else:
                # Trajectoire non trouvée : retourner une trajectoire vide
                result.append({'steps': [], 'id': idx})
        return result

    def update_targets(self, indices: List[int],
                       new_targets: np.ndarray) -> None:
        """
        Met à jour les cibles TD après réanalyse.
        C'est ici que les vieilles targets périmées sont remplacées par des targets fraîches.
        new_targets : liste d'arrays, un array de targets par trajectoire
        """
        for idx, targets in zip(indices, new_targets):
            if idx in self.trajectories and len(targets) > 0:
                # self.trajectories[idx] est directement une liste de steps (pas un dict)
                for t, step in enumerate(self.trajectories[idx]):
                    if t < len(targets):
                        step['target'] = float(targets[t])

    def compute_staleness_score(self, idx: int) -> float:
        """
        Score de suppression composite :
        score = w_age * âge_normalisé
              + w_td  * (1 / td_error_normalisé)   ← faible erreur = moins utile
              + w_redundancy * redondance_estimée

        Plus le score est élevé, plus la trajectoire est candidate à la suppression.

        CONTRIBUTION ORIGINALE :
        Au lieu du FIFO classique (supprimer le plus vieux), on supprime
        la trajectoire qui apporte le moins d'information :
        - Vieille ET maîtrisée (faible TD) = inutile même après réanalyse
        - Similaire aux autres (redondante) = n'apporte pas de diversité
        """
        if idx not in self.trajectories:
            return 0.0

        # Âge normalisé [0, 1]
        age = self.current_step - self.timestamps.get(idx, 0)
        max_age = max(1, self.current_step)
        age_score = age / max_age

        # Score TD inverse : faible erreur TD = moins utile à réapprendre
        td_err = self.td_errors.get(idx, 1.0)
        td_score = 1.0 / (td_err + 1e-6)
        # Normaliser entre 0 et 1
        all_td = [1.0 / (e + 1e-6) for e in self.td_errors.values()]
        if len(all_td) > 1:
            td_min, td_max = min(all_td), max(all_td)
            if td_max > td_min:
                td_score = (td_score - td_min) / (td_max - td_min)
            else:
                td_score = 0.0

        # Redondance estimée : proportion de trajectoires avec âge similaire
        # (proxy simple — une implémentation avancée utiliserait un arbre KD)
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
        """
        Supprime les n trajectoires avec le score de suppression le plus élevé.
        Alternative intelligente au FIFO classique.

        DIFFÉRENCE AVEC FIFO :
        FIFO supprime toujours le plus vieux.
        smart_delete supprime celui qui apporte le moins d'information,
        même si ce n'est pas le plus vieux.
        """
        if len(self.trajectories) == 0:
            return

        scores = {idx: self.compute_staleness_score(idx)
                  for idx in self.trajectories}
        # Trier par score décroissant et supprimer les n premiers
        to_delete = sorted(scores, key=scores.get, reverse=True)[:n]
        for idx in to_delete:
            del self.trajectories[idx]
            if idx in self.timestamps:
                del self.timestamps[idx]
            if idx in self.td_errors:
                del self.td_errors[idx]

    def get_all_trajectory_ids(self) -> List[int]:
        """Retourne tous les IDs de trajectoires dans le buffer."""
        return list(self.trajectories.keys())

    def get_staleness_stats(self) -> Dict:
        """
        Retourne des statistiques sur le staleness du buffer.
        Utilisé dans evaluate.py pour la métrique staleness moyen.
        """
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
