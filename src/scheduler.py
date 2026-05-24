"""
scheduler.py — Scheduler de réanalyse

FUSION Doc40 + Doc41 — toutes les corrections :
  [S1] step_count      : skip si LAZY (Doc40)
  [S2] TRIGGERED       : old_params mis à jour SEULEMENT après réanalyse (Doc40)
  [S3] get_k_steps()   : _last_k_used tracké pour log_stats (Doc40)
  [S4] reanalyze_ratio : 0.25 CPU / 0.8 GPU (Doc40)
  [S5] log_stats()     : loggue _last_k_used (Doc40)
  [+]  period défaut   : 1000 au lieu de 100 (Doc41) — 500 réanalyses/500k steps

Référence : ReZero (ICLR 2025), Adaptive Offline Data Replay (ICLR 2024)
"""

import numpy as np
from enum import Enum
from typing import Optional


class ReanalyzeMode(Enum):
    CONTINUOUS = "continuous"
    PERIODIC   = "periodic"
    TRIGGERED  = "triggered"
    LAZY       = "lazy"


class ReanalyzeScheduler:

    def __init__(self, mode: str, config: dict):
        self.mode   = ReanalyzeMode(mode)
        self.config = config
        self.step_count = 0

        # TRIGGERED
        self.old_policy_params = None
        self.kl_threshold = config.get('kl_threshold', 0.1)

        # PERIODIC — [fusion Doc41] défaut 1000 au lieu de 100
        # 100 → 5000 réanalyses/500k steps (trop coûteux)
        # 1000 → 500 réanalyses/500k steps (raisonnable)
        self.period = config.get('period', 1000)

        # [S4] ratio adaptatif : 0.25 sur CPU, 0.8 sur GPU (comme MuZero)
        device = config.get('device', 'cpu')
        default_ratio = 0.25 if device == 'cpu' else 0.8
        self.reanalyze_ratio = config.get('reanalyze_ratio', default_ratio)

        self.k_steps    = config.get('k_steps', 5)
        self.k_adaptive = (self.k_steps == 'adaptive')

        self.n_reanalyzes = 0
        # [S3] tracker la dernière valeur k utilisée pour log_stats
        self._last_k_used = self.k_steps if not self.k_adaptive else 5

    def should_reanalyze(self, agent=None) -> bool:
        """
        CONTINUOUS : probabilité reanalyze_ratio
        PERIODIC   : toutes les N transitions (défaut 1000)
        TRIGGERED  : quand politique a suffisamment changé (KL > seuil)
        LAZY       : toujours False — géré dans train.py
        """
        # [S1] ne pas incrémenter step_count en mode LAZY
        # (la décision est prise ailleurs, pas ici)
        if self.mode != ReanalyzeMode.LAZY:
            self.step_count += 1

        if self.mode == ReanalyzeMode.CONTINUOUS:
            decision = np.random.random() < self.reanalyze_ratio
            if decision:
                self.n_reanalyzes += 1
            return decision

        elif self.mode == ReanalyzeMode.PERIODIC:
            decision = (self.step_count % self.period == 0)
            if decision:
                self.n_reanalyzes += 1
            return decision

        elif self.mode == ReanalyzeMode.TRIGGERED:
            if agent is None:
                return False
            kl = self._compute_kl_distance(agent)
            decision = kl > self.kl_threshold
            if decision:
                self.n_reanalyzes += 1
                # [S2] old_params mis à jour SEULEMENT quand on réanalyse
                # Si mis à jour à chaque step → distance toujours ~0
                # → trigger ne se déclenche jamais sur changement graduel
                self.old_policy_params = agent.get_policy_params()
            return decision

        elif self.mode == ReanalyzeMode.LAZY:
            return False

        return False

    def get_k_steps(self, traj_length: Optional[int] = None) -> int:
        """
        [S3] Retourne k et le stocke dans _last_k_used pour log_stats.
        traj_length passé depuis train.py pour mode adaptatif.
        """
        if not self.k_adaptive:
            self._last_k_used = int(self.k_steps)
            return self._last_k_used

        if traj_length is None:
            k = 5
        elif traj_length < 10:
            k = 1
        elif traj_length < 30:
            k = 3
        elif traj_length < 100:
            k = 5
        else:
            k = 10

        self._last_k_used = k
        return k

    def get_n_trajectories(self) -> int:
        return self.config.get('n_trajectories', 16)

    def _compute_kl_distance(self, agent) -> float:
        """
        Distance L2 normalisée comme proxy de KL divergence.
        [S2] old_policy_params initialisé ici au premier appel,
        mais mis à jour uniquement dans should_reanalyze() après décision.
        """
        if not hasattr(agent, 'get_policy_params'):
            return 0.0

        current_params = agent.get_policy_params()

        if self.old_policy_params is None:
            self.old_policy_params = current_params
            return 0.0

        total_distance = 0.0
        total_norm     = 0.0
        for cur, old in zip(current_params, self.old_policy_params):
            diff = cur - old
            total_distance += np.sum(diff ** 2)
            total_norm     += np.sum(old ** 2) + 1e-8

        return float(np.sqrt(total_distance / total_norm))

    def log_stats(self) -> dict:
        # [S5] _last_k_used = vraie valeur utilisée (pas 'adaptive')
        return {
            'scheduler/mode':            self.mode.value,
            'scheduler/step':            self.step_count,
            'scheduler/k_steps':         self._last_k_used,
            'scheduler/n_reanalyzes':    self.n_reanalyzes,
            'scheduler/ratio':           self.n_reanalyzes / max(1, self.step_count),
            'scheduler/reanalyze_ratio': self.reanalyze_ratio,
            'scheduler/period':          self.period,
        }