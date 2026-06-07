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

        
        self.old_policy_params = None
        self.kl_threshold = config.get('kl_threshold', 0.1)

       self.period = config.get('period', 1000)

        device = config.get('device', 'cpu')
        default_ratio = 0.25 if device == 'cpu' else 0.8
        self.reanalyze_ratio = config.get('reanalyze_ratio', default_ratio)

        self.k_steps    = config.get('k_steps', 5)
        self.k_adaptive = (self.k_steps == 'adaptive')

        self.n_reanalyzes = 0
        self._last_k_used = self.k_steps if not self.k_adaptive else 5

    def should_reanalyze(self, agent=None) -> bool:

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
                self.old_policy_params = agent.get_policy_params()
            return decision

        elif self.mode == ReanalyzeMode.LAZY:
            return False

        return False

    def get_k_steps(self, traj_length: Optional[int] = None) -> int:
        
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
        return {
            'scheduler/mode':            self.mode.value,
            'scheduler/step':            self.step_count,
            'scheduler/k_steps':         self._last_k_used,
            'scheduler/n_reanalyzes':    self.n_reanalyzes,
            'scheduler/ratio':           self.n_reanalyzes / max(1, self.step_count),
            'scheduler/reanalyze_ratio': self.reanalyze_ratio,
            'scheduler/period':          self.period,
        }