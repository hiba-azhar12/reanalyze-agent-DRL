"""
scheduler.py — Scheduler de réanalyse

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - should_reanalyze() : les 4 modes entièrement implémentés
      CONTINUOUS : retourne True avec probabilité reanalyze_ratio
      PERIODIC   : retourne True si step_count % period == 0
      TRIGGERED  : calcule la distance L2 entre ancienne et nouvelle politique,
                   déclenche si > kl_threshold, met à jour old_policy_params
      LAZY       : retourne toujours False (décision prise dans sample())
  - compute_kl_divergence() : distance L2 normalisée entre paramètres
  - get_k_steps() : k adaptatif basé sur longueur de trajectoire implémenté

CONTRIBUTION ORIGINALE :
  Les 4 modes de timing sont notre question de recherche principale.
  Pour un budget computationnel fixe, quelle combinaison mode × k_steps
  maximise la sample efficiency ?

Référence : ReZero (ICLR 2025), Adaptive Offline Data Replay (ICLR 2024)
"""

import numpy as np
import torch
from enum import Enum
from typing import Optional, Tuple


class ReanalyzeMode(Enum):
    CONTINUOUS = "continuous"
    PERIODIC   = "periodic"
    TRIGGERED  = "triggered"
    LAZY       = "lazy"


class ReanalyzeScheduler:
    """
    Scheduler qui décide quand et combien réanalyser.

    Question de recherche (contribution) :
        Pour un budget computationnel fixe, quelle combinaison
        mode × k_steps maximise la sample efficiency ?
    """

    def __init__(self, mode: str, config: dict):
        self.mode = ReanalyzeMode(mode)
        self.config = config
        self.step_count = 0

        # Mode TRIGGERED : stocker l'ancienne politique pour comparaison
        self.old_policy_params = None
        self.kl_threshold = config.get('kl_threshold', 0.1)

        # Mode PERIODIC
        self.period = config.get('period', 100)

        # Ratio réanalyse/collecte (mode CONTINUOUS)
        # MuZero utilise 80/20 — on teste plusieurs valeurs
        self.reanalyze_ratio = config.get('reanalyze_ratio', 0.8)

        # Profondeur de réanalyse
        self.k_steps = config.get('k_steps', 5)
        self.k_adaptive = (self.k_steps == 'adaptive')

        # Compteur de réanalyses (pour les stats)
        self.n_reanalyzes = 0

    def should_reanalyze(self, agent=None) -> bool:
        """
        Retourne True si on doit réanalyser à ce step.

        CONTINUOUS  : aléatoire avec probabilité reanalyze_ratio
                      Simule le ratio 80/20 de MuZero
        PERIODIC    : toutes les N transitions
                      Plus efficace car parallélisable (ReZero)
        TRIGGERED   : quand la politique a beaucoup changé
                      Adaptatif — réanalyse plus quand l'agent progresse vite
                      (Adaptive Offline Data Replay)
        LAZY        : jamais ici — décision dans sample() du buffer
        """
        self.step_count += 1

        if self.mode == ReanalyzeMode.CONTINUOUS:
            # Réanalyser avec probabilité reanalyze_ratio
            decision = np.random.random() < self.reanalyze_ratio
            if decision:
                self.n_reanalyzes += 1
            return decision

        elif self.mode == ReanalyzeMode.PERIODIC:
            # Réanalyser toutes les `period` transitions
            decision = (self.step_count % self.period == 0)
            if decision:
                self.n_reanalyzes += 1
            return decision

        elif self.mode == ReanalyzeMode.TRIGGERED:
            # Réanalyser quand la politique a suffisamment changé
            if agent is None:
                return False
            kl = self.compute_kl_divergence(agent)
            decision = kl > self.kl_threshold
            if decision:
                self.n_reanalyzes += 1
            return decision

        elif self.mode == ReanalyzeMode.LAZY:
            # LAZY ne décide pas ici — la réanalyse est faite dans reanalyze()
            # juste avant d'utiliser chaque trajectoire
            return False

        return False

    def get_k_steps(self, traj_length: Optional[int] = None) -> int:
        """
        Retourne la profondeur de réanalyse pour cette trajectoire.

        k fixe : retourne self.k_steps (1, 3, 5, 10, ou T)
        k adaptatif : ajuste selon la longueur de la trajectoire

        LOGIQUE ADAPTATIVE :
        - Trajectoire courte (< 10 steps) : k=1 ou k=3 pour éviter variance
        - Trajectoire moyenne (10-50 steps) : k=5 est l'optimum typique
        - Trajectoire longue (> 50 steps) : k=10 pour exploiter plus de signal
        """
        if not self.k_adaptive:
            return self.k_steps

        if traj_length is None:
            return 5  # valeur par défaut

        if traj_length < 10:
            return 1
        elif traj_length < 30:
            return 3
        elif traj_length < 100:
            return 5
        else:
            return 10

    def get_n_trajectories(self) -> int:
        """Retourne le nombre de trajectoires à réanalyser."""
        return self.config.get('n_trajectories', 16)

    def compute_kl_divergence(self, agent) -> float:
        """
        Estime la distance entre ancienne et nouvelle politique.

        On utilise la distance L2 normalisée entre les paramètres
        comme proxy de la vraie KL divergence.

        Pourquoi L2 et pas la vraie KL :
        La vraie KL nécessite un batch d'états pour évaluer π_old(a|s) et π_new(a|s).
        La distance L2 sur les paramètres est une approximation plus rapide,
        suffisante pour décider si la politique a "assez changé".

        NOUVEAU par rapport au code original — logique complète implémentée.
        """
        if not hasattr(agent, 'get_policy_params'):
            return 0.0

        current_params = agent.get_policy_params()

        if self.old_policy_params is None:
            # Première fois : stocker et retourner 0
            self.old_policy_params = current_params
            return 0.0

        # Distance L2 normalisée entre paramètres
        total_distance = 0.0
        total_norm = 0.0
        for cur, old in zip(current_params, self.old_policy_params):
            diff = cur - old
            total_distance += np.sum(diff ** 2)
            total_norm += np.sum(old ** 2) + 1e-8

        kl_proxy = np.sqrt(total_distance / total_norm)

        # Mettre à jour l'ancienne politique
        self.old_policy_params = current_params

        return float(kl_proxy)

    def log_stats(self) -> dict:
        """Retourne des stats pour le logging."""
        return {
            'scheduler/mode':        self.mode.value,
            'scheduler/step':        self.step_count,
            'scheduler/k_steps':     self.k_steps if not self.k_adaptive else 'adaptive',
            'scheduler/n_reanalyzes': self.n_reanalyzes,
            'scheduler/ratio':       self.n_reanalyzes / max(1, self.step_count),
        }
