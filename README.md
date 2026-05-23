# Reanalyze Agent

Implémentation d'un Reanalyze Agent basé sur MuZero Reanalyze (Schrittwieser et al., 2021),
avec comparaisons état de l'art et contribution originale sur la stratégie de réanalyse.

## Structure

```
reanalyze-agent/
├── src/
│   ├── buffer.py        # SumTree, ReplayBuffer, PrioritizedReplayBuffer, ReanalyzeBuffer
│   ├── networks.py      # MLP, DQN network, loss de consistance latente
│   ├── agent.py         # DQNAgent avec select_action(), update(), sync_target()
│   ├── reanalyze.py     # Fonction de réanalyse + variantes (EfficientZero, DreamerV3, TD-MPC2)
│   ├── scheduler.py     # 4 modes de timing : continu, périodique, déclenché, lazy
│   └── train.py         # Boucle d'entraînement complète
├── evaluate/
│   ├── evaluate.py      # Évaluation multi-seeds avec intervalles de confiance
│   └── visualize.py     # 7 courbes + 4 visualisations buffer
├── configs/
│   ├── dqn_baseline.yaml
│   ├── dqn_per.yaml
│   ├── reanalyze_base.yaml
│   ├── reanalyze_efficientzero.yaml
│   ├── reanalyze_dreamer.yaml
│   └── reanalyze_tdmpc2.yaml
├── notebooks/           # Tests rapides par étape
└── results/             # Courbes, métriques, checkpoints
```

## Agents comparés

| Agent | Description |
|-------|-------------|
| DQN vanilla | Baseline — buffer FIFO, pas de réanalyse |
| DQN + PER | Prioritized Experience Replay |
| Reanalyze base | MuZero Reanalyze simplifié (80/20) |
| + EfficientZero | + loss de consistance latente |
| + DreamerV3 | + trajectoires fictives mixées |
| + TD-MPC2 | + planification par échantillonnage |
| Contribution | Timing × Profondeur variable |

## Contribution originale

Question de recherche : **pour un budget computationnel fixe, quelle combinaison
timing × profondeur de réanalyse maximise la sample efficiency ?**

4 modes de timing : continu / périodique / déclenché (KL divergence) / lazy  
6 profondeurs : k=1, k=3, k=5, k=10, k=T, k adaptatif

## Environnements

- CartPole-v1 — débogage uniquement
- LunarLander-v2 — résultats principaux
- Atari 100k — comparaison avec la littérature

## Installation

```bash
pip install -r requirements.txt
```

## Lancer un entraînement

```bash
python src/train.py --config configs/dqn_baseline.yaml
python src/train.py --config configs/reanalyze_base.yaml
```

## Évaluation complète (5 seeds)

```bash
python evaluate/evaluate.py --config configs/reanalyze_base.yaml --seeds 5
```

## Références

- Mnih et al. (2015) — DQN
- Schaul et al. (2016) — Prioritized Experience Replay
- Schrittwieser et al. (2020) — MuZero
- Schrittwieser et al. (2021) — MuZero Reanalyze
- Fedus et al. (2020) — Revisiting Fundamentals of Experience Replay
- Hafner et al. (2023) — DreamerV3
- Hansen et al. (2024) — TD-MPC2
- Wang et al. (2024) — EfficientZero V2
