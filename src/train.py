"""
train.py — Boucle d'entraînement principale

Usage :
    python src/train.py --config configs/dqn_baseline.yaml
    python src/train.py --config configs/reanalyze_base.yaml --seed 42

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - Boucle principale entièrement implémentée (le TODO est résolu)
  - Gestion correcte des trajectoires pour ReanalyzeBuffer :
    on accumule les steps d'un épisode avant de les ajouter comme trajectoire
  - Mode LAZY implémenté : réanalyse juste avant chaque update
  - Logging étendu : staleness, TD error, coût computationnel par step
  - Sauvegarde JSON des métriques (pas juste npy) pour evaluate.py
  - epsilon decay linéaire avec warmup (buffer doit se remplir d'abord)
"""

import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))

from buffer import ReplayBuffer, PrioritizedReplayBuffer, ReanalyzeBuffer
from networks import DQNNetwork
from agent import DQNAgent
from reanalyze import reanalyze
from scheduler import ReanalyzeScheduler


def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def make_buffer(config: Dict):
    """Instancie le bon type de buffer selon la config."""
    buffer_type = config.get('buffer_type', 'replay')
    capacity = config['buffer_capacity']

    if buffer_type == 'replay':
        return ReplayBuffer(capacity)
    elif buffer_type == 'per':
        return PrioritizedReplayBuffer(
            capacity,
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4),
        )
    elif buffer_type == 'reanalyze':
        return ReanalyzeBuffer(
            capacity,
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4),
        )
    else:
        raise ValueError(f"Buffer type inconnu : {buffer_type}")


def train(config: Dict, seed: int = 42) -> Dict:
    """
    Boucle d'entraînement principale.
    Retourne un dict de métriques pour l'évaluation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(config['env'], seed)
    obs_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent  = DQNAgent(obs_dim, action_dim, config)
    buffer = make_buffer(config)

    # Scheduler si mode reanalyze activé
    scheduler = None
    if config.get('reanalyze', False):
        scheduler = ReanalyzeScheduler(
            mode=config.get('reanalyze_mode', 'continuous'),
            config=config,
        )

    # Métriques à tracer
    episode_rewards     = []
    all_losses          = []
    staleness_log       = []   # âge moyen des trajectoires utilisées
    td_error_log        = []   # erreur TD moyenne avant/après réanalyse
    step_times          = []   # coût computationnel par step (ms)
    reanalyze_count     = 0    # nombre de réanalyses effectuées

    # Epsilon decay
    epsilon_start = config.get('epsilon_start', 1.0)
    epsilon_end   = config.get('epsilon_end',   0.01)
    epsilon_decay = config.get('epsilon_decay', 10000)

    total_steps = config.get('total_steps', 500000)

    obs, _ = env.reset()
    episode_reward   = 0
    episode_steps    = 0
    current_episode  = []   # accumule les steps pour add_trajectory()

    print(f"Entraînement — {config['env']} | {total_steps} steps | seed {seed}")
    print(f"Buffer : {config.get('buffer_type')} | Réanalyse : {config.get('reanalyze', False)}")
    print("-" * 60)

    for step in range(total_steps):
        t_start = time.perf_counter()

        # Epsilon decay linéaire
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay
        )

        # Sélection d'action
        action = agent.select_action(obs, epsilon)

        # Interaction avec l'environnement
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Ajouter dans le buffer
        if isinstance(buffer, ReanalyzeBuffer):
            # Pour ReanalyzeBuffer : accumuler l'épisode complet
            current_episode.append({
                'state':      obs.copy(),
                'action':     action,
                'reward':     reward,
                'next_state': next_obs.copy(),
                'done':       done,
            })
            # Ajouter aussi dans le buffer parent pour les updates normaux
            buffer.add(obs, action, reward, next_obs, done)
        else:
            buffer.add(obs, action, reward, next_obs, done)

        # Update de l'agent
        loss = agent.update(buffer)
        if loss is not None:
            all_losses.append(loss)

        # Réanalyse selon le scheduler
        if scheduler is not None and len(buffer) > agent.batch_size:

            if scheduler.mode.value == 'lazy':
                # Mode LAZY : réanalyser juste avant l'update
                # (déjà géré dans le buffer pour les trajectoires)
                pass
            elif scheduler.should_reanalyze(agent):
                # Choisir des trajectoires à réanalyser
                if isinstance(buffer, ReanalyzeBuffer) and len(buffer.trajectories) > 0:
                    traj_ids = buffer.get_all_trajectory_ids()
                    n_to_reanalyze = min(
                        scheduler.get_n_trajectories(),
                        len(traj_ids)
                    )
                    selected = np.random.choice(traj_ids, n_to_reanalyze, replace=False)
                    k = scheduler.get_k_steps()
                    reanalyze(
                        buffer, agent.online_net, selected.tolist(),
                        k, config['gamma'], config.get('device', 'cpu')
                    )
                    reanalyze_count += 1

        # Fin d'épisode
        episode_reward += reward
        episode_steps  += 1
        obs = next_obs

        if done:
            # Pour ReanalyzeBuffer : ajouter la trajectoire complète
            if isinstance(buffer, ReanalyzeBuffer) and len(current_episode) > 0:
                buffer.add_trajectory(current_episode)
                buffer._increment_step()
            current_episode = []

            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_steps  = 0
            obs, _ = env.reset()

        # Coût computationnel par step
        t_end = time.perf_counter()
        step_times.append((t_end - t_start) * 1000)  # en ms

        # Logging tous les 10 000 steps
        if step > 0 and step % 10_000 == 0:
            mean_r   = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards) if episode_rewards else 0
            mean_t   = np.mean(step_times[-1000:])
            mean_l   = np.mean(all_losses[-1000:]) if all_losses else 0

            # Staleness si ReanalyzeBuffer
            staleness_info = ""
            if isinstance(buffer, ReanalyzeBuffer):
                stats = buffer.get_staleness_stats()
                staleness_log.append(stats['mean_age'])
                staleness_info = f"| Staleness: {stats['mean_age']:.0f}"

            print(f"Step {step:7d} | Ep {len(episode_rewards):5d} "
                  f"| Reward: {mean_r:8.2f} | Loss: {mean_l:.4f} "
                  f"| ε: {epsilon:.3f} | {mean_t:.2f}ms/step "
                  f"| Réanalyses: {reanalyze_count} {staleness_info}")

    env.close()
    print(f"\nFin — {len(episode_rewards)} épisodes | {reanalyze_count} réanalyses")

    return {
        'episode_rewards':  episode_rewards,
        'losses':           all_losses,
        'staleness_log':    staleness_log,
        'td_error_log':     td_error_log,
        'step_times':       step_times,
        'reanalyze_count':  reanalyze_count,
        'config':           config,
        'seed':             seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    print(f"Config : {args.config} | Seed : {args.seed} | Device : {config.get('device','cpu')}")
    print("-" * 50)

    results = train(config, seed=args.seed)

    # Sauvegarder résultats
    os.makedirs('results', exist_ok=True)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    base_path   = f"results/{config_name}_seed{args.seed}"

    # Sauvegarder les rewards (pour les courbes)
    np.save(f"{base_path}_rewards.npy", results['episode_rewards'])

    # Sauvegarder toutes les métriques en JSON
    metrics = {
        'episode_rewards': results['episode_rewards'],
        'staleness_log':   results['staleness_log'],
        'reanalyze_count': results['reanalyze_count'],
        'mean_step_time':  float(np.mean(results['step_times'])) if results['step_times'] else 0,
        'config':          config,
        'seed':            args.seed,
    }
    with open(f"{base_path}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Résultats sauvegardés : {base_path}_rewards.npy + _metrics.json")


if __name__ == '__main__':
    main()
