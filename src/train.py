"""
train.py — Boucle d'entraînement principale

Usage :
    python src/train.py --config configs/dqn_baseline.yaml
    python src/train.py --config configs/reanalyze_base.yaml --seed 42

CORRECTIONS APPLIQUÉES :
  [T1,T2] make_buffer() : beta_increment passé depuis config pour PER et ReanalyzeBuffer
  [T3]    epsilon_decay  : défaut 10000 → 100000 (exploration suffisante)
  [T4]    warmup         : réanalyse bloquée pendant reanalyze_warmup_steps
  [T5]    chemin results : chemin relatif au script (plus de path Kaggle codé en dur)
  [T6]    logging        : affiche 'warmup' si losses encore vides
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
        # [T1] beta_increment passé depuis config
        return PrioritizedReplayBuffer(
            capacity,
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4),
            beta_increment=config.get('per_beta_increment', 0.0001),
        )

    elif buffer_type == 'reanalyze':
        # [T2] idem pour ReanalyzeBuffer
        return ReanalyzeBuffer(
            capacity,
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4),
            beta_increment=config.get('per_beta_increment', 0.0001),
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
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent  = DQNAgent(obs_dim, action_dim, config)
    buffer = make_buffer(config)

    scheduler = None
    if config.get('reanalyze', False):
        scheduler = ReanalyzeScheduler(
            mode=config.get('reanalyze_mode', 'continuous'),
            config=config,
        )

    # Métriques
    episode_rewards = []
    all_losses      = []
    staleness_log   = []
    td_error_log    = []
    step_times      = []
    reanalyze_count = 0

    # Epsilon decay
    epsilon_start = config.get('epsilon_start', 1.0)
    epsilon_end   = config.get('epsilon_end',   0.01)
    # [T3] défaut 100000 au lieu de 10000
    epsilon_decay = config.get('epsilon_decay', 100000)

    # [T4] warmup : pas de réanalyse avant ce nombre de steps
    reanalyze_warmup = config.get('reanalyze_warmup_steps', 50000)

    total_steps = config.get('total_steps', 500000)

    obs, _ = env.reset()
    episode_reward  = 0
    episode_steps   = 0
    current_episode = []

    print(f"Entraînement — {config['env']} | {total_steps} steps | seed {seed}")
    print(f"Buffer : {config.get('buffer_type')} | Réanalyse : {config.get('reanalyze', False)}")
    print(f"epsilon_decay : {epsilon_decay} | warmup réanalyse : {reanalyze_warmup}")
    print("-" * 60)

    for step in range(total_steps):
        t_start = time.perf_counter()

        # Epsilon decay linéaire
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay
        )

        action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if isinstance(buffer, ReanalyzeBuffer):
            current_episode.append({
                'state':      obs.copy(),
                'action':     action,
                'reward':     reward,
                'next_state': next_obs.copy(),
                'done':       done,
            })
        else:
            buffer.add(obs, action, reward, next_obs, done)

        loss = agent.update(buffer)
        if loss is not None:
            all_losses.append(loss)

        # [T4] Réanalyse avec warmup — réseau doit avoir appris avant de réanalyser
        reanalyze_ready = (
            scheduler is not None
            and len(buffer) > agent.batch_size
            and step > reanalyze_warmup          # ← warmup respecté
        )

        if reanalyze_ready:
            if scheduler.mode.value == 'lazy':
                pass
            elif scheduler.should_reanalyze(agent):
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

        episode_reward += reward
        episode_steps  += 1
        obs = next_obs

        if done:
            if isinstance(buffer, ReanalyzeBuffer) and len(current_episode) > 0:
                buffer.add_trajectory(current_episode)
                buffer._increment_step()
            current_episode = []

            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_steps  = 0
            obs, _ = env.reset()

        t_end = time.perf_counter()
        step_times.append((t_end - t_start) * 1000)

        if step > 0 and step % 10_000 == 0:
            mean_r = (
                np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50
                else np.mean(episode_rewards) if episode_rewards
                else 0
            )
            mean_t = np.mean(step_times[-1000:])

            # [T6] affiche 'warmup' si losses encore vides
            if all_losses:
                mean_l = np.mean(all_losses[-1000:])
                loss_str = f"{mean_l:.4f}"
            else:
                loss_str = "warmup"

            staleness_info = ""
            if isinstance(buffer, ReanalyzeBuffer):
                stats = buffer.get_staleness_stats()
                staleness_log.append(stats['mean_age'])
                staleness_info = f"| Staleness: {stats['mean_age']:.0f}"

            # Affiche si on est encore en warmup réanalyse
            warmup_info = ""
            if scheduler is not None and step <= reanalyze_warmup:
                warmup_info = f"| Warmup réanalyse ({step}/{reanalyze_warmup})"

            print(f"Step {step:7d} | Ep {len(episode_rewards):5d} "
                  f"| Reward: {mean_r:8.2f} | Loss: {loss_str} "
                  f"| ε: {epsilon:.3f} | {mean_t:.2f}ms/step "
                  f"| Réanalyses: {reanalyze_count} {staleness_info}{warmup_info}")

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

    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # [T5] chemin relatif au script — fonctionne partout (Kaggle, local, Colab)
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    base_path = f"{results_dir}/{config_name}_seed{args.seed}"

    np.save(f"{base_path}_rewards.npy", results['episode_rewards'])

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