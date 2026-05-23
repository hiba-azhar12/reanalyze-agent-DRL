"""
evaluate.py — Évaluation multi-seeds avec intervalles de confiance

Usage :
    python evaluate/evaluate.py --configs configs/dqn_baseline.yaml configs/reanalyze_base.yaml --seeds 5

NOUVEAU — entièrement implémenté :
  - Lance train() sur N seeds pour chaque config
  - Calcule moyenne + IC à 95% sur les episode_rewards
  - Test de Wilcoxon entre deux agents pour valider "agent A > agent B"
  - Sauvegarde les résultats pour visualize.py
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train, load_config


def run_seeds(config_path: str, n_seeds: int = 5) -> Dict:
    """Lance l'entraînement sur n_seeds seeds et agrège les résultats."""
    config = load_config(config_path)
    all_rewards = []
    all_metrics = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed}/{n_seeds-1} ---")
        results = train(config, seed=seed)
        all_rewards.append(results['episode_rewards'])
        all_metrics.append({
            'staleness': results['staleness_log'],
            'reanalyze_count': results['reanalyze_count'],
            'mean_step_time': float(np.mean(results['step_times'])) if results['step_times'] else 0,
        })

    # Trouver la longueur minimale commune
    min_len = min(len(r) for r in all_rewards)
    rewards_array = np.array([r[:min_len] for r in all_rewards])  # (n_seeds, episodes)

    # Moyenne et IC à 95%
    mean   = rewards_array.mean(axis=0)
    std    = rewards_array.std(axis=0)
    ci_95  = 1.96 * std / np.sqrt(n_seeds)

    # Score final = moyenne des 100 derniers épisodes par seed
    final_scores = [np.mean(r[-100:]) if len(r) >= 100 else np.mean(r) for r in all_rewards]

    return {
        'config_path':   config_path,
        'config':        config,
        'all_rewards':   all_rewards,
        'rewards_array': rewards_array.tolist(),
        'mean':          mean.tolist(),
        'ci_95':         ci_95.tolist(),
        'final_scores':  final_scores,
        'mean_final':    float(np.mean(final_scores)),
        'std_final':     float(np.std(final_scores)),
        'metrics':       all_metrics,
    }


def wilcoxon_test(scores_a: list, scores_b: list) -> Dict:
    """
    Test de Wilcoxon pour comparer deux agents.
    Retourne p-value et interprétation.

    On ne peut affirmer "agent A > agent B" que si p < 0.05.
    """
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    return {
        'statistic': float(stat),
        'p_value':   float(p_value),
        'significant': p_value < 0.05,
        'interpretation': (
            f"Différence significative (p={p_value:.4f} < 0.05)"
            if p_value < 0.05
            else f"Pas de différence significative (p={p_value:.4f} >= 0.05)"
        )
    }


def main():
    from typing import Dict  # import local pour éviter problème de scope

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True,
                        help='Chemins vers les configs à comparer')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--output', default='results/evaluation.json')
    args = parser.parse_args()

    all_results = {}

    for config_path in args.configs:
        print(f"\n{'='*60}")
        print(f"Évaluation : {config_path}")
        print('='*60)
        results = run_seeds(config_path, args.seeds)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        all_results[config_name] = results
        print(f"Score final : {results['mean_final']:.1f} ± {results['std_final']:.1f}")

    # Tests de Wilcoxon entre toutes les paires
    config_names = list(all_results.keys())
    print(f"\n{'='*60}")
    print("Tests de Wilcoxon (comparaisons par paires)")
    print('='*60)
    wilcoxon_results = {}
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            a_name = config_names[i]
            b_name = config_names[j]
            if len(all_results[a_name]['final_scores']) == len(all_results[b_name]['final_scores']):
                test = wilcoxon_test(
                    all_results[a_name]['final_scores'],
                    all_results[b_name]['final_scores']
                )
                key = f"{a_name}_vs_{b_name}"
                wilcoxon_results[key] = test
                print(f"{a_name} vs {b_name} : {test['interpretation']}")

    # Sauvegarder
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'rewards_array'}
                    for k, v in all_results.items()},
        'wilcoxon': wilcoxon_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats sauvegardés : {args.output}")


if __name__ == '__main__':
    main()
