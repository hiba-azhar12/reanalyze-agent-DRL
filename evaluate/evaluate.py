"""
evaluate.py — Évaluation multi-seeds avec intervalles de confiance

Usage :
    python evaluate/evaluate.py --results_dir results/ --configs dqn_per reanalyze_base --seeds 0 1 2

CORRECTIONS APPLIQUÉES :
  [E1] run_seeds() : lit les fichiers .npy/.json existants au lieu de re-entraîner
  [E2] wilcoxon    : warning explicite si seeds manquants
  [E3] rewards_array : inclus dans le JSON de sortie
  [E4] chemin output : absolu relatif au script
  [E5] staleness + td_error : lus depuis metrics.json et comparés
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def find_results_dir() -> str:
    """Trouve le dossier results/ relatif au script — fonctionne sur Kaggle et local."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'results')


def load_seed_results(results_dir: str, config_name: str, seed: int) -> Optional[Dict]:
    """
    [E1] Lit les fichiers existants pour un config+seed donné.
    Retourne None si les fichiers n'existent pas.
    """
    base = os.path.join(results_dir, f"{config_name}_seed{seed}")
    rewards_path = f"{base}_rewards.npy"
    metrics_path = f"{base}_metrics.json"

    if not os.path.exists(rewards_path):
        print(f"  ⚠️  Fichier manquant : {rewards_path}")
        return None
    if not os.path.exists(metrics_path):
        print(f"  ⚠️  Fichier manquant : {metrics_path}")
        return None

    rewards = np.load(rewards_path, allow_pickle=True).tolist()
    with open(metrics_path) as f:
        metrics = json.load(f)

    return {
        'rewards':         rewards,
        'staleness_log':   metrics.get('staleness_log', []),
        'td_error_log':    metrics.get('td_error_log', []),
        'reanalyze_count': metrics.get('reanalyze_count', 0),
        'mean_step_time':  metrics.get('mean_step_time', 0.0),
        'config':          metrics.get('config', {}),
        'seed':            seed,
    }


def run_seeds(results_dir: str, config_name: str, seeds: List[int]) -> Dict:
    """
    [E1] Agrège les résultats existants pour tous les seeds d'une config.
    Ne relance PAS l'entraînement.
    """
    print(f"\n📊 Config : {config_name} | Seeds : {seeds}")

    all_rewards      = []
    all_staleness    = []
    all_td_errors    = []
    reanalyze_counts = []
    step_times       = []
    loaded_seeds     = []

    for seed in seeds:
        result = load_seed_results(results_dir, config_name, seed)
        if result is None:
            print(f"  ⚠️  Seed {seed} ignoré (fichiers manquants)")
            continue
        all_rewards.append(result['rewards'])
        all_staleness.append(result['staleness_log'])
        all_td_errors.append(result['td_error_log'])
        reanalyze_counts.append(result['reanalyze_count'])
        step_times.append(result['mean_step_time'])
        loaded_seeds.append(seed)
        print(f"  ✅ Seed {seed} chargé — {len(result['rewards'])} épisodes")

    if not all_rewards:
        print(f"  ❌ Aucun résultat trouvé pour {config_name}")
        return None

    # Aligner sur la longueur minimale
    min_len = min(len(r) for r in all_rewards)
    rewards_array = np.array([r[:min_len] for r in all_rewards])  # (n_seeds, episodes)

    mean  = rewards_array.mean(axis=0)
    std   = rewards_array.std(axis=0)
    n     = len(loaded_seeds)
    ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else std

    # Score final = moyenne des 100 derniers épisodes par seed
    final_scores = [
        float(np.mean(r[-100:])) if len(r) >= 100 else float(np.mean(r))
        for r in all_rewards
    ]

    # Staleness moyen par seed (si disponible)
    mean_staleness = [
        float(np.mean(s)) if s else 0.0
        for s in all_staleness
    ]

    # TD error moyen par seed (si disponible)
    mean_td = [
        float(np.mean(t)) if t else 0.0
        for t in all_td_errors
    ]

    print(f"  Score final : {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
    if any(s > 0 for s in mean_staleness):
        print(f"  Staleness moyen : {np.mean(mean_staleness):.1f}")
    if any(t > 0 for t in mean_td):
        print(f"  TD error moyen  : {np.mean(mean_td):.4f}")

    return {
        'config_name':     config_name,
        'loaded_seeds':    loaded_seeds,
        'n_seeds':         n,
        'all_rewards':     all_rewards,
        'rewards_array':   rewards_array.tolist(),  # [E3] inclus
        'mean':            mean.tolist(),
        'ci_95':           ci_95.tolist(),
        'final_scores':    final_scores,
        'mean_final':      float(np.mean(final_scores)),
        'std_final':       float(np.std(final_scores)),
        'mean_staleness':  mean_staleness,
        'mean_td_errors':  mean_td,
        'reanalyze_counts': reanalyze_counts,
        'mean_step_time':  float(np.mean(step_times)) if step_times else 0.0,
    }


def wilcoxon_test(name_a: str, scores_a: list, name_b: str, scores_b: list) -> Dict:
    """
    [E2] Test de Wilcoxon avec warning si seeds manquants ou tailles différentes.
    """
    if len(scores_a) != len(scores_b):
        msg = (f"  ⚠️  Wilcoxon {name_a} vs {name_b} ignoré : "
               f"{len(scores_a)} seeds vs {len(scores_b)} seeds — "
               f"tailles différentes")
        print(msg)
        return {
            'skipped': True,
            'reason': f"Tailles différentes : {len(scores_a)} vs {len(scores_b)}"
        }

    if len(scores_a) < 2:
        print(f"  ⚠️  Wilcoxon {name_a} vs {name_b} ignoré : besoin d'au moins 2 seeds")
        return {'skipped': True, 'reason': "Moins de 2 seeds"}

    try:
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
        return {
            'skipped':        False,
            'statistic':      float(stat),
            'p_value':        float(p_value),
            'significant':    p_value < 0.05,
            'interpretation': (
                f"✅ Différence significative (p={p_value:.4f} < 0.05)"
                if p_value < 0.05
                else f"❌ Pas de différence significative (p={p_value:.4f} >= 0.05)"
            )
        }
    except Exception as e:
        return {'skipped': True, 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Évalue les résultats d'entraînement existants"
    )
    parser.add_argument(
        '--results_dir', default=None,
        help='Dossier results/ (défaut: auto-détecté relatif au script)'
    )
    parser.add_argument(
        '--configs', nargs='+', required=True,
        help='Noms des configs à comparer (ex: dqn_per reanalyze_base)'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[0, 1, 2],
        help='Seeds à charger (défaut: 0 1 2)'
    )
    parser.add_argument(
        '--output', default=None,
        help='Chemin du JSON de sortie (défaut: results/evaluation.json)'
    )
    args = parser.parse_args()

    # [E4] chemins absolus
    results_dir = args.results_dir or find_results_dir()
    output_path = args.output or os.path.join(results_dir, 'evaluation.json')

    print(f"📁 Dossier results : {results_dir}")
    print(f"📋 Configs         : {args.configs}")
    print(f"🌱 Seeds           : {args.seeds}")

    all_results = {}
    for config_name in args.configs:
        result = run_seeds(results_dir, config_name, args.seeds)
        if result is not None:
            all_results[config_name] = result

    if not all_results:
        print("\n❌ Aucun résultat chargé — vérifier le dossier results/")
        return

    # Tests de Wilcoxon entre toutes les paires
    print(f"\n{'='*60}")
    print("Tests de Wilcoxon (comparaisons par paires)")
    print('='*60)

    config_names   = list(all_results.keys())
    wilcoxon_results = {}

    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            a, b = config_names[i], config_names[j]
            test = wilcoxon_test(
                a, all_results[a]['final_scores'],
                b, all_results[b]['final_scores']
            )
            key = f"{a}_vs_{b}"
            wilcoxon_results[key] = test
            if not test.get('skipped'):
                print(f"  {a} vs {b} : {test['interpretation']}")

    # Résumé final
    print(f"\n{'='*60}")
    print("Résumé des scores finaux")
    print('='*60)
    for name, res in all_results.items():
        print(f"  {name:30s} : {res['mean_final']:8.1f} ± {res['std_final']:.1f}"
              f"  (seeds: {res['loaded_seeds']})")

    # [E3] Sauvegarder avec rewards_array inclus
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output = {
        'results':  all_results,
        'wilcoxon': wilcoxon_results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Résultats sauvegardés : {output_path}")


if __name__ == '__main__':
    main()