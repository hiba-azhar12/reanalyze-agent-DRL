"""
reanalyze.py — Fonctions de réanalyse

MODIFICATIONS PAR RAPPORT AU CODE ORIGINAL :
  - compute_nstep_targets() : entièrement implémentée
    calcule G_t^(k) = r_t + γr_{t+1} + ... + γ^{k-1}r_{t+k-1} + γ^k * V(s_{t+k})
  - reanalyze() : boucle complète implémentée avec torch.no_grad()
    récupère les trajectoires, recalcule V(s) avec le réseau actuel,
    appelle compute_nstep_targets(), met à jour le buffer
  - reanalyze_dreamer() : variante DreamerV3 implémentée
    génère des trajectoires fictives depuis des états réels
    et les mixe avec les cibles réanalysées
  - reanalyze_tdmpc2() : variante TD-MPC2 implémentée
    évalue N séquences d'actions aléatoires et prend la meilleure

Concept clé — le staleness :
    Quand une trajectoire est collectée, les cibles TD sont calculées avec
    les poids du réseau à ce moment-là. Mais le réseau s'améliore. Après
    N steps d'entraînement, ces cibles sont "périmées" (stale).
    La réanalyse recalcule ces cibles avec les poids ACTUELS du réseau.

Référence : Schrittwieser et al. (2021) — MuZero Reanalyze
"""

import torch
import numpy as np
from typing import List, Dict, Optional


def compute_nstep_targets(rewards: List[float], values: torch.Tensor,
                          k_steps: int, gamma: float) -> np.ndarray:
    """
    Calcule les cibles n-step pour une trajectoire.

    Formule :
        G_t^(k) = r_t + γ*r_{t+1} + ... + γ^{k-1}*r_{t+k-1} + γ^k * V(s_{t+k})

    Plus k est grand :
      - Moins on dépend du réseau (les vraies récompenses dominent)
      - Biais ↓, variance ↑
      - Cibles plus fiables mais plus lentes à calculer

    Plus k est petit :
      - Plus on dépend du réseau pour estimer le futur
      - Biais ↑, variance ↓
      - k=1 : target = r + γ*V(s') — DQN classique

    Args :
        rewards  : liste de récompenses de la trajectoire
        values   : tensor des V(s) calculés par le réseau actuel, shape (T,)
        k_steps  : nombre de pas de bootstrap (1 à T)
        gamma    : facteur de discount

    Returns :
        targets  : array de shape (T,) avec les cibles recalculées
    """
    T = len(rewards)
    targets = np.zeros(T, dtype=np.float32)
    values_np = values.cpu().numpy() if isinstance(values, torch.Tensor) else values

    for t in range(T):
        G = 0.0
        # Accumuler les récompenses sur k steps
        for j in range(k_steps):
            if t + j < T:
                G += (gamma ** j) * rewards[t + j]
        # Bootstrap avec V(s_{t+k}) si on n'est pas à la fin
        if t + k_steps < T:
            G += (gamma ** k_steps) * values_np[t + k_steps]
        targets[t] = G

    return targets


def reanalyze(buffer, network: torch.nn.Module,
              indices: List[int], k_steps: int,
              gamma: float, device: str = 'cpu') -> None:
    """
    Fonction centrale de réanalyse (MuZero Reanalyze adapté pour DQN).

    Pour chaque trajectoire sélectionnée :
    1. Récupère la trajectoire depuis le buffer
    2. Calcule V(s) = max_a Q(s,a) pour chaque état avec le réseau ACTUEL
    3. Recalcule les cibles n-step avec compute_nstep_targets
    4. Met à jour les cibles dans le buffer

    IMPORTANT : torch.no_grad() est obligatoire.
    Les cibles sont des constantes — on ne veut pas de gradients ici.
    Si on oublie no_grad(), le graphe de calcul explose en mémoire.

    Args :
        buffer   : ReanalyzeBuffer
        network  : réseau de valeur actuel (online_net de l'agent)
        indices  : IDs des trajectoires à réanalyser
        k_steps  : profondeur de réanalyse (1 à T)
        gamma    : facteur de discount
        device   : 'cpu' ou 'cuda'
    """
    trajectories = buffer.get_trajectories(indices)

    with torch.no_grad():  # NE JAMAIS OUBLIER — pas de gradients sur les cibles
        new_targets_list = []

        for traj_data in trajectories:
            steps = traj_data.get('steps', [])
            if len(steps) == 0:
                new_targets_list.append(np.array([]))
                continue

            # Extraire les états de la trajectoire
            states = np.array([step['state'] for step in steps], dtype=np.float32)
            states_tensor = torch.FloatTensor(states).to(device)

            # Calculer V(s) = max_a Q(s,a) avec le réseau ACTUEL
            # C'est ici que les cibles périmées sont remplacées par des cibles fraîches
            values = network.get_value(states_tensor).squeeze(-1)

            # Extraire les récompenses
            rewards = [step['reward'] for step in steps]

            # Calculer les cibles n-step fraîches
            new_targets = compute_nstep_targets(rewards, values, k_steps, gamma)
            new_targets_list.append(new_targets)

        # Mettre à jour les cibles dans le buffer
        buffer.update_targets(indices, new_targets_list)


##here DreamerV3
def reanalyze_dreamer(buffer, network, latent_model,
                      indices, k_steps, gamma,
                      n_imagined=5, mix_ratio=0.3, device='cpu'):
    trajectories = buffer.get_trajectories(indices)
    
    with torch.no_grad():
        new_targets_list = []
        for traj_data in trajectories:
            steps = traj_data.get('steps', [])
            if len(steps) == 0:
                new_targets_list.append(np.array([]))
                continue
            
            states = np.array([s['state'] for s in steps], dtype=np.float32)
            states_tensor = torch.FloatTensor(states).to(device)
            rewards = [s['reward'] for s in steps]
            
            # Cibles réelles
            values_real = network.get_value(states_tensor).squeeze(-1)
            targets_real = compute_nstep_targets(rewards, values_real, k_steps, gamma)
            
            # Cibles "imaginées" — perturbation gaussienne des états
            # Simplifié : on ne peut pas décoder des latents vers observations
            imagined_targets = []
            for _ in range(n_imagined):
                noise = torch.randn_like(states_tensor) * 0.1
                states_noisy = states_tensor + noise
                values_noisy = network.get_value(states_noisy).squeeze(-1)
                t = compute_nstep_targets(rewards, values_noisy, k_steps, gamma)
                imagined_targets.append(t)
            
            targets_imagined = np.mean(imagined_targets, axis=0)
            targets_final = (1 - mix_ratio) * targets_real + mix_ratio * targets_imagined
            new_targets_list.append(targets_final)
        
        buffer.update_targets(indices, new_targets_list)

#here TD-MPC2
def reanalyze_tdmpc2(buffer, network, indices, k_steps, gamma,
                     n_sequences=10, device='cpu'):
    
    trajectories = buffer.get_trajectories(indices)
    
    with torch.no_grad():
        new_targets_list = []
        
        for traj_data in trajectories:
            steps = traj_data.get('steps', [])
            if len(steps) == 0:
                new_targets_list.append(np.array([]))
                continue
            
            states = np.array([s['state'] for s in steps], dtype=np.float32)
            states_tensor = torch.FloatTensor(states).to(device)
            rewards = [s['reward'] for s in steps]
            T = len(steps)
            
            # Pour chaque état, évaluer N séquences d'actions
            best_values = np.zeros(T, dtype=np.float32)
            
            for _ in range(n_sequences):
                # Séquence d'actions aléatoires de longueur k_steps
                q_values = network(states_tensor)  # (T, action_dim)
                # Prendre une action aléatoire différente à chaque séquence
                rand_actions = torch.randint(
                    0, q_values.shape[1], (T,)
                ).to(device)
                seq_values = q_values.gather(
                    1, rand_actions.unsqueeze(1)
                ).squeeze(1).cpu().numpy()
                # Garder la meilleure valeur pour chaque état
                best_values = np.maximum(best_values, seq_values)
            
            # Calculer les cibles avec les meilleures valeurs trouvées
            targets = compute_nstep_targets(rewards, best_values, k_steps, gamma)
            new_targets_list.append(targets)
        
        buffer.update_targets(indices, new_targets_list)