import torch
import numpy as np
from typing import List, Dict, Optional


def compute_nstep_targets(rewards: List[float], values: torch.Tensor,
                          k_steps: int, gamma: float) -> np.ndarray:

    T = len(rewards)
    targets = np.zeros(T, dtype=np.float32)
    values_np = values.cpu().numpy() if isinstance(values, torch.Tensor) else values

    for t in range(T):
        G = 0.0
        for j in range(k_steps):
            if t + j < T:
                G += (gamma ** j) * rewards[t + j]
        if t + k_steps < T:
            G += (gamma ** k_steps) * values_np[t + k_steps]
        targets[t] = G

    return targets


def reanalyze(buffer, network: torch.nn.Module,
              indices: List[int], k_steps: int,
              gamma: float, device: str = 'cpu') -> None:
    
    trajectories = buffer.get_trajectories(indices)

    with torch.no_grad():  
        new_targets_list = []

        for traj_data in trajectories:
            steps = traj_data.get('steps', [])
            if len(steps) == 0:
                new_targets_list.append(np.array([]))
                continue

            states = np.array([step['state'] for step in steps], dtype=np.float32)
            states_tensor = torch.FloatTensor(states).to(device)

            values = network.get_value(states_tensor).squeeze(-1)

            rewards = [step['reward'] for step in steps]

            new_targets = compute_nstep_targets(rewards, values, k_steps, gamma)
            new_targets_list.append(new_targets)

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
            
            
            values_real = network.get_value(states_tensor).squeeze(-1)
            targets_real = compute_nstep_targets(rewards, values_real, k_steps, gamma)
            
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
            
            best_values = np.zeros(T, dtype=np.float32)
            
            for _ in range(n_sequences):
    
                q_values = network(states_tensor) 
                rand_actions = torch.randint(
                    0, q_values.shape[1], (T,)
                ).to(device)
                seq_values = q_values.gather(
                    1, rand_actions.unsqueeze(1)
                ).squeeze(1).cpu().numpy()
                best_values = np.maximum(best_values, seq_values)
            
            targets = compute_nstep_targets(rewards, best_values, k_steps, gamma)
            new_targets_list.append(targets)
        
        buffer.update_targets(indices, new_targets_list)