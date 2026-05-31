import numpy as np
import pandas as pd
import gymnasium as gym


class EnergyBuildingEnv(gym.Env):
    DATA_URL = (
        "https://raw.githubusercontent.com/intelligent-environments-lab/"
        "CityLearn/v2.1.0/citylearn/data/"
        "citylearn_challenge_2022_phase_1/Building_1.csv"
    )

    def __init__(self, battery_capacity=6.4, battery_efficiency=0.9):
        super().__init__()
        self.df = pd.read_csv(self.DATA_URL).fillna(0).reset_index(drop=True)
        self.battery_capacity   = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.n_steps            = len(self.df)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self.reset()

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['Equipment Electric Power [kWh]'],
            row['Solar Generation [W/kW]'] / 1000,
            self.battery_charge / self.battery_capacity,
            row['Hour'] / 24.0,
            row['Month'] / 12.0,
        ], dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.step_idx       = 0
        self.battery_charge = self.battery_capacity / 2
        return self._get_obs(), {}

    def step(self, action):
        row         = self.df.iloc[self.step_idx]
        consumption = row['Equipment Electric Power [kWh]']
        solar       = row['Solar Generation [W/kW]'] / 1000

        if action == 2:
            charge_amount       = min(1.0, self.battery_capacity - self.battery_charge)
            self.battery_charge += charge_amount * self.battery_efficiency
            net_consumption     = consumption + charge_amount - solar
        elif action == 0:
            discharge_amount    = min(1.0, self.battery_charge)
            self.battery_charge -= discharge_amount
            net_consumption     = consumption - discharge_amount * self.battery_efficiency - solar
        else:
            net_consumption = consumption - solar

        net_consumption = max(0, net_consumption)
        peak_penalty    = -2.0 * net_consumption if net_consumption > 3.0 else 0.0
        reward          = -net_consumption + peak_penalty

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1

        return self._get_obs(), reward, done, False, {}


def register_energy_env():
    if 'EnergyBuilding-v0' not in gym.envs.registry:
        gym.register(
            id='EnergyBuilding-v0',
            entry_point='envs.energy_building_env:EnergyBuildingEnv'
        )