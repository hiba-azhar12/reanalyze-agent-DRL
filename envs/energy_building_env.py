import numpy as np
import pandas as pd
import gymnasium as gym


class EnergyBuildingEnv(gym.Env):
    BASE_URL = (
        "https://raw.githubusercontent.com/intelligent-environments-lab/"
        "CityLearn/v2.1.0/citylearn/data/citylearn_challenge_2022_phase_1/"
    )

    # Valeurs max réelles du dataset pour normalisation
    MAX_CONSUMPTION = 7.99
    MAX_SOLAR       = 976.25
    MAX_CARBON      = 0.28
    MAX_PRICE       = 0.54

    def __init__(self, battery_capacity=6.4, battery_efficiency=0.9):
        super().__init__()
        self.df = pd.read_csv(
            self.BASE_URL + "Building_1.csv"
        ).fillna(0).reset_index(drop=True)

        self.carbon = pd.read_csv(
            self.BASE_URL + "carbon_intensity.csv"
        ).fillna(0).reset_index(drop=True)

        self.pricing = pd.read_csv(
            self.BASE_URL + "pricing.csv"
        ).fillna(0).reset_index(drop=True)

        self.battery_capacity   = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.n_steps            = len(self.df)

        # État : consommation, solaire, batterie, heure, mois, carbon, prix actuel, prix 24h
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self.reset()

    def _get_total_consumption(self, row):
        return (
            row['Equipment Electric Power [kWh]']
            + row['DHW Heating [kWh]']
            + row['Cooling Load [kWh]']
            + row['Heating Load [kWh]']
        )

    def _get_obs(self):
        row     = self.df.iloc[self.step_idx]
        carbon  = float(self.carbon.iloc[self.step_idx % len(self.carbon)]['kg_CO2/kWh'])
        pricing = self.pricing.iloc[self.step_idx % len(self.pricing)]

        consumption = self._get_total_consumption(row)

        return np.array([
            np.clip(consumption / self.MAX_CONSUMPTION, 0.0, 1.0),
            np.clip(row['Solar Generation [W/kW]'] / self.MAX_SOLAR, 0.0, 1.0),
            self.battery_charge / self.battery_capacity,
            row['Hour'] / 24.0,
            row['Month'] / 12.0,
            np.clip(carbon / self.MAX_CARBON, 0.0, 1.0),
            np.clip(float(pricing['Electricity Pricing [$/kWh]']) / self.MAX_PRICE, 0.0, 1.0),
            np.clip(float(pricing['24h Prediction Electricity Pricing [$/kWh]']) / self.MAX_PRICE, 0.0, 1.0),
        ], dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.step_idx       = 0
        self.battery_charge = self.battery_capacity / 2
        return self._get_obs(), {}

    def step(self, action):
        row     = self.df.iloc[self.step_idx]
        pricing = self.pricing.iloc[self.step_idx % len(self.pricing)]
        price   = float(pricing['Electricity Pricing [$/kWh]'])

        consumption = self._get_total_consumption(row)
        solar       = row['Solar Generation [W/kW]'] / self.MAX_SOLAR * self.MAX_CONSUMPTION

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

        net_consumption = max(0.0, net_consumption)

        # Reward simple et clair
        peak_penalty  = -2.0 * net_consumption if net_consumption > 3.0 else 0.0
        price_penalty = -price * net_consumption
        reward        = -net_consumption + peak_penalty + price_penalty

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1

        return self._get_obs(), reward, done, False, {}


def register_energy_env():
    if 'EnergyBuilding-v0' not in gym.envs.registry:
        gym.register(
            id='EnergyBuilding-v0',
            entry_point='envs.energy_building_env:EnergyBuildingEnv'
        )