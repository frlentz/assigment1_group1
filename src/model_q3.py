from __future__ import annotations
import gurobipy as gp
from .model_q2 import QuadraticDisutilityModel

class MinEnergyModel(QuadraticDisutilityModel):
    def build(self) -> "MinEnergyModel":
        """
        Input : self.data (InputData), with min_daily_energy_kWh = E_min[kWh]
        Output: self, with self.con['min_energy'] added; its dual is mu[DKK/kWh]
        """
        d = self.data

        # Variables, objective and constraints of Question 2.c
        super().build()
        m = self.m # model
        T = self.T # hours
        load = self.var['load']

        # New constraint
        self.con['min_energy'] = m.addConstr(gp.quicksum(load[t] for t in T) >= d.min_daily_energy_kWh, name='min_energy') # sum_t L_t >= E_min
        m.update()

        return self