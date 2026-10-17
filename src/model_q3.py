import gurobipy as gp
from gurobipy import GRB
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

class BatteryModel(MinEnergyModel):
    def build(self) -> "BatteryModel":
        """
        Input : self.data (InputData), with the battery: capacity S_max[kWh], max charging/discharging[kW],
                efficiencies eta_ch, eta_dis[-] and initial state of charge S_init[kWh]
        Output: self, with the variables charge, discharge, soc[kWh] and the battery constraints added,
                and the power balance replaced by one that includes the battery
        """
        d = self.data

        # Variables, objective and constraints of Question 3.a
        super().build()
        m = self.m # model
        T = self.T # hours
        load = self.var['load']
        pv = self.var['pv']
        imp = self.var['import']
        exp = self.var['export']

        # New variables
        ch = self.var['charge'] = m.addVars(T, lb=-GRB.INFINITY, name='charge')       # P^ch_t
        dis = self.var['discharge'] = m.addVars(T, lb=-GRB.INFINITY, name='discharge') # P^dis_t
        soc = self.var['soc'] = m.addVars(T, lb=-GRB.INFINITY, name='soc')             # S_t

        # Power balance with the battery replaces the old one
        m.remove(self.con['balance'])
        self.con['balance'] = m.addConstrs((pv[t] + imp[t] - exp[t] + dis[t] == load[t] + ch[t] for t in T), name='balance')

        # New constraints
        self.con['soc'] = m.addConstrs((soc[t] == (soc[t - 1] if t > 0 else d.battery_initial_soc_kWh) + d.battery_charging_efficiency * ch[t] - dis[t] / d.battery_discharging_efficiency for t in T), name='soc') # S_t = S_{t-1} + eta_ch P^ch_t - P^dis_t / eta_dis
        self.con['soc_nonneg'] = m.addConstrs((soc[t] >= 0 for t in T), name='soc_nonneg')
        self.con['soc_max'] = m.addConstrs((soc[t] <= d.battery_capacity_kWh for t in T), name='soc_max')
        self.con['charge_nonneg'] = m.addConstrs((ch[t] >= 0 for t in T), name='charge_nonneg')
        self.con['charge_max'] = m.addConstrs((ch[t] <= d.battery_max_charge_kW for t in T), name='charge_max')
        self.con['discharge_nonneg'] = m.addConstrs((dis[t] >= 0 for t in T), name='discharge_nonneg')
        self.con['discharge_max'] = m.addConstrs((dis[t] <= d.battery_max_discharge_kW for t in T), name='discharge_max')
        self.con['soc_end'] = m.addConstr(soc[T[-1]] >= d.battery_initial_soc_kWh, name='soc_end') # S_23 >= S_init
        m.update()

        return self