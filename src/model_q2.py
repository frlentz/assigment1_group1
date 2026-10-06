"""
Question 2 models: disutility of deviating from a reference load profile.

LinearDisutilityModel (2.b): D_t = c_L * |L_t - l_ref_t|, written as an LP by splitting
the deviation into an upward and a downward part.

Convention: cost minimisation. The balance is written supply == demand, so its dual
is the marginal cost of energy and is non-negative.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .model import FlexibleConsumerModel

class LinearDisutilityModel(FlexibleConsumerModel):
    """
    Consumer minimizing procurement cost plus a linear disutility (Question 2.b)

    Needs ``data.reference_load`` (kWh/h) and ``data.linear_disutility`` (DKK/kWh)

    Variables   : load, pv, export, dev_up, dev_down (kWh, one per hour)
    Constraints : balance, load_min, load_max, pv_min, pv_max, import_nonneg, export_nonneg, deviation, dev_up_nonneg, dev_down_nonneg

    This class inherits from FlexibleConsumerModel, polymorphism is used to overload the build function from the FlexibleConsumerModel
    """

    def build(self) -> "LinearDisutilityModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None or d.linear_disutility is None:
            raise ValueError(f"{d.question} has no reference profile or linear diutility")

        # Decision variables
        self.var['load'] = m.addVars(T, lb=-GRB.INFINITY, name='load')  # L_t: flexible consumption
        self.var['pv'] = m.addVars(T, lb=-GRB.INFINITY, name='pv')      # g_t: PV produced / used
        self.var['import'] = m.addVars(T, lb=-GRB.INFINITY, name='import')      # i_t: grid import
        self.var['export'] = m.addVars(T, lb=-GRB.INFINITY, name='export')      # e_t: grid export
        self.var['dev_up'] = m.addVars(T, lb=-GRB.INFINITY, name='dev_up')      # d+_t: how many kWh the load is above the reference at hour t
        self.var['dev_down'] = m.addVars(T, lb=-GRB.INFINITY, name='dev_down')  # d-_t: how many kWh the load is below the reference at hour t

        # Objective
        load = self.var['load']
        p = d.energy_price
        pv = self.var['pv']
        imp = self.var['import']
        exp = self.var['export']
        d_up = self.var['dev_up']
        d_down = self.var['dev_down']

        m.setObjective(
            gp.quicksum(
                imp[t]*(p[t] + d.import_tariff) -
                exp[t]*(p[t] - d.export_tariff) +
                d.pv_marginal_cost * pv[t] +
                d.linear_disutility*(d_up[t] + d_down[t])
            for t in T),
            GRB.MINIMIZE
        )

        # Constraints
        self.con['balance'] = m.addConstrs((pv[t] + imp[t] - exp[t] == load[t] for t in T), name='balance')
        self.con['load_min'] = m.addConstrs((load[t] >= d.load_min_kWh for t in T), name='load_min')
        self.con['load_max'] = m.addConstrs((load[t] <= d.load_max_kWh for t in T), name='load_max')
        self.con['pv_min'] = m.addConstrs((pv[t] >= 0 for t in T), name='pv_min')
        self.con['pv_max'] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T), name='pv_max')
        self.con['import_nonneg'] = m.addConstrs((imp[t] >= 0 for t in T), name='import_nonneg')
        self.con['export_nonneg'] = m.addConstrs((exp[t] >= 0 for t in T), name='export_nonneg')

        # New constraints
        self.con['dev_up_nonneg'] = m.addConstrs((d_up[t] >= 0 for t in T), name='dev_up_nonneg') # d+_t >= 0
        self.con['dev_down_nonneg'] = m.addConstrs((d_down[t] >= 0 for t in T), name='dev_down_nonneg') # d-_t >= 0
        self.con['deviation'] = m.addConstrs((load[t] - d.reference_load[t] == d_up[t] - d_down[t] for t in T), name='deviation') # L_t - l^ref_{t} = d+_t - d-_t       


        m.update()

        return self

class QuadraticDisutilityModel(FlexibleConsumerModel):
    """
    Consumer minimizing procurement cost plus a quadratic disutility (Question 2.c)
    D_t = c_Q * (L_t - l_ref_t)^2 a convex QP. No auxiliary variables needed.

    Needs: ``data.reference.load`` (kWh/h) and ``data.quadratic_disutility`` (DKK/kWh^2)

    Variables: load, pv, import, export (kWh, one per hour)
    Constraints: balance, load_min, load_max, pv_min, pv_max, import_nonneg, export_nonneg
    """

    def build(self) -> "QuadraticDisutilityModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None or d.quadratic_disutility is None:
            raise ValueError(f'{d.question} has no reference profile or quadratic disutility')

        # Decision variables
        self.var['load'] = m.addVars(T, lb=-GRB.INFINITY, name='load')  # L_t: flexible consumption
        self.var['pv'] = m.addVars(T, lb=-GRB.INFINITY, name='pv')      # g_t: PV produced / used
        self.var['import'] = m.addVars(T, lb=-GRB.INFINITY, name='import')      # i_t: grid import
        self.var['export'] = m.addVars(T, lb=-GRB.INFINITY, name='export')      # e_t: grid export

        # Objective
        load = self.var['load']
        p = d.energy_price
        pv = self.var['pv']
        imp = self.var['import']
        exp = self.var['export']
        ref = d.reference_load

        m.setObjective(
            gp.quicksum(
                imp[t]*(p[t] + d.import_tariff) -
                exp[t]*(p[t] - d.export_tariff) +
                d.pv_marginal_cost * pv[t] +
                d.quadratic_disutility*(load[t] - ref[t])*(load[t] - ref[t])
            for t in T),
            GRB.MINIMIZE
        )

        # Constraints
        self.con['balance'] = m.addConstrs((pv[t] + imp[t] - exp[t] == load[t] for t in T), name='balance')
        self.con['load_min'] = m.addConstrs((load[t] >= d.load_min_kWh for t in T), name='load_min')
        self.con['load_max'] = m.addConstrs((load[t] <= d.load_max_kWh for t in T), name='load_max')
        self.con['pv_min'] = m.addConstrs((pv[t] >= 0 for t in T), name='pv_min')
        self.con['pv_max'] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T), name='pv_max')
        self.con['import_nonneg'] = m.addConstrs((imp[t] >= 0 for t in T), name='import_nonneg')
        self.con['export_nonneg'] = m.addConstrs((exp[t] >= 0 for t in T), name='export_nonneg')

        m.update()

        return self