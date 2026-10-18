"""
Question 2 models: disutility of deviating from a reference load profile.

LinearDisutilityModel (2.b): D_t = c_L * |L_t - l_ref_t|, written as an LP by splitting
the deviation into an upward and a downward part.
QuadraticDisutilityModel (2.c): D_t = c_Q * (L_t - l_ref_t)^2, a QP.
NonSeparableDisutilityModel (2.e): c_Q times the square of a deviation that accumulates and
fades by a factor alpha per hour, so sustained deviations cost more. Not separable in time.

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

class NonSeparableDisutilityModel(FlexibleConsumerModel):
    """
    Consumer minimizing procurement cost plus a quadratic disutility of the accumulated deviation (Question 2.e)

    The felt deviation s_t (kWh) remembers past deviations and fades by a factor alpha per hour:
        s_t = alpha * s_{t-1} + (L_t - l_ref_t),   s_{-1} = 0
        D   = c_Q * [(1 - alpha^2) * sum_t s_t^2 + alpha^2 * s_23^2]
            = c_Q * sum_t sum_k alpha^|t-k| (L_t - l_ref_t)(L_k - l_ref_k)
    The last term is the disutility of s_23 fading out after the end of the day. A deviation in one
    isolated hour costs c_Q * dev^2 as in 2.c, and alpha = 0 gives back QuadraticDisutilityModel.

    Needs: ``data.reference_load`` (kWh/h) and ``data.quadratic_disutility`` (DKK/kWh^2)
    memory: alpha in [0, 1), dimensionless; not in the data files (our own choice for 2.e, base case 0.7)

    Variables: load, pv, import, export, felt_dev (kWh, one per hour)
    Constraints: balance, load_min, load_max, pv_min, pv_max, import_nonneg, export_nonneg, felt_dev
    """

    def __init__(self, data, memory: float = 0.7, **kwargs):
        if not 0 <= memory < 1:
            raise ValueError(f'memory must lie in [0, 1), got {memory}')
        super().__init__(data, **kwargs)
        self.memory = memory

    def build(self) -> "NonSeparableDisutilityModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None or d.quadratic_disutility is None:
            raise ValueError(f'{d.question} has no reference profile or quadratic disutility')

        # Decision variables
        self.var['load'] = m.addVars(T, lb=-GRB.INFINITY, name='load')  # L_t: flexible consumption
        self.var['pv'] = m.addVars(T, lb=-GRB.INFINITY, name='pv')      # g_t: PV produced / used
        self.var['import'] = m.addVars(T, lb=-GRB.INFINITY, name='import')      # i_t: grid import
        self.var['export'] = m.addVars(T, lb=-GRB.INFINITY, name='export')      # e_t: grid export
        self.var['felt_dev'] = m.addVars(T, lb=-GRB.INFINITY, name='felt_dev')  # s_t: accumulated deviation, free sign

        # Objective
        load = self.var['load']
        p = d.energy_price
        pv = self.var['pv']
        imp = self.var['import']
        exp = self.var['export']
        s = self.var['felt_dev']
        ref = d.reference_load
        alpha, c_Q, last = self.memory, d.quadratic_disutility, T[-1]

        m.setObjective(
            gp.quicksum(
                imp[t]*(p[t] + d.import_tariff) -
                exp[t]*(p[t] - d.export_tariff) +
                d.pv_marginal_cost * pv[t] +
                c_Q*(1 - alpha**2)*s[t]*s[t]
            for t in T) + c_Q*alpha**2*s[last]*s[last],
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

        # New constraint: s_t = alpha * s_{t-1} + L_t - l^ref_t, the only link between the hours
        self.con['felt_dev'] = m.addConstrs(
            (s[t] == (alpha*s[t - 1] if t > 0 else 0) + load[t] - ref[t] for t in T), name='felt_dev')

        m.update()

        return self