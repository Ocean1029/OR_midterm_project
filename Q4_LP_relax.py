import time
import gurobipy as gp
from gurobipy import GRB
from typing import Mapping
from Q4_strategy_core import StrategySolution

def lp_relaxation(instance: Mapping) -> StrategySolution:
    """
    Solve the problem using LP relaxation.
    
    Args:
        instance: Problem instance dictionary
    
    Returns:
        StrategySolution object containing the solution
    """
    start = time.time()
    
    # Unpack instance data
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    pc = instance["purchase_cost"]
    cv1, cv2 = instance["cv1"], instance["cv2"]
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]
    Vi = instance["V_i"]
    cont_cost = instance["container_cost"]
    inv_cost = instance.get(
        "inventory_cost",
        pc * instance.get("inventory_cost_percent", 0.02)
    )

    # Create model
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-6)

    # Define sets
    items, periods, methods = range(N), range(T), [0, 1, 2]
    lead, fix = {0: 1, 1: 2, 2: 3}, {0: 100, 1: 80, 2: 50}

    # Variables
    x = model.addVars(items, methods, periods, lb=0, vtype=GRB.CONTINUOUS, name="x")
    v = model.addVars(items, periods, lb=0, vtype=GRB.CONTINUOUS, name="v")
    y = model.addVars(methods, periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(periods, lb=0, vtype=GRB.CONTINUOUS, name="z")

    # Objective
    model.setObjective(
        gp.quicksum(inv_cost[i] * v[i, t] for i in items for t in periods) +
        gp.quicksum((pc[i] + (cv1[i] if j == 0 else cv2[i] if j == 1 else 0)) * x[i, j, t]
                    for i in items for j in methods for t in periods) +
        gp.quicksum(fix[j] * y[j, t] for j in methods for t in periods) +
        gp.quicksum(cont_cost * z[t] for t in periods),
        GRB.MINIMIZE
    )

    # Constraints
    for i in items:
        for t in periods:
            # Inventory balance
            arrivals = gp.quicksum(
                x[i, j, t - (lead[j] - 1)] 
                for j in methods
                if t - (lead[j] - 1) >= 0
            )
            prev = I0[i] if t == 0 else v[i, t-1]
            transit = I1[i] if t == 0 else I2[i] if t == 1 else 0
            model.addConstr(
                v[i, t] == prev + transit + arrivals - demand[i, t]
            )

            # Service level constraint
            if t >= 1:
                model.addConstr(v[i, t-1] >= demand[i, t])

    # Big-M constraints
    bigM = demand.sum()
    for j in methods:
        for t in periods:
            model.addConstr(gp.quicksum(x[i, j, t] for i in items) <= bigM * y[j, t])
    
    # Container capacity constraints
    for t in periods:
        model.addConstr(gp.quicksum(Vi[i] * x[i, 2, t] for i in items) <= 30 * z[t])

    # Solve
    model.optimize()

    # Extract solution
    runtime = time.time() - start
    orders = {
        (i, j, t): v.X
        for v in model.getVars() if v.VarName.startswith("x[") and v.X > 1e-6
        for i, j, t in [map(int, v.VarName[2:-1].split(","))]
    }

    return StrategySolution(
        name="LP_Relax",
        orders=orders,
        total_cost=model.objVal,
        total_qty=sum(orders.values()),
        total_containers=sum(v.X for v in model.getVars() if v.VarName.startswith("z[")),
        run_time=runtime
    ) 