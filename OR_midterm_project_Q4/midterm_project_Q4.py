# Import necessary libraries
import random
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Step 1: Define parameters and levels
# -----------------------------
levels = {
    "scale": {
        "Small": {"N": 10, "T": 6},
        "Medium": {"N": 100, "T": 20},
        "Large": {"N": 500, "T": 50},
    },
    "container_cost": {
        "Low": 1375,
        "Medium": 2750,
        "High": 5500,
    },
    "inventory_cost": {
        "Low": 0.01,
        "Medium": 0.02,
        "High": 0.04,
    },
}

# -----------------------------
# Step 2: 7 Scenario Combination
# -----------------------------
scenarios = [
    ("Medium", "Medium", "Medium"),  # Scenario 1
    ("Small", "Medium", "Medium"),   # Scenario 2
    ("Large", "Medium", "Medium"),   # Scenario 3
    ("Medium", "Low", "Medium"),     # Scenario 4
    ("Medium", "High", "Medium"),    # Scenario 5
    ("Medium", "Medium", "Low"),     # Scenario 6
    ("Medium", "Medium", "High"),    # Scenario 7
]

# -----------------------------
# Step 3: Generate 30 instances for each scenario
# -----------------------------
random.seed(42)
num_instances = 30
instances = {}

for scenario in scenarios:
    size, ship_cost_level, inv_cost_level = scenario
    N, T = levels["scale"][size]["N"], levels["scale"][size]["T"]
    container_cost = levels["container_cost"][ship_cost_level]
    inventory_cost_percent = levels["inventory_cost"][inv_cost_level]

    instances[scenario] = []
    for i in range(num_instances):
        demand = np.random.uniform(0, 200, size=(N, T))
        purchase_cost = np.random.uniform(1000, 10000, size=N)
        cv1 = np.random.uniform(40, 100, size=N)
        alpha = np.random.uniform(0.4, 0.6, size=N)
        cv2 = alpha * cv1
        I_2 = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 50, size=N))
        I_1 = np.where(np.random.rand(N) < 0.5, 0, np.random.uniform(0, 200, size=N))
        I_0 = np.random.uniform(demand[:, 0], 400, size=N)
        V_i = np.random.uniform(0, 1, size=N)

        instances[scenario].append({
            "N": N,
            "T": T,
            "demand": demand,
            "purchase_cost": purchase_cost,
            "cv1": cv1,
            "cv2": cv2,
            "alpha": alpha,
            "I_0": I_0,
            "I_1": I_1,
            "I_2": I_2,
            "V_i": V_i,
            "container_cost": container_cost,
            "inventory_cost_percent": inventory_cost_percent
        })     

# -----------------------------
# Step 4: Gurobi Solver (Linear Relaxation)
# -----------------------------

def solve_linear_relaxation(instance):
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    purchase_cost = instance["purchase_cost"]
    cv1 = instance["cv1"]
    cv2 = instance["cv2"]
    alpha = instance["alpha"]
    I_0 = instance["I_0"]
    I_1 = instance["I_1"]
    I_2 = instance["I_2"]
    V_i = instance["V_i"]
    container_cost = instance["container_cost"]
    inventory_cost_rate = instance["inventory_cost_percent"]

    inventory_cost = purchase_cost * inventory_cost_rate

    model = gp.Model()
    model.setParam('OutputFlag', 0)

    # Index sets
    items = range(N)
    periods = range(T)
    methods = [0, 1, 2]  # 0: Express, 1: Air, 2: Ocean
    lead_time = {0: 1, 1: 2, 2: 3}
    fixed_cost = {0: 100, 1: 80, 2: 50}

    # Decision variables
    x = model.addVars(items, methods, periods, lb=0, vtype=GRB.CONTINUOUS, name="x")
    v = model.addVars(items, periods, lb=0, vtype=GRB.CONTINUOUS, name="v")
    y = model.addVars(methods, periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(periods, lb=0, vtype=GRB.CONTINUOUS, name="z")

    # Objective
    total_cost = gp.quicksum(
        inventory_cost[i] * v[i, t] for i in items for t in periods
    ) + gp.quicksum(
        (purchase_cost[i] + (cv1[i] if j == 0 else cv2[i] if j == 1 else 0)) * x[i, j, t]
        for i in items for j in methods for t in periods
    ) + gp.quicksum(
        fixed_cost[j] * y[j, t] for j in methods for t in periods
    ) + gp.quicksum(
        container_cost * z[t] for t in periods
    )

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints

    # Inventory balance
    for i in items:
        for t in periods:
            # Demand
            demand_t = demand[i, t]

            # Previous inventory
            if t == 0:
                prev_inventory = I_0[i]
                in_transit = I_1[i]
            elif t == 1:
                prev_inventory = v[i, t - 1]
                in_transit = I_2[i]
            else:
                prev_inventory = v[i, t - 1]
                in_transit = 0

            # Arrivals from orders placed in earlier periods
            arrivals = gp.quicksum(
                x[i, j, t - lead_time[j]] for j in methods if t - lead_time[j] >= 0
            )

            model.addConstr(
                v[i, t] == prev_inventory + in_transit + arrivals - demand_t,
                name=f"inventory_balance_{i}_{t}"
            )

    # If any quantity is ordered with method j in t, then y[j, t] should be triggered
    bigM = demand.sum()
    for j in methods:
        for t in periods:
            model.addConstr(
                gp.quicksum(x[i, j, t] for i in items) <= bigM * y[j, t],
                name=f"trigger_y_{j}_{t}"
            )

    # Container constraint for ocean shipping (method 2)
    for t in periods:
        model.addConstr(
            gp.quicksum(V_i[i] * x[i, 2, t] for i in items) <= 30 * z[t],
            name=f"container_limit_{t}"
        )

    # Solve the model
    model.optimize()
    return model

# -----------------------------
# Step 5: Very naive heuristic
# -----------------------------
def naive_heuristic(instance):
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    purchase_cost = instance["purchase_cost"]
    cv1 = instance["cv1"]
    I_0 = instance["I_0"]
    I_1 = instance["I_1"]
    I_2 = instance["I_2"]

    total_cost = 0.0

    for i in range(N):
        remaining_I = I_0[i]
        for t in range(T):
            D = demand[i, t]
            if t == 0:
                D -= I_0[i]
                D -= I_1[i]
            elif t == 1:
                D -= I_2[i]
            # If demand is not met, we need to order
            if D > 0:
                total_cost += purchase_cost[i] * D + cv1[i] * D  # express only

    total_cost += 100 * T  # fixed cost for express shipping
    return total_cost

# -----------------------------
# Step 6: Using two model to solve the instance and evaluate optimality gap
# -----------------------------

import statistics

def evaluate_gap_all_instances(instances):
    scenario_gaps = {}

    for scenario, inst_list in instances.items():
        gaps = []
        for idx, inst in enumerate(inst_list):
            # Solve LP (Relaxation)
            model = solve_linear_relaxation(inst)
            if model.status != GRB.OPTIMAL:
                continue
            opt_val = model.objVal

            # Naive heuristic
            naive_val = naive_heuristic(inst)

            # Gap
            gap = (naive_val - opt_val) / opt_val * 100
            gaps.append(gap)

        if gaps:
            mean_gap = statistics.mean(gaps)
            std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
            scenario_gaps[scenario] = (mean_gap, std_gap)

    print("\n=== Summary: Mean and Std of Optimality Gap per Scenario ===")
    for scenario, (mean_gap, std_gap) in scenario_gaps.items():
        print(f"{scenario}: Mean Gap = {mean_gap:.2f}%, Std = {std_gap:.2f}%")

evaluate_gap_all_instances(instances)




# -----------------------------
from generate_experiment_output import generate_experiment_output
final_df = generate_experiment_output(
    instances=instances,
    solve_linear_relaxation=solve_linear_relaxation,
    naive_heuristic=naive_heuristic
)
