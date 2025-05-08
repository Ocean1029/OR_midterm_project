from __future__ import annotations

import time
import random
from typing import Mapping

import numpy as np
from numpy import ceil
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm  # 進度條
import argparse
from heuristic import heuristic_two_mode_jit
from reporter import export_strategies
from strategy_core import StrategySolution
from pathlib import Path



# ---------------------------------------------------
# 基本參數設定
# ---------------------------------------------------

NUM_INSTANCES = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

LEVELS = {
    "scale": {
        "Small":  {"N": 10,  "T": 6},
        "Medium": {"N": 100, "T": 20},
        "Large":  {"N": 500, "T": 50},
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

SCENARIOS: list[tuple[str, str, str]] = [
    ("Medium", "Medium", "Medium"),
    ("Small",  "Medium", "Medium"),
    ("Large",  "Medium", "Medium"),
    ("Medium", "Low",    "Medium"),
    ("Medium", "High",   "Medium"),
    ("Medium", "Medium", "Low"),
    ("Medium", "Medium", "High"),
]

BASE_CASE = {
        "N": 10,
        "T": 6,
        "demand": [
            [138.0, 55.0, 172.0, 194.0, 94.0, 185.0],
            [190.0, 101.0, 68.0, 185.0, 13.0, 136.0],
            [79.0, 179.0, 21.0, 49.0, 199.0, 200.0],
            [142.0, 103.0, 78.0, 131.0, 146.0, 155.0],
            [35.0, 62.0, 83.0, 90.0, 197.0, 49.0],
            [91.0, 95.0, 107.0, 127.0, 116.0, 183.0],
            [105.0, 164.0, 19.0, 116.0, 119.0, 175.0],
            [37.0, 155.0, 10.0, 77.0, 168.0, 32.0],
            [108.0, 185.0, 188.0, 176.0, 81.0, 172.0],
            [46.0, 178.0, 162.0, 200.0, 154.0, 199.0]
        ],
        "purchase_cost": [5000.0, 2000.0, 9000.0, 9000.0, 2000.0, 9000.0, 7000.0, 5000.0, 9000.0, 7000.0],
        "cv1": [44.0, 89.0, 86.0, 91.0, 50.0, 51.0, 83.0, 96.0, 80.0, 49.0],
        "cv2": [18.0, 45.0, 38.0, 46.0, 21.0, 25.0, 46.0, 49.0, 35.0, 20.0],
        "alpha": None,
        "I_0": [800.0, 600.0, 425.0, 350.0, 400.0, 524.0, 453.0, 218.0, 673.0, 200.0],
        "I_1": [0.0, 48.0, 0.0, 153.0, 0.0, 18.0, 28.0, 0.0, 109.0, 0.0],
        "I_2": [0.0, 0.0, 20.0, 0.0, 0.0, 23.0, 45.0, 0.0, 34.0, 0.0],
        "V_i": [0.073, 0.005, 0.043, 0.063, 0.045, 0.086, 0.079, 0.082, 0.068, 0.098],
        "container_cost": 2750,
        "inventory_cost": [100, 40, 180, 180, 40, 180, 140, 100, 180, 140]
    }

# ---------------------------------------------------
# 1. 產生單一情境的實例
# ---------------------------------------------------

def generate_instance(levels: dict,
                      scenario: tuple[str, str, str],
                      num_instances: int,
                      seed: int | None = None) -> list[dict]:
    """回傳 [instance_dict, ...]"""
    size, ship_lvl, inv_lvl = scenario
    rng = np.random.default_rng(seed)

    N = levels["scale"][size]["N"]
    T = levels["scale"][size]["T"]
    container_cost = levels["container_cost"][ship_lvl]
    inv_pct = levels["inventory_cost"][inv_lvl]

    def vrand(low, high, *shape):
        return rng.uniform(low, high, size=shape)

    inst_list = []
    for _ in range(num_instances):
        demand = vrand(0, 200, N, T)
        purchase_cost = vrand(1_000, 10_000, N)
        cv1 = vrand(40, 100, N)
        alpha = vrand(0.4, 0.6, N)
        cv2 = alpha * cv1
        I_2 = np.where(rng.random(N) < .5, 0, vrand(0, 50, N))
        I_1 = np.where(rng.random(N) < .5, 0, vrand(0, 200, N))
        I_0 = rng.uniform(demand[:, 0], 400)
        V_i = vrand(0, 1, N)

        inst_list.append(dict(
            N=N, T=T, demand=demand, purchase_cost=purchase_cost,
            cv1=cv1, cv2=cv2, alpha=alpha,
            I_0=I_0, I_1=I_1, I_2=I_2, V_i=V_i,
            container_cost=container_cost,
            inventory_cost_percent=inv_pct,
        ))
    return inst_list

def prep_instance(raw: dict) -> dict:
    """確保 list → np.ndarray"""
    import copy
    inst = copy.deepcopy(raw)
    # 需轉陣列欄位
    for key in ("demand", "purchase_cost", "cv1", "cv2", "I_0", "I_1", "I_2", "V_i", "inventory_cost"):
        if key in inst and not isinstance(inst[key], np.ndarray):
            inst[key] = np.array(inst[key])
    return inst

# ---------------------------------------------------
# 2. 解法：Relax & MIP
# ---------------------------------------------------


def build_model(instance: Mapping, integer: bool) -> tuple[gp.Model]:
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    pc = instance["purchase_cost"]
    cv1, cv2 = instance["cv1"], instance["cv2"]
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]
    Vi = instance["V_i"]
    cont_cost = instance["container_cost"]
    inv_cost = instance.get("inventory_cost", pc * instance.get("inventory_cost_percent", 0.02)) # 如果沒有給定 inventory_cost，就用 purchase_cost 乘上預設的百分比

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 1e-6)

    items, periods, methods = range(N), range(T), [0, 1, 2]
    lead, fix = {0: 1, 1: 2, 2: 3}, {0: 100, 1: 80, 2: 50}

    y_type = GRB.BINARY if integer else GRB.CONTINUOUS
    z_type = GRB.INTEGER if integer else GRB.CONTINUOUS

    x = model.addVars(items, methods, periods, lb=0, vtype=GRB.CONTINUOUS, name="x")
    v = model.addVars(items, periods, lb=0, vtype=GRB.CONTINUOUS, name="v")
    y = model.addVars(methods, periods, lb=0, ub=1, vtype=y_type, name="y")
    z = model.addVars(periods, lb=0, vtype=z_type, name="z")

    model.setObjective(gp.quicksum(inv_cost[i] * v[i, t] for i in items for t in periods) +
                   gp.quicksum((pc[i] + (cv1[i] if j == 0 else cv2[i] if j == 1 else 0)) * x[i, j, t]
                               for i in items for j in methods for t in periods) +
                   gp.quicksum(fix[j] * y[j, t] for j in methods for t in periods) +
                   gp.quicksum(cont_cost * z[t] for t in periods), GRB.MINIMIZE)

    for i in items:
        for t in periods:
            arrivals = gp.quicksum(
                x[i, j, t - (lead[j] - 1)] 
                for j in methods
                if t - (lead[j] - 1) >= 0
            ) # 代表第 i 個商品該月的到貨量 
            prev = I0[i] if t == 0 else v[i, t-1]
            transit = I1[i] if t == 0 else I2[i] if t == 1 else 0
            model.addConstr(
                v[i, t] == prev + transit + arrivals - demand[i, t]
            )

            # 服務水準式 (4)
            if t >= 1:
                model.addConstr(v[i, t-1] >= demand[i, t])

    bigM = demand.sum()
    for j in methods:
        for t in periods:
            model.addConstr(gp.quicksum(x[i, j, t] for i in items) <= bigM * y[j, t])
    for t in periods:
        model.addConstr(gp.quicksum(Vi[i] * x[i, 2, t] for i in items) <= 30 * z[t])

    model.optimize()

    return model

# ---------------------------------------------------
# 3. Naive Heuristic
# ---------------------------------------------------

def naive_heuristic(instance: Mapping):
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    purchase_cost = instance["purchase_cost"]
    cv1 = instance["cv1"]
    I_0, I_1, I_2 = instance["I_0"], instance["I_1"], instance["I_2"]

    total_cost = 0.0
    for i in range(N):
        for t in range(T):
            req = demand[i, t]
            if t == 0:
                req -= I_0[i] + I_1[i]
            elif t == 1:
                req -= I_2[i]
            if req > 0:
                total_cost += purchase_cost[i] * req + cv1[i] * req
    total_cost += 100 * T
    return total_cost

def solve_all(instance: Mapping) -> Dict[str, StrategySolution]:
    sols: Dict[str, StrategySolution] = {}

    # # --- MIP ---
    # t0 = time.time()
    # mip = build_model(instance, integer=True)
    # t1 = time.time() - t0
    # mip_orders = {(i,j,t): v.X
    #               for v in mip.getVars() if v.VarName.startswith("x[") and v.X>1e-6
    #               for i,j,t in [map(int, v.VarName[2:-1].split(","))]}
    # sols["MIP"] = StrategySolution(
    #     "MIP", mip_orders, mip.ObjVal,
    #     total_qty=sum(mip_orders.values()),
    #     total_containers=sum(v.X for v in mip.getVars() if v.VarName.startswith("z[")),
    #     run_time=t1
    # )

    # --- Relaxation ---
    t0 = time.time()
    lp = build_model(instance, integer=False)
    t1 = time.time() - t0
    lp_orders = {(i,j,t): v.X
                 for v in lp.getVars() if v.VarName.startswith("x[") and v.X>1e-6
                 for i,j,t in [map(int, v.VarName[2:-1].split(","))]}
    sols["Relax"] = StrategySolution(
        "Relax", lp_orders, lp.ObjVal,
        total_qty=sum(lp_orders.values()),
        total_containers=sum(v.X for v in lp.getVars() if v.VarName.startswith("z[")),
        run_time=t1
    )

    # --- Heuristic-JIT ---
    t0 = time.time()
    orders_h, obj_h = heuristic_two_mode_jit(instance)
    t1 = time.time() - t0
    V = instance["V_i"]; N, T = instance["N"], instance["T"]
    cont_h = sum(ceil(sum(V[i]*orders_h.get((i,2,t),0) for i in range(N)) / 30)
                 for t in range(T))
    sols["Heur"] = StrategySolution(
        "Heur", orders_h, obj_h,
        total_qty=sum(orders_h.values()),
        total_containers=cont_h,
        run_time=t1
    )
    return sols

# ---------------------------------------------------
# 5. Evaluate 210 instances
# ---------------------------------------------------
def run_eval():
    records = []
    rng_seed = SEED
    for sc in SCENARIOS:
        inst_list = generate_instance(LEVELS, sc, NUM_INSTANCES, seed=rng_seed)
        for k, raw in enumerate(tqdm(inst_list, desc=str(sc)), 1):
            inst = prep_instance(raw)
            sols = solve_all(inst)
            opt = sols["Relax"].total_cost
            for tag in ("Relax", "Heur"):
                gap = (sols[tag].total_cost - opt) / opt * 100
                records.append({
                    "Scenario": sc, "Case": k, "Method": tag,
                    "Obj": sols[tag].total_cost,
                    "Gap(%)": gap,
                    "Time(s)": sols[tag].run_time
                })

    df = pd.DataFrame(records)
    out_dir = Path("report") / f"Evaluation_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"raw_results.csv", index=False)

    summary = df.groupby(["Scenario", "Method"]).agg(
        ObjVal_Average=("Obj", "mean"),
        Gap_Average=("Gap(%)", "mean"),
        Gap_Std=("Gap(%)", "std"),
        Time_Average=("Time(s)", "mean"),
        Time_Std=("Time(s)", "std")
    ).reset_index()
    summary.to_excel(out_dir/"summary.xlsx", index=False)
    print("✔ Summary saved to", out_dir/"summary.xlsx")

# ---------------------------------------------------
# 6. 單一 BaseCase 報表
# ---------------------------------------------------
def run_strategies():
    inst = prep_instance(BASE_CASE)
    sols = solve_all(inst)
    export_strategies(
        list(sols.values()),
        instance = inst,
        out_prefix= "BaseCase"
    )

# ---------------------------------------------------
# 7.  CLI
# ---------------------------------------------------
if __name__ == "__main__":
    run_eval()
    run_strategies()