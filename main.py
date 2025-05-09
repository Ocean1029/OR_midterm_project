from __future__ import annotations

import time
import random
from typing import Mapping

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm  # 進度條
from typing import Dict
from naive import naive_solution
from heuristic import heuristic_two_mode_jit
from strategy import export_strategies
from strategy_core import StrategySolution
from pathlib import Path
import copy
    
# ---------------------------------------------------
# 基本參數設定
# ---------------------------------------------------

NUM_INSTANCES = 1
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

def prep_instance(raw: dict) -> dict:
    """把 Base_Case 的 list → np.ndarray"""
    instance = copy.deepcopy(raw)
    for key in ("demand", "purchase_cost", "cv1", "cv2", "I_0", "I_1", "I_2", "V_i", "inventory_cost"):
        if key in instance and not isinstance(instance[key], np.ndarray):
            instance[key] = np.array(instance[key])
    return instance

# ---------------------------------------------------
# 1. 產生單一情境的實例
# ---------------------------------------------------

def generate_instance(levels: dict,
                      scenario: tuple[str, str, str],
                      num_instances: int,
                      seed: int | None = None) -> list[dict]:
    """回傳 [instance_dict, ...]"""
    size, ship_lvl, inv_lvl = scenario

    N = levels["scale"][size]["N"]
    T = levels["scale"][size]["T"]
    container_cost = levels["container_cost"][ship_lvl]
    inv_pct = levels["inventory_cost"][inv_lvl]

    def vrand(low, high, *shape):
        return rng.uniform(low, high, size=shape)

    inst_list = []
    for idx in range(num_instances):
        rng = np.random.default_rng(None if seed is None else seed + idx) 
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

# ---------------------------------------------------
# 2. 解法：Relax & MIP
# ---------------------------------------------------

def build_model(instance: Mapping, integer: bool) -> tuple[gp.Model]:

    t0 = time.time()

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

    rt = time.time() - t0
    orders = {
        (i, j, t): v.X
        for v in model.getVars() if v.VarName.startswith("x[") and v.X > 1e-6
        for i, j, t in [map(int, v.VarName[2:-1].split(","))]
    }
    return StrategySolution(
        name="Relax",
        orders=orders,
        total_cost=model.ObjVal,
        total_qty=sum(orders.values()),
        total_containers=sum(v.X for v in model.getVars() if v.VarName.startswith("z[")),
        run_time=rt
    )

def solve_all(instance: Mapping) -> Dict[str, StrategySolution]:
    sols = {}
    sols["Relax"] = build_model(instance, integer=False)
    sols["Heur"]  = heuristic_two_mode_jit(instance)
    sols["Naive"] = naive_solution(instance)
    return sols

def run_evaluation(
        benchmark_model: str = "Relax", 
        evaluation_models: list[str] = ["Heur"],
        output_prefix: str = "Evaluation",    
    ):
    records = []
    rng_seed = SEED
    for scenario in SCENARIOS:
        inst_list = generate_instance(LEVELS, scenario, NUM_INSTANCES, seed=rng_seed)
        for k, raw in enumerate(tqdm(inst_list, desc=str(scenario)), 1):
            instances = prep_instance(raw)
            sols = solve_all(instances)

            optimal = sols[benchmark_model].total_cost
            for tag in (evaluation_models + [benchmark_model]):
                gap = (sols[tag].total_cost - optimal) / optimal * 100
                records.append({
                    "Scenario": scenario, "Case": k, "Method": tag,
                    "Obj": sols[tag].total_cost,
                    "Gap(%)": gap,
                    "Time(s)": sols[tag].run_time
                })

    df = pd.DataFrame(records)
    out_dir = Path("report") / f"{output_prefix}_{time.strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/"Raw_evaluation_results.csv", index=False)

    summary = df.groupby(["Scenario", "Method"]).agg(
        ObjVal_Average=("Obj", "mean"),
        Gap_Average=("Gap(%)", "mean"),
        Gap_Std=("Gap(%)", "std"),
        Time_Average=("Time(s)", "mean"),
        Time_Std=("Time(s)", "std")
    ).reset_index()

    summary.to_excel(out_dir/"Evaluation_summary.xlsx", index=False)
    print("✔ Summary saved to", out_dir/"Evaluation_summary.xlsx")

# ---------------------------------------------------
# 6. 單一 BaseCase 報表
# ---------------------------------------------------
def run_strategies(case: str = "BaseCase") -> None:
    """
    case = "BaseCase"
        → 使用題目提供的 BASE_CASE 常數

    case = "<Scale>_<ContainerCost>_<InventoryCost>"
        → 例如 "Medium_Medium_Medium"
        → 會依 SCENARIOS 產生 *一筆* 隨機實例
    """
    case = case.strip()

    # ---------- 1) 取得 instance ----------
    if case.lower() == "basecase":
        inst = prep_instance(BASE_CASE)
        scenario_tag = "BaseCase"
    else:
        parts = tuple(case.split("_"))
        if parts not in SCENARIOS:
            raise ValueError(f"Invalid case '{case}'. Must be 'BaseCase' or one of {SCENARIOS}")

        # 只產生 1 筆隨機實例（seed 固定方便重現）
        inst = prep_instance(
            generate_instance(LEVELS, parts, num_instances=1, seed=SEED)[0]
        )
        
        scenario_tag = "_".join(parts)
        scenario_tag = scenario_tag.replace("Low", "L").replace("Medium", "M").replace("High", "H")
        scenario_tag = scenario_tag.replace("Small", "S").replace("Large", "L")
        scenario_tag = scenario_tag.replace("_", "-")
        
    # ---------- 2) 求解並輸出 ----------
    sols = solve_all(inst)
    export_strategies(
        list(sols.values()),
        instance=inst,
        scenario=scenario_tag 
    )
    print(f"✔ Finished strategies export for '{scenario_tag}'")

# ---------------------------------------------------
# 7.  CLI
# ---------------------------------------------------
if __name__ == "__main__":
    run_evaluation(benchmark_model="Relax", evaluation_models=["Heur", "Naive"])
    run_strategies(case="Medium_Medium_Medium")