import pandas as pd
import time
from gurobipy import GRB


def generate_experiment_output(instances, solve_linear_relaxation, naive_heuristic, output_excel=True):
    
    rows = []
    scenario_id_map = {sc: idx + 1 for idx, sc in enumerate(instances.keys())}

    for scenario, inst_list in instances.items():
        sid = scenario_id_map[scenario]
        for idx, data in enumerate(inst_list):
            instance_id = idx + 1

            # --- LP Relaxation ---
            start_lp = time.perf_counter()
            model = solve_linear_relaxation(data)
            end_lp = time.perf_counter()
            lp_obj = model.objVal if model.status == GRB.OPTIMAL else None
            lp_time = end_lp - start_lp

            # --- Proposed Algorithm (Blank now) ---
            proposed_obj = ""
            proposed_time = ""

            # --- Very Naive ---
            start_nv = time.perf_counter()
            naive_obj = naive_heuristic(data)
            end_nv = time.perf_counter()
            naive_time = end_nv - start_nv

            # --- Gaps ---
            gap_prop = ""
            gap_nv = ((naive_obj - lp_obj) / lp_obj * 100) if lp_obj is not None else ""

            # Add row
            rows.append([
                sid, instance_id,
                lp_obj, proposed_obj, naive_obj,
                lp_time, proposed_time, naive_time,
                gap_prop, gap_nv
            ])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=[
        "ScenarioID", "InstanceID",
        "LP_Obj", "Proposed_Obj", "Naive_Obj",
        "LP_Time", "Proposed_Time", "Naive_Time",
        "Gap_LP_Proposed", "Gap_LP_Naive"
    ])

    # Output to Excel
    df.to_excel("experiment_results.xlsx", index=False)    
    return df
