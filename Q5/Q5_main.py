import sys
import os
import time
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "Q3"))
sys.path.append(os.path.join(BASE_DIR, "..", "Q4"))

from Q4_LP_relax import lp_relaxation
from Q4_Naive import naive_solution
from Q3_Heuristic import heuristic_two_mode_jit
from Q4_strategy_core import StrategySolution
from Q4_generate_experiment import generate_experiment, LEVELS



# ---------------------------------------------------
# Basic Parameters
# ---------------------------------------------------

NUM_INSTANCES = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

SCENARIOS: List[tuple[str, str, str]] = [
    ("Medium", "Medium", "Medium"),
    ("Small",  "Medium", "Medium"),
    ("Large",  "Medium", "Medium"),
    ("Medium", "Low",    "Medium"),
    ("Medium", "High",   "Medium"),
    ("Medium", "Medium", "Low"),
    ("Medium", "Medium", "High"),
]

def run_evaluation(
    evaluation_models: List[str],
    benchmark_model: str = "LP_Relax",
    output_prefix: str = "Evaluation"
) -> None:
    """
    Run evaluation on multiple scenarios and models.
    
    Args:
        evaluation_models: List of model names to evaluate
        benchmark_model: Model to use as benchmark for gap calculation
        output_prefix: Prefix for output files
    """
    records = []
    
    for scenario in SCENARIOS:
        print(f"\nProcessing scenario: {scenario}")
        
        # Generate instances
        instances = generate_experiment(
            LEVELS,
            scenario,
            num_instances=NUM_INSTANCES,
            seed=SEED
        )
        
        # Run each model
        for k, instance in enumerate(tqdm(instances, desc=str(scenario)), 1):
            instance_results = {}
            
            # Run benchmark model first
            if benchmark_model == "LP_Relax":
                instance_results[benchmark_model] = lp_relaxation(instance)
            else:
                raise ValueError(f"Unknown benchmark model: {benchmark_model}")
            
            # Run other models
            for model_name in evaluation_models:
                if model_name == "Heuristic":
                    instance_results[model_name] = heuristic_two_mode_jit(instance)
                elif model_name == "Naive":
                    instance_results[model_name] = naive_solution(instance)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            
            # Calculate gaps and record results
            optimal = instance_results[benchmark_model].total_cost
            record = {
                "Scenario": scenario,
                "Instance_ID": k,
                "LP_Relax_Obj (Q4)": instance_results["LP_Relax"].total_cost,
                "Proposed_Heuristic_Obj (Q5)": instance_results["Heuristic"].total_cost,
                "Naive_Obj (Q4)": instance_results["Naive"].total_cost,
                "LP_Relax_Time": instance_results["LP_Relax"].run_time,
                "Proposed_Heuristic_Time": instance_results["Heuristic"].run_time,
                "Naive_Time": instance_results["Naive"].run_time,
            }
            
            # Calculate optimality gaps
            record["Proposed_Heuristic_Gap (Q5)"] = (instance_results["Heuristic"].total_cost - optimal) / optimal * 100
            record["Naive_Gap (Q4)"] = (instance_results["Naive"].total_cost - optimal) / optimal * 100
            
            records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save raw results to root directory
    df.to_csv("Raw_evaluation_results.csv", index=False)
    print("✔ Raw results saved to Raw_evaluation_results.csv")
    
    # Calculate and save summary
    summary = df.groupby("Scenario").agg(
        LP_Relax_Obj_Avg=("LP_Relax_Obj (Q4)", "mean"),
        Heuristic_Obj_Avg=("Proposed_Heuristic_Obj (Q5)", "mean"),
        Naive_Obj_Avg=("Naive_Obj (Q4)", "mean"),
        LP_Relax_Time_Avg=("LP_Relax_Time", "mean"),
        Heuristic_Time_Avg=("Proposed_Heuristic_Time", "mean"),
        Naive_Time_Avg=("Naive_Time", "mean"),
        Heuristic_Gap_Avg=("Proposed_Heuristic_Gap (Q5)", "mean"),
        Heuristic_Gap_Std=("Proposed_Heuristic_Gap (Q5)", "std"),
        Naive_Gap_Avg=("Naive_Gap (Q4)", "mean"),
        Naive_Gap_Std=("Naive_Gap (Q4)", "std")
    ).reset_index()
    
    # Save summary to Q5 directory
    summary.to_excel(os.path.join(BASE_DIR, "Evaluation_summary.xlsx"), index=False)
    print("✔ Summary saved to Q5/Evaluation_summary.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval"], default="eval")
    args = parser.parse_args()

    if args.mode == "eval":
        run_evaluation(
            evaluation_models=["Heuristic", "Naive"],
            benchmark_model="LP_Relax",
            output_prefix="Evaluation"
        )