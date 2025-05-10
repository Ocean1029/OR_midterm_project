import time
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from Q4_naive import naive_solution
from Q3_heuristic import heuristic_two_mode_jit
from Q4_strategy_core import StrategySolution
from Q4_generate_instance import generate_instance, LEVELS
from Q4_LP_relax import lp_relaxation

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
        instances = generate_instance(
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
            for model_name in [benchmark_model] + evaluation_models:
                gap = (instance_results[model_name].total_cost - optimal) / optimal * 100
                records.append({
                    "Scenario": scenario,
                    "Case": k,
                    "Method": model_name,
                    "Obj": instance_results[model_name].total_cost,
                    "Gap(%)": gap,
                    "Time(s)": instance_results[model_name].run_time
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save results
    out_dir = Path("results") / f"{output_prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    df.to_csv(out_dir / "Raw_evaluation_results.csv", index=False)
    
    # Calculate and save summary
    summary = df.groupby(["Scenario", "Method"]).agg(
        ObjVal_Average=("Obj", "mean"),
        Gap_Average=("Gap(%)", "mean"),
        Gap_Std=("Gap(%)", "std"),
        Time_Average=("Time(s)", "mean"),
        Time_Std=("Time(s)", "std")
    ).reset_index()
    
    summary.to_excel(out_dir / "Evaluation_summary.xlsx", index=False)
    print("✔ Summary saved to", out_dir / "Evaluation_summary.xlsx")

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