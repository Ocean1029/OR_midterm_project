# report_exporter.py
"""Generate strategy Excel reports (Relaxation, Heuristic, MIP) for a single instance.

Usage in Q4.py
---------------
from report_exporter import generate_strategy_reports

# inst 已經是 prep_instance() 後的字典
# build_model 與 naive_heuristic 是你現有的函式

generate_strategy_reports(
    instance=inst,
    build_model=build_model,
    naive_heuristic=naive_heuristic,
    out_prefix="base_case"  # 生成 base_case_*.xlsx
)
"""

from __future__ import annotations
from typing import Callable, Dict
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from heuristic import heuristic_two_mode_jit

# --------------------------------------------------
# 工具函式
# --------------------------------------------------

def _extract_orders(model: gp.Model, method_map: Dict[int, str]) -> pd.DataFrame:
    """Return long‑format DataFrame: Period, Product, Method, Qty (x > 0)."""
    rows = []
    for v in model.getVars():
        if not v.VarName.startswith("x[") or abs(v.X) <= 1e-6:
            continue
        i, j, t = map(int, v.VarName[2:-1].split(","))  # x[i,j,t]
        rows.append([t, i + 1, method_map[j], v.X])  # product index 1‑based
    return pd.DataFrame(rows, columns=["Period", "Product", "Method", "Qty"])

def _pivot_full(df_long: pd.DataFrame, n_prod: int, n_period: int) -> Dict[int, pd.DataFrame]:
    """Return {period: DataFrame} with **all** products & methods.
    Any missing combo is filled 0, then cast to int safely."""
    prods = list(range(1, n_prod + 1))
    full: Dict[int, pd.DataFrame] = {}

    # base empty frame to guarantee all cols exist
    base_cols = ["Product", "Express", "Air", "Ocean"]
    empty_tmpl = pd.DataFrame({c: [] for c in base_cols})

    for t in range(n_period):
        sub = df_long[df_long["Period"] == t]
        piv = (
            sub.pivot(index="Product", columns="Method", values="Qty")
               .reindex(index=prods, columns=["Express", "Air", "Ocean"], fill_value=0)
               .reset_index()
        )
        # ensure all columns exist even if completely missing
        piv = pd.concat([empty_tmpl, piv], ignore_index=True).fillna(0).groupby("Product", as_index=False).sum()
        full[t] = piv.astype({"Express": int, "Air": int, "Ocean": int})
    return full

def _export_strategy_excel(pivots: Dict[int, pd.DataFrame],
                           fname: str,
                           year_stats: Dict[str, float]) -> None:
    """Write each period sheet + Year_Summary."""
    with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
        for t, df in pivots.items():
            sheet = f"Month{t + 1}"
            df.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame({"KPI": list(year_stats.keys()),
                      "Value": [f"{v:,.2f}" for v in year_stats.values()]}) \
            .to_excel(writer, sheet_name="Year_Summary", index=False)
    print(f"✔ Exported {fname}")

def _export_kpi_comparison(kpi_dict: Dict[str, Dict[str, float]], fname: str) -> None:
    rows = [{"Method": m, **stats} for m, stats in kpi_dict.items()]
    pd.DataFrame(rows).to_excel(fname, index=False)
    print(f"✔ Exported {fname}")

# --------------------------------------------------
# 主函式
# --------------------------------------------------

def generate_strategy_reports(
    instance: dict,
    build_model: Callable[[dict, bool], gp.Model],
    naive_heuristic: Callable[[dict], float],
    out_prefix: str = "strategy"
) -> None:
    """Generate Relaxation / Heuristic / MIP strategy Excel files + KPI comparison."""
    n_prod, n_period = instance["N"], instance["T"]
    method_map = {0: "Express", 1: "Air", 2: "Ocean"}

    # ---------------- MIP ----------------
    mip_model = build_model(instance, integer=True)
    if mip_model.status != GRB.OPTIMAL:
        raise RuntimeError("MIP did not reach optimality")
    mip_long = _extract_orders(mip_model, method_map)
    mip_pivots = _pivot_full(mip_long, n_prod, n_period)
    mip_stats = {
        "TotalCost": mip_model.ObjVal,
        "TotalQty": mip_long["Qty"].sum(),
        "TotalContainers": sum(v.X for v in mip_model.getVars() if v.VarName.startswith("z["))
    }
    _export_strategy_excel(mip_pivots, f"{out_prefix}_MIP.xlsx", mip_stats)

    # ---------------- Relaxation ----------------
    relax_model = build_model(instance, integer=False)
    if relax_model.status != GRB.OPTIMAL:
        raise RuntimeError("Relaxation did not reach optimality")
    relax_long = _extract_orders(relax_model, method_map)
    relax_pivots = _pivot_full(relax_long, n_prod, n_period)
    relax_stats = {
        "TotalCost": relax_model.ObjVal,
        "TotalQty": relax_long["Qty"].sum(),
        "TotalContainers": sum(v.X for v in relax_model.getVars() if v.VarName.startswith("z["))
    }
    _export_strategy_excel(relax_pivots, f"{out_prefix}_Relaxation.xlsx", relax_stats)

        # ---------------- Heuristic ----------------

    orders_h, heur_obj = heuristic_two_mode_jit(instance)

    # 轉成 long-format DataFrame (Period, Product, Method, Qty)
    rows = [
        [t, i + 1, method_map[j], q]
        for (i, j, t), q in orders_h.items()
        if abs(q) > 1e-6
    ]
    heur_long = pd.DataFrame(rows, columns=["Period", "Product", "Method", "Qty"])

    heur_pivots = _pivot_full(heur_long, n_prod, n_period)

    # 計算總箱數：每期海運體積 ÷ 30 CBM 向上取整
    V_i = instance["V_i"]
    total_cont = 0
    for t in range(n_period):
        vol_t = sum(V_i[i] * orders_h.get((i, 2, t), 0) for i in range(n_prod))
        if vol_t > 0:
            total_cont += int(np.ceil(vol_t / 30))

    heur_stats = {
        "TotalCost": heur_obj,
        "TotalQty": heur_long["Qty"].sum(),
        "TotalContainers": total_cont,
    }

    _export_strategy_excel(heur_pivots, f"{out_prefix}_Heuristic.xlsx", heur_stats)

    # ---------------- KPI Comparison ----------------
    kpi_all = {"MIP": mip_stats, "Relaxation": relax_stats, "Heuristic": heur_stats}
    _export_kpi_comparison(kpi_all, f"{out_prefix}_KPI_Comparison.xlsx")