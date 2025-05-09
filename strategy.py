# report_exporter.py  ── 完全重構版
from typing import List, Dict
import pandas as pd
from datetime import datetime
from pathlib import Path
from strategy_core import StrategySolution


def _pivot_orders(orders: Dict, N:int, T:int) -> Dict[int, pd.DataFrame]:
    """把 (i,j,t)->qty 轉成 {period : DataFrame(Product, Express, Air, Ocean)}"""
    prods = range(1, N+1)
    pivs: Dict[int,pd.DataFrame] = {}
    for t in range(T):
        mat = {p:{0:0,1:0,2:0} for p in prods}          # init 0
        for (i,j,tt), q in orders.items():
            if tt==t:
                mat[i+1][j] += q
        rows = [{"Product":p,
                 "Express":mat[p][0],
                 "Air":    mat[p][1],
                 "Ocean":  mat[p][2]} for p in prods]
        pivs[t] = pd.DataFrame(rows)
    return pivs

# ----------------------------------------------------------

def export_strategies(sols: List[StrategySolution],
                      instance: dict,
                      scenario: str) -> None:
    N, T = instance["N"], instance["T"]


    out_dir = Path("report") / f"Strategy_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =============== 個別 Excel =============================
    for sol in sols:
        piv = _pivot_orders(sol.orders, N, T)
        stats = {"TotalCost":sol.total_cost,
                 "TotalQty": sol.total_qty,
                 "TotalContainers": sol.total_containers,
                 "TotalRunTime": sol.run_time}
        fname = out_dir / f"{sol.name}.xlsx"
        with pd.ExcelWriter(fname, engine="xlsxwriter") as w:
            for t,df in piv.items():
                df.to_excel(w, sheet_name=f"Month{t+1}", index=False)
            pd.DataFrame({"KPI":stats.keys(),
                          "Value":[f'{v:,.2f}' for v in stats.values()]
                         }).to_excel(w, sheet_name="Year_Summary", index=False)
        print("✔ Exported", fname)

    # =============== KPI 彙總 ===============================
    rows = [dict(Method=sol.name,
                 TotalCost=sol.total_cost,
                 TotalQty=sol.total_qty,
                 TotalContainers=sol.total_containers,
                TotalRunTime=sol.run_time
                ) for sol in sols]
    kpi_path = out_dir / "KPI_Comparison.xlsx"
    pd.DataFrame(rows).to_excel(kpi_path, index=False)
    print("✔ Exported", kpi_path)