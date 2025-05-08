from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class StrategySolution:
    name: str                               # 例如 "MIP" / "Relax" / "Heuristic-JIT"
    orders: Dict[Tuple[int,int,int], float] # (i,j,t) ➜ qty
    total_cost: float
    total_qty: float
    total_containers: float
    run_time: float = 0.0                       # 方便記錄求解時間