# naive.py
from Q4_strategy_core import StrategySolution
import time
import numpy as np

FIX  = {0: 100}   # Express 固定費 (j=0)
LEAD = 1          # Express 前置期

def naive_solution(instance) -> StrategySolution:
    """
    改良版 Naive：
    - 只用 Express 補貨
    - 確保期末庫存 ≥ 下一期需求 (滿足題目式 4)
    """
    start = time.time()
    orders: dict[tuple[int, int, int], float] = {}

    # === unpack ==========================================================
    N, T      = instance["N"], instance["T"]
    D         = instance["demand"]               # (N,T)
    pc        = instance["purchase_cost"]
    cv1       = instance["cv1"]                  # Express 變動費
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]
    inv_cost  = instance.get(
        "inventory_cost",
        pc * instance.get("inventory_cost_percent", 0.02)
    ) # 如果沒有給定 inventory_cost，就用 purchase_cost 乘上預設的百分比

    # 期初庫存
    ending_inv = np.array(I0, dtype=float)

    total_cost = 0.0
    for t in range(T):
        # 1) 本期到貨 (I1 / I2 只在前兩期生效)
        arrivals = I1 if t == 0 else I2 if t == 1 else np.zeros(N)
        ending_inv += arrivals

        # 2) 先確保本期出貨
        shortage_now = D[:, t] - ending_inv
        shortage_now = np.maximum(shortage_now, 0)

        # 3) 再確保期末庫存 ≥ 下期需求
        next_demand  = D[:, t + 1] if t < T - 1 else np.zeros(N)
        after_ship   = ending_inv - D[:, t] + shortage_now  # ship 前先補貨
        shortage_next = np.maximum(next_demand - after_ship, 0)

        # 4) 合併兩種缺口一次下單
        total_shortage = shortage_now + shortage_next
        if total_shortage.sum() > 0:          # 本期有下單才計固定費
            total_cost += FIX[0]

        for i, q in enumerate(total_shortage):
            if q > 0:
                orders[(i, 0, t)] = q
                total_cost += (pc[i] + cv1[i]) * q
                ending_inv[i] += q            # 補貨立即到達 (Lead=1)

        # 5) 扣除本期出貨，更新期末庫存
        ending_inv -= D[:, t]

        # 6) 期末持有成本
        total_cost += (inv_cost * ending_inv).sum()

    runtime = time.time() - start

    return StrategySolution(
        name="Naive",
        orders=orders,
        total_cost=total_cost,
        total_qty=sum(orders.values()),
        total_containers=0,        # Express 不用箱子
        run_time=runtime
    )
