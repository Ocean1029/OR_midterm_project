# naive.py
from strategy_core import StrategySolution
import time
import numpy as np

FIX = {0: 100}          # 只用 Express，固定費 100   (題目 j=0)
LEAD = 1                # Express 前置期 1

def naive_solution(instance) -> StrategySolution:
    """
    最單純：每期先吃掉期初 + 在途庫存；不足就用 Express 當期下單。
    完全不考慮提前備貨、也不碰空/海運 → 不會出現過量訂單。
    """
    start = time.time()
    orders = {}                       # (i,j,t) → qty

    # ---- unpack ----------------------------------------------------------
    N, T = instance["N"], instance["T"]
    D    = instance["demand"]
    pc   = instance["purchase_cost"]
    cv1  = instance["cv1"]            # Express 變動費
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]
    inv_cost = instance.get("inventory_cost", pc * instance.get("inventory_cost_percent", 0.02)) # 如果沒有給定 inventory_cost，就用 purchase_cost 乘上預設的百分比
    
    # 期末庫存
    inv = np.array(I0, dtype=float)

    total_cost = 0.0
    for t in range(T):
        # 本期可用存貨：庫存 + 剛好到貨
        arrivals = I1 if t == 0 else I2 if t == 1 else np.zeros(N)
        avail = inv + arrivals

        shortage = D[:, t] - avail
        shortage = np.maximum(shortage, 0)      # clip 負值

        # 當期用 Express 補足
        for i, q in enumerate(shortage):
            if q > 0:
                orders[(i, 0, t)] = q
                total_cost += (pc[i] + cv1[i]) * q
                inv[i] = 0
            else:
                inv[i] -= D[i, t]

        # 固定費：只算一次／期 有下單才計
        if shortage.sum() > 0:
            total_cost += FIX[0]

        # hodling cost for each product
        for i in range(N):
            if inv[i] > 0:
                total_cost += inv_cost[i] * inv[i]

    runtime = time.time() - start

    return StrategySolution(
        name="Naive",
        orders=orders,
        total_cost=total_cost,
        total_qty=sum(orders.values()),
        total_containers=0,          # 全用 Express，無集裝箱
        run_time=runtime
    )
