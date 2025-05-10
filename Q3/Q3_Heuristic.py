import numpy as np
from typing import Mapping, Dict, Tuple
from math import ceil
from Q4_strategy_core import StrategySolution
import time

FIX = {0: 100, 1: 80, 2: 50}   # 固定費 (Express / Air / Ocean)
LEAD = {0: 1, 1: 2, 2: 3}      # 前置期

# Base case instance for testing
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
# Heuristic Algorithm – Two-Mode + Service-Level Safe
# ---------------------------------------------------
def heuristic_two_mode_jit(instance: Mapping) -> StrategySolution:
    st = time.time()

    # ---------- unpack ----------------------------------------------------
    N, T = instance["N"], instance["T"]
    D    = instance["demand"]              # (N,T)
    pc   = instance["purchase_cost"]       # (N,)
    cv1  = instance["cv1"]
    cv2  = instance["cv2"]
    V    = instance["V_i"]
    CH   = instance.get(
        "inventory_cost",
        pc * instance.get("inventory_cost_percent", 0.02)
    )
    CC   = instance["container_cost"]
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]

    # ---------- S1. 每產品選擇 Air / Ocean -------------------------------
    c_sea = CC * V / 30          # (不含 50 固定) 單位海運攤提
    mode  = np.where(c_sea < cv2, 2, 1)   # 2:海運 1:空運

    # ---------- bookkeeping ----------------------------------------------
    inv = np.zeros((N, T + 1))   # 期末庫存，inv[:,0] = Mar end
    inv[:, 0] = I0
    orders: Dict[Tuple[int,int,int], float] = {}

    # helper
    def place(i: int, j: int, t: int, qty: float):
        if qty > 0:
            orders[(i, j, t)] = orders.get((i, j, t), 0.0) + qty

    # ---------- S2. per period loop --------------------------------------
    for i in range(N):
        for t in range(T):
            # 1) 期初可用庫存
            avail = inv[i, t]
            if t == 0:
                avail += I1[i]              # 到貨期數與題意一致
            elif t == 1:
                avail += I2[i]

            # 2) 目標：本期需求 + 下一期需求 (若存在)
            target = D[i, t] + (D[i, t + 1] if t < T - 1 else 0)
            shortage = target - avail
            if shortage <= 0:
                inv[i, t + 1] = avail - D[i, t]          # 仍保留≥下期需求
                continue

            # 3) 決定運輸方式 & 下單時間
            if t == 0:                                   # 第一月只能 Express
                j = 0
                order_t = t
            elif t == 1:                                 # 第二月可用 Air
                j = 1
                order_t = t - (LEAD[j] - 1)              # =0
            else:
                j = mode[i]
                order_t = t - (LEAD[j] - 1)

            place(i, j, order_t, shortage)
            inv[i, t + 1] = target - D[i, t]             # = 下一期需求

    # ---------- S3. 成本計算 ---------------------------------------------
    # (a) 變動採購 + 運費
    purchase_cost = sum(
        (pc[i] + (cv1[i] if j == 0 else cv2[i] if j == 1 else 0)) * q
        for (i, j, _), q in orders.items()
    )

    # (b) 固定費：每期每模式若有下單則計一次
    fix_cost = sum(
        FIX[j]
        for j, t in {(j, t) for (_, j, t) in orders}
    )

    # (c) 集裝箱成本：僅海運
    container_cost = 0.0
    for t in range(T):
        vol_t = sum(V[i] * orders.get((i, 2, t), 0.0) for i in range(N))
        k = int(np.ceil(vol_t / 30))
        container_cost += k * CC                         # 50 固定已算入 FIX[2]

    # (d) 持有成本：每期期末庫存 (t=0..T-1)
    holding_cost = (CH[:, None] * inv[:, 1:T + 1]).sum()


    total_cost = purchase_cost + fix_cost + container_cost + holding_cost

    # ---------- 其他 KPI --------------------------------------------------
    total_cont = sum(
        ceil(sum(V[i] * orders.get((i, 2, t), 0.0) for i in range(N)) / 30)
        for t in range(T)
    )

    return StrategySolution(
        name="Heur",
        orders=orders,
        total_cost=total_cost,
        total_qty=sum(orders.values()),
        total_containers=total_cont,
        run_time=time.time() - st,
    )

if __name__ == "__main__":
    # Convert lists to numpy arrays
    instance = BASE_CASE.copy()
    for key in ("demand", "purchase_cost", "cv1", "cv2", "I_0", "I_1", "I_2", "V_i", "inventory_cost"):
        if key in instance and not isinstance(instance[key], np.ndarray):
            instance[key] = np.array(instance[key])
    
    # Run heuristic algorithm
    solution = heuristic_two_mode_jit(instance)
    
    # Print results
    print("\n=== Heuristic Algorithm Results ===")
    print(f"Total Cost: {solution.total_cost:,.2f}")
    print(f"Total Order Quantity: {solution.total_qty:,.2f}")
    print(f"Total Containers: {solution.total_containers}")
    print(f"Runtime: {solution.run_time:.3f} seconds")
    
    # Print order details
    print("\nOrder Details:")
    for (i, j, t), qty in sorted(solution.orders.items()):
        mode = "Express" if j == 0 else "Air" if j == 1 else "Ocean"
        print(f"Product {i+1}, Mode {mode}, Period {t+1}: {qty:,.2f}")

