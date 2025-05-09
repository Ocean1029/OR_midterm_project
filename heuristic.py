import numpy as np
from typing import Mapping, Tuple, Dict
from math import ceil
from strategy_core import StrategySolution
import time

# ---------------------------------------------------
# Heuristic Algorithm for Problem 3 – Two‑Mode JIT
# ---------------------------------------------------

def heuristic_two_mode_jit(instance: Mapping) -> Tuple[Dict, float]:
    """
    Return:
        orders : dict[(i,j,t)] -> Q_ijt  第 i 個產品在 t 期用第 j 種運輸方式的訂單量
        obj    : float, total cost of the heuristic solution
    """
    # ---- unpack instance --------------------------------------------------
    N, T = instance["N"], instance["T"]
    D   = instance["demand"]          # shape (N,T)
    pc  = instance["purchase_cost"]   # length N
    cv1 = instance["cv1"]             # express
    cv2 = instance["cv2"]             # air
    V   = instance["V_i"]             # volume
    CH = instance.get("inventory_cost", pc * instance.get("inventory_cost_percent", 0.02)) # 如果沒有給定 inventory_cost，就用 purchase_cost 乘上預設的百分比
    CC  = instance["container_cost"]  # container cost
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]

    # ---- S1. compute unit sea cost ----------------------------------------
    c_sea = CC * V / 30.0             # 每單位攤提的箱子成本 (忽略 $50 固定)
    c_air = cv2

    # ---- S2. decide mode per item -----------------------------------------
    mode = np.where(c_sea < c_air, 2, 1)   # 2: sea, 1: air
    # lead time dict
    LEAD = {0: 1, 1: 2, 2: 3}

    # ---- bookkeeping arrays -----------------------------------------------
    inv   = np.zeros((N, T+1))        # ending inventory (t=0 is March end)
    inv[:,0] = I0                     # initial inventory
    orders = {}                       # Q_ijt

    # helper: place order
    def place(i: int, j: int, t: int, qty: float):
        if qty <= 0: return
        orders[(i,j,t)] = orders.get((i,j,t), 0.0) + qty

    # ---- S3. loop over periods  -------------------------
    t0 = time.time()
    for i in range(N):
        for t in range(T):
            # arrival from previous orders / in‑transit
            if t == 0:
                inv_prev = inv[i,0] + I1[i]  
            elif t == 1:
                inv_prev = inv[i,1] + I2[i] 
            else:
                inv_prev = inv[i,t]

            need = D[i,t] - inv_prev
            if need <= 0:
                inv[i,t+1] = inv_prev - D[i,t]
                continue

            # decide shipping mode & order timing
            if t == 0:              # month 1 ➜ express
                j = 0
                order_timing = t             # order now
            elif t == 1:            # month 2 ➜ air
                j = 1
                order_timing = t-1           # lead 2, so order at t-1 (March)
            else:
                j = mode[i]         # 1 or 2
                order_timing = t - (LEAD[j]-1)

            place(i, j, order_timing, need)
            inv[i,t+1] = 0.0        # JIT, no leftover

    # ---- S4. simple sea container adjustment ------------------------------
    for t in range(T):
        vol_t = sum(V[i] * orders.get((i,2,t), 0) for i in range(N))
        if vol_t == 0:
            continue
        used = vol_t / 30.0
        frac = used - np.floor(used)
        if 0 < frac < 0.5 and t < T-1:
            # 嘗試把 t+1 期部分海運貨提前
            compressable = []
            for i in range(N):
                if mode[i] == 2:
                    nxt_qty = orders.get((i,2,t+1), 0)
                    if nxt_qty > 0:
                        compressable.append((i, nxt_qty, V[i]))
            # sort by holding cost impact ascending
            compressable.sort(key=lambda x: CH[x[0]])
            vol_gap = (1-frac)*30
            moved_units = 0
            for i, qty, vol in compressable:
                max_move = min(qty, vol_gap/vol)
                if max_move <= 0: continue
                # holding cost to move 1 unit one month: CH[i]
                if CH[i] * max_move < CC * (1-frac):
                    # move!
                    place(i,2,t,   max_move)
                    place(i,2,t+1,-max_move)
                    vol_gap -= max_move * vol
                    moved_units += max_move
                if vol_gap <= 0: break
            # 重新計算 vol_t (可省略，這只是成本評估啟發)

    # ---- S5. compute objective --------------------------------------------
    # 購買 + 運費
    purchase_cost = sum(
        (pc[i] + (cv1[i] if j == 0 else cv2[i] if j == 1 else 0)) * q
        for (i,j,t), q in orders.items()
    )
    # 固定運費
    FIX = {0:100, 1:80, 2:50}
    fix_cost = sum(
        FIX[j] for (j,t) in {(j,t) for (_,j,t) in orders if j in (0,1,2)}
    )
    # 集裝箱成本
    container_cost = 0.0
    for t in range(T):
        vol = sum(V[i] * orders.get((i,2,t),0) for i in range(N))
        k   = int(np.ceil(vol/30))
        if k>0:
            container_cost += k * CC + 50   # 50 = CF_3 (ocean fixed)

    # 持有成本（僅月 0~T-1 庫存）
    holding_cost = sum(CH[i] * inv[i,t] for i in range(N) for t in range(1,T+1))

    obj = purchase_cost + fix_cost + container_cost + holding_cost

    cont = sum(
        ceil(sum(V[i] * orders.get((i, 2, t), 0) for i in range(N)) / 30)
        for t in range(T)
    )

    rt = time.time() - t0
    return StrategySolution(
        name="Heur",
        orders=orders,
        total_cost=obj,
        total_qty=sum(orders.values()),
        total_containers=cont,
        run_time=rt
    )
    
