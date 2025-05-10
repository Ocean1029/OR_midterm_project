"""
Do not execute this file directly!
Instead, run the Q5_main.py file, which will import this file and execute algorithm here on the generated result.
"""

# naive.py
from Q4_strategy_core import StrategySolution
import time
import numpy as np

FIX  = {0: 100}   # Express fixed cost (j=0)
LEAD = 1          # Express lead time

def naive_solution(instance) -> StrategySolution:
    """
    Naive Solution:
    - Only uses Express shipping
    - Ensures ending inventory ≥ next period demand (satisfies constraint 4)
    """
    start = time.time()
    orders: dict[tuple[int, int, int], float] = {}

    # === unpack ==========================================================
    N, T      = instance["N"], instance["T"]
    D         = instance["demand"]               # (N,T)
    pc        = instance["purchase_cost"]
    cv1       = instance["cv1"]                  # Express variable cost
    I0, I1, I2 = instance["I_0"], instance["I_1"], instance["I_2"]
    inv_cost  = instance.get(
        "inventory_cost",
        pc * instance.get("inventory_cost_percent", 0.02)
    ) # If inventory_cost is not given, use purchase_cost * default percentage

    # Initial inventory
    ending_inv = np.array(I0, dtype=float)

    total_cost = 0.0
    for t in range(T):
        # 1) Current period arrivals (I1 / I2 only effective in first two periods)
        arrivals = I1 if t == 0 else I2 if t == 1 else np.zeros(N)
        ending_inv += arrivals

        # 2) First ensure current period shipment
        shortage_now = D[:, t] - ending_inv
        shortage_now = np.maximum(shortage_now, 0)

        # 3) Then ensure ending inventory ≥ next period demand
        next_demand  = D[:, t + 1] if t < T - 1 else np.zeros(N)
        after_ship   = ending_inv - D[:, t] + shortage_now  # replenish before shipping
        shortage_next = np.maximum(next_demand - after_ship, 0)

        # 4) Combine both shortages for one order
        total_shortage = shortage_now + shortage_next
        if total_shortage.sum() > 0:          # Only count fixed cost if there's an order
            total_cost += FIX[0]

        for i, q in enumerate(total_shortage):
            if q > 0:
                orders[(i, 0, t)] = q
                total_cost += (pc[i] + cv1[i]) * q
                ending_inv[i] += q            # replenishment arrives immediately (Lead=1)

        # 5) Deduct current period shipment, update ending inventory
        ending_inv -= D[:, t]

        # 6) Ending inventory holding cost
        total_cost += (inv_cost * ending_inv).sum()

    runtime = time.time() - start

    return StrategySolution(
        name="Naive",
        orders=orders,
        total_cost=total_cost,
        total_qty=sum(orders.values()),
        total_containers=0,        # Express doesn't use containers
        run_time=runtime
    )