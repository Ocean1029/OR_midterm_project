# naive.py  或直接放同一檔
from strategy_core import StrategySolution
import time
from numpy import ceil

def naive_solution(instance) -> StrategySolution:
    t0 = time.time()
    orders = {}  # (i,j,t) -> qty

    # ---- unpack instance --------------------------------------------------
    
    N, T = instance["N"], instance["T"]
    demand = instance["demand"]
    purchase_cost = instance["purchase_cost"]
    cv1 = instance["cv1"]
    I_0, I_1, I_2 = instance["I_0"], instance["I_1"], instance["I_2"]

    total_cost = 0.0
    for i in range(N):
        for t in range(T):
            req = demand[i, t]
            if t == 0:
                req -= I_0[i] + I_1[i]
            elif t == 1:
                req -= I_2[i]
            if req > 0:
                orders[(i, 2, t)] = req
                total_cost += purchase_cost[i] * req + cv1[i] * req
    total_cost += 100 * T

    rt  = time.time() - t0

    return StrategySolution(
        name="Naive",
        orders=orders,
        total_cost=total_cost,
        total_qty=sum(orders.values()),
        total_containers=0,
        run_time=rt
    )
