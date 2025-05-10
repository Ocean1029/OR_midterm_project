import numpy as np
import random
from typing import Mapping, Dict, List

# Parameter levels for instance generation
LEVELS = {
    "scale": {
        "Small":  {"N": 10,  "T": 6},
        "Medium": {"N": 100, "T": 20},
        "Large":  {"N": 500, "T": 50},
    },
    "container_cost": {
        "Low": 1375,
        "Medium": 2750,
        "High": 5500,
    },
    "inventory_cost": {
        "Low": 0.01,
        "Medium": 0.02,
        "High": 0.04,
    },
}

def generate_instance(levels: Dict,
                     scenario: tuple[str, str, str],
                     num_instances: int,
                     seed: int | None = None) -> List[Dict]:
    """
    Generate test instances based on given parameters.
    
    Args:
        levels: Dictionary containing parameter levels
        scenario: Tuple of (size, shipping_level, inventory_level)
        num_instances: Number of instances to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of instance dictionaries
    """
    size, ship_lvl, inv_lvl = scenario

    N = levels["scale"][size]["N"]
    T = levels["scale"][size]["T"]
    container_cost = levels["container_cost"][ship_lvl]
    inv_pct = levels["inventory_cost"][inv_lvl]

    def vrand(low, high, *shape):
        return rng.uniform(low, high, size=shape)

    inst_list = []
    for idx in range(num_instances):
        rng = np.random.default_rng(None if seed is None else seed + idx) 
        demand = vrand(0, 200, N, T)
        purchase_cost = vrand(1_000, 10_000, N)
        cv1 = vrand(40, 100, N)
        alpha = vrand(0.4, 0.6, N)
        cv2 = alpha * cv1
        I_2 = np.where(rng.random(N) < .5, 0, vrand(0, 50, N))
        I_1 = np.where(rng.random(N) < .5, 0, vrand(0, 200, N))
        I_0 = rng.uniform(demand[:, 0], 400)
        V_i = vrand(0, 1, N)

        inst_list.append(dict(
            N=N, T=T, demand=demand, purchase_cost=purchase_cost,
            cv1=cv1, cv2=cv2, alpha=alpha,
            I_0=I_0, I_1=I_1, I_2=I_2, V_i=V_i,
            container_cost=container_cost,
            inventory_cost_percent=inv_pct,
        ))
    return inst_list 