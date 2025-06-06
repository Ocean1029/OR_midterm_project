import pandas as pd
import numpy as np
import os

######################## PARAMTERS STARTING HERE ################################
# Read the Excel file from the 'Demand' sheet
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "OR113-2_midtermProject_data.xlsx")
df_demand = pd.read_excel(file_path, sheet_name="Demand")
N = df_demand.shape[0] - 1   # -1 because of the first row, +1 for indices' consistency
T = df_demand.shape[1] - 2  # -2 because of the first two columns, +1 for indices' consistency  
print("N:", N, "T:", T)

# Display the dataframe to verify the data
I = np.zeros([N, T])
D = np.zeros([N, T])
I_0 = np.zeros([N])

for i in range(N):
    I_0[i] = df_demand.iloc[i+1, 1]
    for t in range(T):
        D[i, t] = df_demand.iloc[i+1, t+2]

print("I_0:", I_0)
print("D:", D)

# Read the Excel file from the 'In-transit' sheet
df_in_transit = pd.read_excel(file_path, sheet_name="In-transit")
for i in range(N):
    for t in range(df_in_transit.shape[1] - 1):
        I[i, t] = df_in_transit.iloc[i+1, t+1]
print("I:", I)

# Read the Excel file from the 'Shipping cost' sheet
df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
J = df_shipping_cost.shape[1] - 1 # -1 because of the first column
df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")


C = {
    "H": np.zeros([N]),
    "P": np.zeros([N]),
    "V": np.zeros([N, J]),
    "F": np.array([100, 80, 50]),
    "C": 2750,
    "L": np.zeros([N]), # Shortage cost
    "B": np.zeros([N]), # Backorder cost
}
V = np.zeros([N])
V_C = 30
for i in range(N):
    C["H"][i] = df_inventory_cost.iloc[i, 3]
    C["P"][i] = df_inventory_cost.iloc[i, 2]
    V[i] = df_shipping_cost.iloc[i, 3]
    for j in range(J):
        if j == J - 1:
            C["V"][i, j] = 0
        else:
            C["V"][i, j] = df_shipping_cost.iloc[i, j+1]
    C["L"][i] = df_inventory_cost.iloc[i, 1] - C["P"][i]  # Shortage cost = Sales price - purchase cost
    C["B"][i] = df_inventory_cost.iloc[i, 4]  # Backorder cost

print("C:", C)
print("V:", V)
T_lead = np.array([1, 2, 3]) # T_j

######################## PARAMTERS ENDING HERE ##################################

import gurobipy as gp
from gurobipy import GRB

# Provided parameters (already read from the Excel file)
# N: number of products, T: number of time periods, J: number of shipping methods
# D: demand, I: in-transit inventory, C: cost parameters, V: volume, T_lead: lead times, V_C: container volume

# Create the Gurobi model
model = gp.Model("InventoryManagement")

# Set error parameter
model.setParam('MIPGap', 0.0)

# Define sets
S_I = range(N)  # Products i in {0,  ..., N-1}
S_T = range(T)  # Time periods t in {0, ..., T-1}
S_J = range(J)  # Shipping methods j in {0, ..., J-1}

# Variables
x = model.addVars(S_I, S_J, S_T, vtype=GRB.CONTINUOUS, name="x")  # Order quantity x_ijt
v = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="v")  # Ending inventory v_it
y = model.addVars(S_J, S_T, vtype=GRB.BINARY, name="y")  # Binary for shipping method y_jt
z = model.addVars(S_T, vtype=GRB.INTEGER, name="z")  # Number of containers z_t
l = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="l")  # Lost sales l_it
b = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="b")  # Backorders b_it
u = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="u")  # total unfulfilled demand u_it
f = model.addVars(S_I, S_T, vtype=GRB.CONTINUOUS, name="f")  # fulfilled backorder f_it


# Objective function (1)
# Holding cost + (Purchasing cost + Variable shipping cost + Fixed shipping cost) + Container cost + Shortage cost + Backorder cost
holding_cost = gp.quicksum(C["H"][i] * v[i, t] for i in S_I for t in S_T)
purchasing_and_shipping_cost = gp.quicksum(
    (C["P"][i] + C["V"][i, j]) * x[i, j, t]
    for i in S_I for j in S_J for t in S_T
) + gp.quicksum(C["F"][j] * y[j, t] for t in S_T for j in S_J)
container_cost = gp.quicksum(C["C"] * z[t] for t in S_T)
shortage_cost = gp.quicksum(C["L"][i] * l[i, t] for i in S_I for t in S_T)
backorder_cost = gp.quicksum(C["B"][i] * (b[i, t] - f[i, t]) for i in S_I for t in S_T)

model.setObjective(holding_cost + purchasing_and_shipping_cost + container_cost + shortage_cost + backorder_cost, GRB.MINIMIZE)

# Constraints
# Inventory balance (2)(3)
J_in_inventory = np.array([1, 2, 3, 3, 3, 3])

for i in S_I:
    unfulfilled_backorder = 0
    for t in S_T:
        # Compute the in-transit quantity arriving at time t
        in_inventory = 0
        for j in range(J_in_inventory[t]):
            in_inventory += x[i, j, t - T_lead[j] + 1]
        unfulfilled_backorder += (b[i, t-1] - f[i, t-1]) if t > 0 else 0
        # Add the constraint for inventory balance
        if t == 0:
            model.addConstr(v[i, t] == in_inventory + I_0[i] + I[i, t] - D[i, t] + u[i, t], name=f"InvBalance_{i}_{t}")
        else:
            model.addConstr(v[i, t] == v[i, t-1] + in_inventory + I[i, t] - D[i, t] + u[i, t] + unfulfilled_backorder, name=f"InvBalance_{i}_{t}")
            model.addConstr(v[i, t-1] >= D[i, t] - u[i, t], name=f"Demand_{i}_{t}")


# Relate order quantity and shipping method (4)
M = sum(sum(D[i, t] for t in S_T) for i in S_I)  # Large number M as per problem statement
for j in S_J:
    for t in S_T:
        model.addConstr(gp.quicksum(x[i, j, t] for i in S_I) <= M * y[j, t], name=f"ShippingMethod_{j}_{t}")

# Container constraint (5)
for t in S_T:
    model.addConstr(
        gp.quicksum(V[i] * x[i, 2, t] for i in S_I) <= V_C * z[t],
        name=f"Container_{t}"
    )

# Relate lost sales, backorder and demand (7)
for i in S_I:
    sum_backorder = 0
    for t in S_T:
        sum_backorder += b[i, t-1] if t > 0 else 0
        r = df_inventory_cost.iloc[i, 5]
        if t == 0:
            model.addConstr(u[i, t] >= D[i, t] - I_0[i] - I[i, t], name=f"TotalUnfulfill_{i}_{t}")
            # 考慮 原有的存貨 I_0[i] 跟 在途存貨 I[i, t]
        else:
            model.addConstr(u[i, t] >= D[i, t] - v[i, t-1], name=f"TotalUnfulfill_{i}_{t}")
            model.addConstr(f[i, t] <= sum_backorder, name=f"FulfilledBackorder_{i}_{t}")

        model.addConstr(b[i, t] == u[i, t] * (r / (1 + r)), name=f"BackorderLinear_{i}_{t}")
        model.addConstr(l[i, t] == u[i, t] * (1 / (1 + r)), name=f"LostSalesLinear_{i}_{t}")
        model.addConstr(v[i, t] >= f[i, t], name=f"FulfilledBackorder_{i}_{t}")
        
# Non-negativity and binary constraints (6)
for i in S_I:
    for j in S_J:
        for t in S_T:
            model.addConstr(x[i, j, t] >= 0, name=f"NonNeg_x_{i}_{j}_{t}")
for i in S_I:
    for t in S_T:
        model.addConstr(v[i, t] >= 0, name=f"NonNeg_v_{i}_{t}")
for j in S_J:
    for t in S_T:
        model.addConstr(y[j, t] >= 0, name=f"Binary_y_{j}_{t}")  # Already binary due to vtype
for t in S_T:
    model.addConstr(z[t] >= 0, name=f"NonNeg_z_{t}")
for i in S_I:
    for t in S_T:
        model.addConstr(l[i, t] >= 0, name=f"NonNeg_l_{i}_{t}")
for i in S_I:
    for t in S_T:
        model.addConstr(u[i, t] >= 0, name=f"NonNeg_u_{i}_{t}")
for i in S_I:
    for t in S_T:
        model.addConstr(b[i, t] >= 0, name=f"NonNeg_b_{i}_{t}")
for i in S_I:
    for t in S_T:
        model.addConstr(f[i, t] >= 0, name=f"NonNeg_f_{i}_{t}")

# Optimize the model
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    print("\nOptimal objective value:", model.objVal)
    print("\nOrder quantities (x_ijt):")
    
    for t in S_T:
        for i in S_I:
            for j in S_J:
                    if x[i, j, t].x > 0:
                        print(f"x[{i+1},{j+1},{t+1}] = {x[i, j, t].x}") # +1 to make the index consistent
    print("\nEnding inventory (v_it):")
    for t in S_T:
        for i in S_I:
                if v[i, t].x > 0:
                    print(f"v[{i+1},{t+1}] = {v[i, t].x}")
    print("\nShipping method usage (y_jt):")
    for t in S_T:
        for j in S_J:
                if y[j, t].x > 0:
                    print(f"y[{j+1},{t+1}] = {y[j, t].x}")
    print("\nNumber of containers (z_t):")
    for t in S_T:
        if z[t].x > 0:
            print(f"z[{t+1}] = {z[t].x}")
    print("\nLost sales (l_it):")
    for t in S_T:
        for i in S_I:
                if l[i, t].x > 0:
                    print(f"l[{i+1},{t+1}] = {l[i, t].x}")
    print("\nBackorders (b_it):")
    for t in S_T:
        for i in S_I:
                if b[i, t].x > 0:
                    print(f"b[{i+1},{t+1}] = {b[i, t].x}")
    print("\nFulfilled backorders (f_it):")
    for t in S_T:
        for i in S_I:
                if f[i, t].x > 0:
                    print(f"f[{i+1},{t+1}] = {f[i, t].x}")
else:
    print("No optimal solution found.")