# 九节点逐时调度模型（0-24点）
# ✅ 中间版本：保留备用、ε松弛、网损处理；关闭 ramp 限制，聚焦可行性与松弛触发源头分析

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
import pandas as pd
import numpy as np
import os

# ========== 1. 参数设定 ==========
file_path = "九节点.xlsx"
current_scenario = 1
current_day = 1
reserve_ratio = 0.05  # 备用容量为总负荷的5%
rainy_season_hours = 4380
reasonable_rainy_output_hours = 3000
hydro_factor_adjusted = 0.9 * (3000 / 4380)
transmission_loss_rate = 0.0317
loss_factor = 1 + transmission_loss_rate
slack_penalty = 1e12

# ========== 2. 数据读取 ==========
xls = pd.ExcelFile(file_path)
nodes_data = pd.read_excel(xls, sheet_name="节点")
generator_data = pd.read_excel(xls, sheet_name="机组")
load_data = pd.read_excel(xls, sheet_name="负荷")
lines_data = pd.read_excel(xls, sheet_name="线路")
renewable_data = pd.read_excel(xls, sheet_name="新能源")

for df in [nodes_data, generator_data, load_data, lines_data, renewable_data]:
    df.columns = df.columns.str.strip()

hours = [f"{h}点" for h in range(1, 25)]

# ========== 3. 数据预处理 ==========
load_data[["场景", "天"]] = load_data[["场景", "天"]].astype(int)
filtered_load = load_data[(load_data["场景"] == current_scenario) & (load_data["天"] == current_day)]
load_matrix = filtered_load.set_index("区域ID")[hours]

renewable_data[["场景", "天"]] = renewable_data[["场景", "天"]].astype(int)
filtered_renewable = renewable_data[(renewable_data["场景"] == current_scenario) & (renewable_data["天"] == current_day)]
renewable_matrix = filtered_renewable.set_index("机组编号")[hours].fillna(1)

generator_data[hours] = generator_data[["装机容量/MW"]].values.repeat(24, axis=1)
renewable_ids = generator_data["机组编号"].isin(renewable_matrix.index)
generator_data.loc[renewable_ids, hours] = (
    generator_data.loc[renewable_ids, "装机容量/MW"].values[:, None] *
    renewable_matrix.loc[generator_data.loc[renewable_ids, "机组编号"].values].values
)

is_hydro = generator_data["机组类型(1:火电；4：水电；14：储能；11：风电；12：光伏；13：光热)"].astype(int) == 4
for h in hours:
    generator_data.loc[is_hydro, h] *= hydro_factor_adjusted

# ========== 4. 调度模型核心 ==========
G = list(generator_data.index)
N = list(nodes_data.index)
L = list(lines_data.index)
hour_map = dict(zip(range(24), hours))
ATC_limits = lines_data["线路容量/MW"].values * 0.95

node_price_output = []
gen_dispatch_output = []

for t in range(0, 24):
    model = ConcreteModel()
    model.G = Set(initialize=G)
    model.N = Set(initialize=N)
    model.L = Set(initialize=L)
    model.Pgen = Var(model.G, domain=NonNegativeReals)
    model.Sreserve = Var(model.G, domain=NonNegativeReals)
    model.NEX = Var(model.L, domain=Reals)
    model.epsilon = Var(model.N, domain=NonNegativeReals)
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    cost = generator_data["可变成本( 元/MWh )"].clip(lower=0).to_dict()
    gen_node = generator_data["所属节点"].to_dict()
    from_node = lines_data["线路起始节点"].values
    to_node = lines_data["线路终止节点"].values
    max_output = {g: generator_data.loc[g, hour_map[t]] for g in G}
    load_at_node = {
        n: load_matrix.loc[nodes_data.loc[n, "区域ID"], hour_map[t]] if nodes_data.loc[n, "区域ID"] in load_matrix.index else 0
        for n in N
    }

    model.obj = Objective(
        expr=sum(model.Pgen[g] * cost[g] + model.Sreserve[g] * cost[g] for g in G) +
             sum(slack_penalty * model.epsilon[n] for n in N),
        sense=minimize
    )

    model.balance = ConstraintList()
    for n in N:
        node_id = nodes_data.loc[n, "节点编号"]
        inflow = sum(model.NEX[l] / loss_factor for l in L if to_node[l] == node_id)
        outflow = sum(model.NEX[l] for l in L if from_node[l] == node_id)
        gen_sum = sum(model.Pgen[g] for g in G if gen_node[g] == node_id)
        model.balance.add(gen_sum + inflow - outflow + model.epsilon[n] == load_at_node[n])

    model.gen_limit = ConstraintList()
    for g in G:
        model.gen_limit.add(model.Pgen[g] + model.Sreserve[g] <= max_output[g])

    model.line_flow = ConstraintList()
    for l in L:
        model.line_flow.add(model.NEX[l] <= ATC_limits[l])
        model.line_flow.add(model.NEX[l] >= -ATC_limits[l])

    total_load = sum(load_at_node.values())
    model.reserve = ConstraintList()
    model.reserve.add(sum(model.Sreserve[g] for g in G) >= reserve_ratio * total_load)
    for g in G:
        model.reserve.add(model.Sreserve[g] <= max_output[g])

    # ❌ ramp 限制已禁用（当前版本用于简化分析）

    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=False)

    feasible = (results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal)

    if feasible:
        eps_values = {n: value(model.epsilon[n]) for n in N}
        active_eps = {n: eps for n, eps in eps_values.items() if eps > 1e-3}
        print(f"✅ 时刻 {t+1} 可行")
        if active_eps:
            print("    ⚠️ 使用 ε 松弛节点：", [(nodes_data.loc[n, "节点编号"], round(eps, 2)) for n, eps in active_eps.items()])

        for n in N:
            node_price_output.append({
                "hour": t+1,
                "node_id": nodes_data.loc[n, "节点编号"],
                "price": model.dual.get(model.balance[n+1], np.nan)
            })
        for g in G:
            gen_dispatch_output.append({
                "hour": t+1,
                "gen_id": generator_data.loc[g, "机组编号"],
                "node_id": gen_node[g],
                "Pgen": value(model.Pgen[g]),
                "Sreserve": value(model.Sreserve[g])
            })
    else:
        print(f"❌ 时刻 {t+1} 不可行")

# ========== 5. 保存结果输出 ==========
pd.DataFrame(node_price_output).to_excel("node_prices_mid.xlsx", index=False)
pd.DataFrame(gen_dispatch_output).to_excel("gen_dispatch_mid.xlsx", index=False)
