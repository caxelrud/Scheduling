#!/usr/bin/env julia
# Prototype / validation script for the polymer (PE/PP) production scheduling MILP
# before it is embedded into the Pluto notebook. Run with:
#   julia --project=. scripts/prototype_model.jl

using JuMP
using HiGHS

# ---------------------------------------------------------------------------
# 1. Data: a single compounding/extrusion line producing PE and PP grades
# ---------------------------------------------------------------------------

family = Dict(
    "HDPE-5502"    => :PE,
    "LLDPE-2020"   => :PE,
    "LDPE-1922"    => :PE,
    "PP-Homo-1100" => :PP,
    "PP-Copo-3300" => :PP,
)

# Orders: grade, quantity in tons, due date in hours from t=0, tardiness weight
orders = [
    (id = 1, grade = "HDPE-5502",    qty = 120.0, due = 30.0,  weight = 1.0),
    (id = 2, grade = "LLDPE-2020",   qty = 80.0,  due = 40.0,  weight = 1.0),
    (id = 3, grade = "PP-Homo-1100", qty = 100.0, due = 55.0,  weight = 1.5),
    (id = 4, grade = "LDPE-1922",    qty = 60.0,  due = 65.0,  weight = 1.0),
    (id = 5, grade = "PP-Copo-3300", qty = 90.0,  due = 80.0,  weight = 1.5),
    (id = 6, grade = "HDPE-5502",    qty = 70.0,  due = 95.0,  weight = 1.0),
    (id = 7, grade = "PP-Homo-1100", qty = 110.0, due = 110.0, weight = 1.5),
]
n = length(orders)

production_rate = Dict( # tons/hour, on the single line
    "HDPE-5502"    => 5.0,
    "LLDPE-2020"   => 4.5,
    "LDPE-1922"    => 4.0,
    "PP-Homo-1100" => 5.5,
    "PP-Copo-3300" => 4.8,
)

proc_time = [orders[i].qty / production_rate[orders[i].grade] for i in 1:n]

# Sequence-dependent changeover time (hours), driven by polymer family:
#   same grade repeated       -> 0.5 h (housekeeping only)
#   same family, diff grade   -> 2.5 h (color/additive change, partial purge)
#   PE <-> PP family switch   -> 7.0 h (full reactor/line purge + catalyst change)
function changeover_time(gi::String, gj::String)
    if gi == gj
        return 0.5
    elseif family[gi] == family[gj]
        return 2.5
    else
        return 7.0
    end
end

s = [changeover_time(orders[i].grade, orders[j].grade) for i in 1:n, j in 1:n]

# Startup time from an idle/cleaned line to the first grade of the campaign
startup_time = 1.0

changeover_cost_per_hour = 450.0   # $/h lost capacity + purge material
tardiness_cost_per_hour  = 300.0   # $/h per unit of tardiness weight

# ---------------------------------------------------------------------------
# 2. MILP: single-machine sequencing with sequence-dependent setups
#    (asymmetric-TSP-with-time-windows-style formulation; node 0 = dummy start)
# ---------------------------------------------------------------------------

model = Model(HiGHS.Optimizer)
set_silent(model)

N  = 1:n
N0 = 0:n  # 0 = dummy "line idle" start node

@variable(model, x[i in N0, j in N0; i != j && j != 0], Bin)
@variable(model, C[N] >= 0)      # completion time of order i
@variable(model, T[N] >= 0)      # tardiness of order i

# exactly one order starts the campaign
@constraint(model, sum(x[0, j] for j in N) == 1)

# every order has exactly one predecessor (from 0 or another order)
@constraint(model, [j in N], sum(x[i, j] for i in N0 if i != j) == 1)

# every order has at most one successor (open path, last order has none)
@constraint(model, [i in N], sum(x[i, j] for j in N if j != i) <= 1)

bigM = startup_time + sum(proc_time) + sum(maximum(s[i, :]) for i in N)

# timing: completion propagates along the chosen arcs; big-M relaxes the
# inactive arcs. Because completion times must strictly increase along any
# chain, this also rules out sub-tours without needing MTZ constraints.
@constraint(model, [j in N], C[j] >= startup_time + proc_time[j] - bigM * (1 - x[0, j]))
@constraint(model, [i in N, j in N; i != j],
    C[j] >= C[i] + s[i, j] + proc_time[j] - bigM * (1 - x[i, j]))

# tardiness
@constraint(model, [i in N], T[i] >= C[i] - orders[i].due)

# objective: minimize total changeover cost + weighted tardiness cost
@objective(model, Min,
    changeover_cost_per_hour * sum(s[i, j] * x[i, j] for i in N, j in N if i != j) +
    changeover_cost_per_hour * sum(startup_time * x[0, j] for j in N) +
    tardiness_cost_per_hour * sum(orders[i].weight * T[i] for i in N)
)

optimize!(model)

println("Termination status: ", termination_status(model))
println("Objective (\$): ", round(objective_value(model), digits = 2))

# reconstruct sequence
xval = value.(x)
succ = Dict{Int, Int}()
first_order = 0
for j in N
    if xval[0, j] > 0.5
        global first_order = j
    end
end
for i in N, j in N
    if i != j && xval[i, j] > 0.5
        succ[i] = j
    end
end

seq = Int[first_order]
let cur = first_order
    while haskey(succ, cur)
        cur = succ[cur]
        push!(seq, cur)
    end
end

println("\nSequence: ", [orders[i].grade for i in seq])
println("\nid  grade            qty   start   end    due   tardy")
for i in seq
    st = value(C[i]) - proc_time[i]
    println(rpad(orders[i].id, 4), rpad(orders[i].grade, 16), rpad(orders[i].qty, 6),
            rpad(round(st, digits = 1), 8), rpad(round(value(C[i]), digits = 1), 7),
            rpad(orders[i].due, 6), round(value(T[i]), digits = 1))
end
