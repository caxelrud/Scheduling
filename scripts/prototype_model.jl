#!/usr/bin/env julia
# Prototype / validation script for the polymer (PE/PP) production scheduling MILP
# before it is embedded into the Pluto notebook. Run with:
#   julia --project=. scripts/prototype_model.jl
#
# PE and PP are made on physically separate trains (different reactor and
# catalyst technology -- e.g. gas-phase/slurry PE lines vs. Ziegler-Natta/
# metallocene PP lines), so this schedules TWO independent single-machine
# problems, one per line, running in parallel from t = 0.

using JuMP
using HiGHS

# ---------------------------------------------------------------------------
# 1. Data: a dedicated PE train and a dedicated PP train
# ---------------------------------------------------------------------------

family = Dict(
    "HDPE-5502"    => :PE,
    "LLDPE-2020"   => :PE,
    "LDPE-1922"    => :PE,
    "PP-Homo-1100" => :PP,
    "PP-Copo-3300" => :PP,
)

orders = [
    (id = 1, grade = "HDPE-5502",    qty = 120.0, due = 30.0,  weight = 1.0),
    (id = 2, grade = "LLDPE-2020",   qty = 80.0,  due = 40.0,  weight = 1.0),
    (id = 3, grade = "PP-Homo-1100", qty = 100.0, due = 55.0,  weight = 1.5),
    (id = 4, grade = "LDPE-1922",    qty = 60.0,  due = 65.0,  weight = 1.0),
    (id = 5, grade = "PP-Copo-3300", qty = 90.0,  due = 80.0,  weight = 1.5),
    (id = 6, grade = "HDPE-5502",    qty = 70.0,  due = 95.0,  weight = 1.0),
    (id = 7, grade = "PP-Homo-1100", qty = 110.0, due = 110.0, weight = 1.5),
]

production_rate = Dict( # tons/hour, on that grade's dedicated line
    "HDPE-5502"    => 5.0,
    "LLDPE-2020"   => 4.5,
    "LDPE-1922"    => 4.0,
    "PP-Homo-1100" => 5.5,
    "PP-Copo-3300" => 4.8,
)

# Changeover time (hours) within a line: only ever between grades of the
# SAME family, since PE and PP never share equipment.
#   same grade repeated -> 0.5 h (housekeeping only)
#   different grade      -> 2.5 h (color/additive change, partial purge)
changeover_time(gi::String, gj::String) = gi == gj ? 0.5 : 2.5

startup_time = 1.0 # line idle -> first grade of the campaign

changeover_cost_per_hour = 450.0   # $/h lost capacity + purge material
tardiness_cost_per_hour  = 300.0   # $/h per unit weight, contractual penalty

# ---------------------------------------------------------------------------
# 2. Single-line sequencing MILP (asymmetric-TSP-with-time-windows style,
#    node 0 = dummy start), applied independently to each line's orders
# ---------------------------------------------------------------------------

function schedule_line(line_orders)
    m = length(line_orders)
    pt = [o.qty / production_rate[o.grade] for o in line_orders]
    sm = [changeover_time(line_orders[i].grade, line_orders[j].grade) for i in 1:m, j in 1:m]

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    M  = 1:m
    M0 = 0:m

    @variable(model, x[i in M0, j in M0; i != j && j != 0], Bin)
    @variable(model, C[M] >= 0)
    @variable(model, T[M] >= 0)

    @constraint(model, sum(x[0, j] for j in M) == 1)
    @constraint(model, [j in M], sum(x[i, j] for i in M0 if i != j) == 1)
    @constraint(model, [i in M], sum(x[i, j] for j in M if j != i) <= 1)

    bigM = startup_time + sum(pt) + (m > 0 ? sum(maximum(sm[i, :]) for i in M) : 0.0)

    @constraint(model, [j in M], C[j] >= startup_time + pt[j] - bigM * (1 - x[0, j]))
    @constraint(model, [i in M, j in M; i != j],
        C[j] >= C[i] + sm[i, j] + pt[j] - bigM * (1 - x[i, j]))
    @constraint(model, [i in M], T[i] >= C[i] - line_orders[i].due)

    @objective(model, Min,
        changeover_cost_per_hour * sum(sm[i, j] * x[i, j] for i in M, j in M if i != j) +
        changeover_cost_per_hour * sum(startup_time * x[0, j] for j in M) +
        tardiness_cost_per_hour * sum(line_orders[i].weight * T[i] for i in M)
    )

    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL

    xval = value.(x)
    succ = Dict{Int, Int}()
    first_order = 0
    for j in M
        xval[0, j] > 0.5 && (first_order = j)
    end
    for i in M, j in M
        i != j && xval[i, j] > 0.5 && (succ[i] = j)
    end
    seq = Int[first_order]
    let cur = first_order
        while haskey(succ, cur)
            cur = succ[cur]
            push!(seq, cur)
        end
    end

    return (seq = seq, pt = pt, C = value.(C), T = value.(T), cost = objective_value(model))
end

pe_orders = filter(o -> family[o.grade] == :PE, orders)
pp_orders = filter(o -> family[o.grade] == :PP, orders)

pe = schedule_line(pe_orders)
pp = schedule_line(pp_orders)

println("Total cost (\$): ", round(pe.cost + pp.cost, digits = 2))

for (label, res, os) in (("PE line", pe, pe_orders), ("PP line", pp, pp_orders))
    println("\n== ", label, " ==  cost = \$", round(res.cost, digits = 2))
    println("id  grade            qty   start   end    due   tardy")
    for i in res.seq
        st = res.C[i] - res.pt[i]
        println(rpad(os[i].id, 4), rpad(os[i].grade, 16), rpad(os[i].qty, 6),
                rpad(round(st, digits = 1), 8), rpad(round(res.C[i], digits = 1), 7),
                rpad(os[i].due, 6), round(res.T[i], digits = 1))
    end
end
