#!/usr/bin/env julia
# Prototype / validation script for the polymer (PE/PP) production scheduling MILP
# before it is embedded into the Pluto notebook. Run with:
#   julia --project=. scripts/prototype_model.jl
#
# PE and PP are made on physically separate trains (different reactor and
# catalyst technology -- e.g. gas-phase/slurry PE lines vs. Ziegler-Natta/
# metallocene PP lines), so this schedules TWO independent single-machine
# problems, one per line, running in parallel from t = 0.
#
# Scale: a full month (720 h) horizon, 8 grades (4 per train), each grade
# running a multi-day campaign and changeovers taking a few hours -- a more
# realistic size than a single short campaign window.

using JuMP
using HiGHS

# ---------------------------------------------------------------------------
# 1. Data: a dedicated PE train and a dedicated PP train, 4 grades each
# ---------------------------------------------------------------------------

const HORIZON_HOURS = 720.0 # 30-day scheduling horizon

family = Dict(
    "HDPE-5502"     => :PE,
    "HDPE-6200"     => :PE,
    "LLDPE-2020"    => :PE,
    "LDPE-1922"     => :PE,
    "PP-Homo-1100"  => :PP,
    "PP-Homo-1305"  => :PP,
    "PP-Copo-3300"  => :PP,
    "PP-Impact-7015" => :PP,
)

production_rate = Dict( # tons/hour, on that grade's dedicated train
    "HDPE-5502"      => 5.0,
    "HDPE-6200"      => 5.5,
    "LLDPE-2020"     => 4.5,
    "LDPE-1922"      => 4.0,
    "PP-Homo-1100"   => 5.5,
    "PP-Homo-1305"   => 5.0,
    "PP-Copo-3300"   => 4.8,
    "PP-Impact-7015" => 4.5,
)

# Raw sales orders across the month: 2 due-date clusters per grade (an
# early-month and a late-month ship window), 2 customer orders per cluster.
customer_orders = [
    # HDPE-5502 (blow molding)
    (customer = "A1", grade = "HDPE-5502", qty = 220.0, due = 212.0),
    (customer = "A2", grade = "HDPE-5502", qty = 180.0, due = 220.0),
    (customer = "A3", grade = "HDPE-5502", qty = 200.0, due = 548.0),
    (customer = "A4", grade = "HDPE-5502", qty = 200.0, due = 556.0),
    # HDPE-6200 (film)
    (customer = "B1", grade = "HDPE-6200", qty = 240.0, due = 260.0),
    (customer = "B2", grade = "HDPE-6200", qty = 200.0, due = 268.0),
    (customer = "B3", grade = "HDPE-6200", qty = 240.0, due = 596.0),
    (customer = "B4", grade = "HDPE-6200", qty = 200.0, due = 604.0),
    # LLDPE-2020 (film)
    (customer = "C1", grade = "LLDPE-2020", qty = 190.0, due = 188.0),
    (customer = "C2", grade = "LLDPE-2020", qty = 150.0, due = 196.0),
    (customer = "C3", grade = "LLDPE-2020", qty = 190.0, due = 524.0),
    (customer = "C4", grade = "LLDPE-2020", qty = 150.0, due = 532.0),
    # LDPE-1922 (extrusion coating)
    (customer = "D1", grade = "LDPE-1922", qty = 150.0, due = 308.0),
    (customer = "D2", grade = "LDPE-1922", qty = 130.0, due = 316.0),
    (customer = "D3", grade = "LDPE-1922", qty = 150.0, due = 644.0),
    (customer = "D4", grade = "LDPE-1922", qty = 130.0, due = 652.0),
    # PP-Homo-1100 (injection)
    (customer = "E1", grade = "PP-Homo-1100", qty = 260.0, due = 236.0),
    (customer = "E2", grade = "PP-Homo-1100", qty = 210.0, due = 244.0),
    (customer = "E3", grade = "PP-Homo-1100", qty = 260.0, due = 572.0),
    (customer = "E4", grade = "PP-Homo-1100", qty = 210.0, due = 580.0),
    # PP-Homo-1305 (fiber)
    (customer = "F1", grade = "PP-Homo-1305", qty = 220.0, due = 284.0),
    (customer = "F2", grade = "PP-Homo-1305", qty = 180.0, due = 292.0),
    (customer = "F3", grade = "PP-Homo-1305", qty = 220.0, due = 620.0),
    (customer = "F4", grade = "PP-Homo-1305", qty = 180.0, due = 628.0),
    # PP-Copo-3300 (random copolymer)
    (customer = "G1", grade = "PP-Copo-3300", qty = 200.0, due = 224.0),
    (customer = "G2", grade = "PP-Copo-3300", qty = 160.0, due = 232.0),
    (customer = "G3", grade = "PP-Copo-3300", qty = 200.0, due = 560.0),
    (customer = "G4", grade = "PP-Copo-3300", qty = 160.0, due = 568.0),
    # PP-Impact-7015 (impact copolymer)
    (customer = "H1", grade = "PP-Impact-7015", qty = 175.0, due = 332.0),
    (customer = "H2", grade = "PP-Impact-7015", qty = 140.0, due = 340.0),
    (customer = "H3", grade = "PP-Impact-7015", qty = 175.0, due = 668.0),
    (customer = "H4", grade = "PP-Impact-7015", qty = 140.0, due = 676.0),
]

# PP grades carry a higher contractual tardiness weight than PE in this
# example (consistent with earlier, smaller-scale versions of this model).
order_weight(g::String) = family[g] == :PP ? 1.5 : 1.0

# Consolidate same-grade customer orders whose due dates fall within
# `window` hours of the earliest (most urgent) due date in the group into
# one production lot: quantity = sum, due = earliest.
function consolidate_orders(customer_orders; window = 24.0)
    lots = NamedTuple[]
    next_id = 1
    for g in unique(o.grade for o in customer_orders)
        group = sort(filter(o -> o.grade == g, customer_orders), by = o -> o.due)
        bucket = eltype(group)[]
        for o in group
            if !isempty(bucket) && o.due - bucket[1].due > window
                push!(lots, (id = next_id, grade = g, qty = sum(b.qty for b in bucket),
                             due = bucket[1].due, weight = order_weight(g),
                             n_combined = length(bucket)))
                next_id += 1
                empty!(bucket)
            end
            push!(bucket, o)
        end
        if !isempty(bucket)
            push!(lots, (id = next_id, grade = g, qty = sum(b.qty for b in bucket),
                         due = bucket[1].due, weight = order_weight(g),
                         n_combined = length(bucket)))
            next_id += 1
        end
    end
    lots
end

orders = consolidate_orders(customer_orders; window = 24.0)

# Changeover time (hours) within a train: an explicit, asymmetric grade-to-
# grade matrix (purging to a lighter/cleaner grade is faster than the
# reverse). Only pairs within the same family are ever needed since PE and
# PP never share equipment. All within "a few hours", as is typical.
const CHANGEOVER = Dict(
    ("HDPE-5502",  "HDPE-5502")  => 0.5, ("HDPE-5502",  "HDPE-6200")  => 2.0,
    ("HDPE-5502",  "LLDPE-2020") => 3.0, ("HDPE-5502",  "LDPE-1922")  => 6.0,
    ("HDPE-6200",  "HDPE-5502")  => 2.5, ("HDPE-6200",  "HDPE-6200")  => 0.5,
    ("HDPE-6200",  "LLDPE-2020") => 3.5, ("HDPE-6200",  "LDPE-1922")  => 6.5,
    ("LLDPE-2020", "HDPE-5502")  => 3.5, ("LLDPE-2020", "HDPE-6200")  => 3.0,
    ("LLDPE-2020", "LLDPE-2020") => 0.5, ("LLDPE-2020", "LDPE-1922")  => 4.0,
    ("LDPE-1922",  "HDPE-5502")  => 5.0, ("LDPE-1922",  "HDPE-6200")  => 5.5,
    ("LDPE-1922",  "LLDPE-2020") => 4.5, ("LDPE-1922",  "LDPE-1922")  => 0.5,

    ("PP-Homo-1100",  "PP-Homo-1100")  => 0.5, ("PP-Homo-1100",  "PP-Homo-1305")  => 1.5,
    ("PP-Homo-1100",  "PP-Copo-3300")  => 3.0, ("PP-Homo-1100",  "PP-Impact-7015") => 4.0,
    ("PP-Homo-1305",  "PP-Homo-1100")  => 2.0, ("PP-Homo-1305",  "PP-Homo-1305")  => 0.5,
    ("PP-Homo-1305",  "PP-Copo-3300")  => 3.5, ("PP-Homo-1305",  "PP-Impact-7015") => 4.5,
    ("PP-Copo-3300",  "PP-Homo-1100")  => 2.5, ("PP-Copo-3300",  "PP-Homo-1305")  => 3.0,
    ("PP-Copo-3300",  "PP-Copo-3300")  => 0.5, ("PP-Copo-3300",  "PP-Impact-7015") => 3.0,
    ("PP-Impact-7015", "PP-Homo-1100")  => 3.5, ("PP-Impact-7015", "PP-Homo-1305")  => 4.0,
    ("PP-Impact-7015", "PP-Copo-3300")  => 2.5, ("PP-Impact-7015", "PP-Impact-7015") => 0.5,
)
changeover_time(gi::String, gj::String) = CHANGEOVER[(gi, gj)]

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

    bigM = startup_time + sum(pt) + sum(maximum(sm[i, :]) for i in M)

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

println(length(customer_orders), " customer orders consolidated into ", length(orders), " production lots over a ",
        Int(HORIZON_HOURS), "-hour (", Int(HORIZON_HOURS / 24), "-day) horizon")

pe_orders = filter(o -> family[o.grade] == :PE, orders)
pp_orders = filter(o -> family[o.grade] == :PP, orders)

pe = schedule_line(pe_orders)
pp = schedule_line(pp_orders)

println("Total cost (\$): ", round(pe.cost + pp.cost, digits = 2))

for (label, res, os) in (("PE line", pe, pe_orders), ("PP line", pp, pp_orders))
    println("\n== ", label, " ==  cost = \$", round(res.cost, digits = 2),
            "  makespan = ", round(maximum(res.C) / 24, digits = 1), " days")
    println("id  grade             qty   start(d) end(d)  due(d)  tardy(d)")
    for i in res.seq
        st = res.C[i] - res.pt[i]
        println(rpad(os[i].id, 4), rpad(os[i].grade, 17), rpad(os[i].qty, 6),
                rpad(round(st / 24, digits = 1), 9), rpad(round(res.C[i] / 24, digits = 1), 8),
                rpad(round(os[i].due / 24, digits = 1), 8), round(res.T[i] / 24, digits = 1))
    end
end
