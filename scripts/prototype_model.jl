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

# Raw sales orders, as they actually arrive from customers: many small,
# separately-negotiated quantities per grade, each with its own ship date.
# A plant does not run one production campaign per purchase order -- orders
# for the same grade shipping close together get pooled into one production
# lot, sized to cover all of them, timed to the earliest ship date in the
# group (producing early never hurts the later ones in the group; it just
# means finished goods sit in the warehouse a few extra hours/days).
customer_orders = [
    (customer = "A1", grade = "HDPE-5502",    qty = 45.0, due = 28.0,  weight = 1.0),
    (customer = "A2", grade = "HDPE-5502",    qty = 35.0, due = 30.0,  weight = 1.0),
    (customer = "A3", grade = "HDPE-5502",    qty = 40.0, due = 33.0,  weight = 1.0),
    (customer = "B1", grade = "LLDPE-2020",   qty = 50.0, due = 38.0,  weight = 1.0),
    (customer = "B2", grade = "LLDPE-2020",   qty = 30.0, due = 42.0,  weight = 1.0),
    (customer = "C1", grade = "PP-Homo-1100", qty = 60.0, due = 52.0,  weight = 1.5),
    (customer = "C2", grade = "PP-Homo-1100", qty = 40.0, due = 57.0,  weight = 1.5),
    (customer = "D1", grade = "LDPE-1922",    qty = 25.0, due = 62.0,  weight = 1.0),
    (customer = "D2", grade = "LDPE-1922",    qty = 35.0, due = 68.0,  weight = 1.0),
    (customer = "E1", grade = "PP-Copo-3300", qty = 50.0, due = 76.0,  weight = 1.5),
    (customer = "E2", grade = "PP-Copo-3300", qty = 40.0, due = 84.0,  weight = 1.5),
    (customer = "A4", grade = "HDPE-5502",    qty = 30.0, due = 90.0,  weight = 1.0),
    (customer = "A5", grade = "HDPE-5502",    qty = 40.0, due = 98.0,  weight = 1.0),
    (customer = "C3", grade = "PP-Homo-1100", qty = 70.0, due = 105.0, weight = 1.5),
    (customer = "C4", grade = "PP-Homo-1100", qty = 40.0, due = 112.0, weight = 1.5),
]

# Consolidate same-grade customer orders whose due dates fall within
# `window` hours of the earliest (most urgent) due date in the group into
# one production lot: quantity = sum, due = earliest, weight = the most
# urgent customer's weight (a lot inherits its most demanding member).
function consolidate_orders(customer_orders; window = 24.0)
    lots = NamedTuple[]
    next_id = 1
    for g in unique(o.grade for o in customer_orders)
        group = sort(filter(o -> o.grade == g, customer_orders), by = o -> o.due)
        bucket = eltype(group)[]
        for o in group
            if !isempty(bucket) && o.due - bucket[1].due > window
                push!(lots, (id = next_id, grade = g, qty = sum(b.qty for b in bucket),
                             due = bucket[1].due, weight = maximum(b.weight for b in bucket),
                             n_combined = length(bucket)))
                next_id += 1
                empty!(bucket)
            end
            push!(bucket, o)
        end
        if !isempty(bucket)
            push!(lots, (id = next_id, grade = g, qty = sum(b.qty for b in bucket),
                         due = bucket[1].due, weight = maximum(b.weight for b in bucket),
                         n_combined = length(bucket)))
            next_id += 1
        end
    end
    lots
end

orders = consolidate_orders(customer_orders; window = 24.0)

production_rate = Dict( # tons/hour, on that grade's dedicated line
    "HDPE-5502"    => 5.0,
    "LLDPE-2020"   => 4.5,
    "LDPE-1922"    => 4.0,
    "PP-Homo-1100" => 5.5,
    "PP-Copo-3300" => 4.8,
)

# Changeover time (hours) within a line: an explicit grade-to-grade matrix,
# since real purge/transition times depend on the specific pair, not just
# "same family or not" -- and they are often ASYMMETRIC (e.g. going to a
# lighter color or lower-additive grade purges faster than the reverse).
# Only pairs within the same family are ever needed since PE and PP never
# share equipment.
const CHANGEOVER = Dict(
    ("HDPE-5502",    "HDPE-5502")    => 0.5,
    ("HDPE-5502",    "LLDPE-2020")   => 2.0,
    ("HDPE-5502",    "LDPE-1922")    => 3.5,
    ("LLDPE-2020",   "HDPE-5502")    => 2.5,
    ("LLDPE-2020",   "LLDPE-2020")   => 0.5,
    ("LLDPE-2020",   "LDPE-1922")    => 2.0,
    ("LDPE-1922",    "HDPE-5502")    => 3.0,
    ("LDPE-1922",    "LLDPE-2020")   => 2.5,
    ("LDPE-1922",    "LDPE-1922")    => 0.5,
    ("PP-Homo-1100", "PP-Homo-1100") => 0.5,
    ("PP-Homo-1100", "PP-Copo-3300") => 3.0,
    ("PP-Copo-3300", "PP-Homo-1100") => 2.0,
    ("PP-Copo-3300", "PP-Copo-3300") => 0.5,
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

println(length(customer_orders), " customer orders consolidated into ", length(orders), " production lots:")
for o in orders
    println(rpad(o.id, 4), rpad(o.grade, 16), rpad(o.qty, 6), rpad(o.due, 6),
             "(combines ", o.n_combined, " customer order(s))")
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
