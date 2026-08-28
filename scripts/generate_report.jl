#!/usr/bin/env julia
# Builds a self-contained (no external network resources) HTML report of the
# polymer PE/PP scheduling case -- same model & data as
# notebooks/polymer_pe_pp_scheduling.jl -- suitable for offline PDF printing.
# Run with:
#   julia --project=. scripts/generate_report.jl

using JuMP
using HiGHS
using Plots
using Printf
using Base64

# ---------------------------------------------------------------------------
# 1. Data (mirrors notebooks/polymer_pe_pp_scheduling.jl)
#
# PE and PP run on physically separate trains (different reactor/catalyst
# technology), so this is two independent single-line problems solved in
# parallel, not one shared line that switches families.
# ---------------------------------------------------------------------------

family = Dict(
    "HDPE-5502"    => :PE,
    "LLDPE-2020"   => :PE,
    "LDPE-1922"    => :PE,
    "PP-Homo-1100" => :PP,
    "PP-Copo-3300" => :PP,
)

production_rate = Dict(
    "HDPE-5502"    => 5.0,
    "LLDPE-2020"   => 4.5,
    "LDPE-1922"    => 4.0,
    "PP-Homo-1100" => 5.5,
    "PP-Copo-3300" => 4.8,
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
n = length(orders)
pe_orders = filter(o -> family[o.grade] == :PE, orders)
pp_orders = filter(o -> family[o.grade] == :PP, orders)

# Changeover time (hours) within a train: only ever between grades of the
# SAME family, since PE and PP never share equipment.
changeover_time(gi::String, gj::String) = gi == gj ? 0.5 : 2.5
startup_time = 1.0

changeover_cost_per_hour = 450.0
tardiness_cost_per_hour  = 300.0

# ---------------------------------------------------------------------------
# 2. Single-train sequencing MILP (identical formulation to the notebook),
#    applied independently to each train's orders
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
    @assert termination_status(model) == MOI.OPTIMAL "Solver did not reach optimality"

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

    (seq = seq, pt = pt, C = value.(C), T = value.(T), cost = objective_value(model))
end

pe = schedule_line(pe_orders)
pp = schedule_line(pp_orders)

total_cost = pe.cost + pp.cost
makespan_pe = maximum(pe.C)
makespan_pp = maximum(pp.C)
overall_makespan = max(makespan_pe, makespan_pp)
total_tardy_h = sum(pe.T) + sum(pp.T)
n_tardy_orders = count(>(1e-6), vcat(pe.T, pp.T))

# ---------------------------------------------------------------------------
# 3. Gantt chart (two rows, one per train, both starting at t = 0) -> PNG ->
#    base64 (so the HTML needs zero external requests)
# ---------------------------------------------------------------------------

family_color = Dict(:PE => RGB(0.20, 0.45, 0.85), :PP => RGB(0.90, 0.55, 0.10))

gantt = plot(
    size = (1000, 380), dpi = 150,
    xlabel = "Time (h)", ylabel = "", yticks = ([1, 2], ["PP train", "PE train"]),
    ylims = (0, 3), legend = :outertop, legendcolumns = 2, framestyle = :box,
    title = "Two dedicated production trains (running in parallel)",
)
seen = Set{Symbol}()
for (res, os, ypos) in ((pe, pe_orders, 2), (pp, pp_orders, 1))
    for i in res.seq
        start_t = res.C[i] - res.pt[i]
        fam = family[os[i].grade]
        lbl = fam in seen ? "" : String(fam)
        push!(seen, fam)
        plot!(gantt, Shape([start_t, res.C[i], res.C[i], start_t],
                            [ypos - 0.35, ypos - 0.35, ypos + 0.35, ypos + 0.35]),
              color = family_color[fam], linecolor = :black, label = lbl)
        annotate!(gantt, (start_t + res.C[i]) / 2, ypos,
                  text(string(os[i].grade, "\n#", os[i].id), 7, :white, :center))
        due_color = res.T[i] > 1e-6 ? :red : :black
        scatter!(gantt, [os[i].due], [ypos + 0.42], marker = :dtriangle, markersize = 6,
                 color = due_color, label = "")
    end
end

gantt_png_path = tempname() * ".png"
savefig(gantt, gantt_png_path)
gantt_b64 = base64encode(read(gantt_png_path))
rm(gantt_png_path)

# ---------------------------------------------------------------------------
# 4. Results table rows
# ---------------------------------------------------------------------------

function rows_for(res, os, train_label)
    io = IOBuffer()
    for i in res.seq
        start_t = res.C[i] - res.pt[i]
        tardy = res.T[i]
        tardy_str = tardy > 1e-6 ? @sprintf("%.1f h late", tardy) : "on time"
        tardy_class = tardy > 1e-6 ? " class=\"tardy\"" : ""
        println(io, """
        <tr$tardy_class>
          <td>$train_label</td>
          <td>$(os[i].id)</td>
          <td>$(os[i].grade)</td>
          <td>$(@sprintf("%.0f", os[i].qty))</td>
          <td>$(@sprintf("%.1f", start_t))</td>
          <td>$(@sprintf("%.1f", res.C[i]))</td>
          <td>$(@sprintf("%.1f", os[i].due))</td>
          <td>$tardy_str</td>
        </tr>""")
    end
    String(take!(io))
end
rows_html = rows_for(pe, pe_orders, "PE") * rows_for(pp, pp_orders, "PP")

# ---------------------------------------------------------------------------
# 5. Assemble the self-contained HTML report
# ---------------------------------------------------------------------------

html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polymer Plant Scheduling — Dedicated PE &amp; PP Production Trains</title>
<style>
  @page { size: A4; margin: 20mm 16mm; }
  * { box-sizing: border-box; }
  body {
    font-family: Georgia, 'Times New Roman', serif;
    color: #1c1c1c;
    max-width: 900px;
    margin: 0 auto;
    padding: 24px 20px 60px;
    line-height: 1.55;
  }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h1 + p.subtitle { color: #555; margin-top: 0; font-size: 14px; }
  h2 {
    font-size: 19px; margin-top: 2.2em; border-bottom: 2px solid #1c1c1c;
    padding-bottom: 4px;
  }
  h3 { font-size: 15px; margin-top: 1.6em; }
  code, pre {
    font-family: 'DejaVu Sans Mono', Consolas, monospace; font-size: 13px;
    background: #f4f4f2; border-radius: 4px;
  }
  pre { padding: 12px 14px; overflow-x: auto; border-left: 3px solid #999; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 13px; }
  th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
  th { background: #1c1c1c; color: #fff; font-weight: 600; }
  tr:nth-child(even) { background: #f7f7f5; }
  tr.tardy td { color: #a30000; font-weight: 600; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.4em 0;
  }
  .kpi {
    border: 1px solid #ccc; border-radius: 6px; padding: 12px; text-align: center;
  }
  .kpi .label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .04em; }
  .kpi .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
  .legend-pe { color: #3373d9; font-weight: 700; }
  .legend-pp { color: #e58c1a; font-weight: 700; }
  img.gantt { width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 1em 0; }
  .note { background: #fff8e1; border-left: 3px solid #e5a300; padding: 10px 14px; font-size: 13px; }
  footer { margin-top: 3em; font-size: 11px; color: #888; border-top: 1px solid #ddd; padding-top: 10px; }
  ul { margin: 0.6em 0; }
  @media print { body { padding: 0; max-width: none; } h2 { break-after: avoid; } table, .kpi-grid, img { break-inside: avoid; } }
</style>
</head>
<body>

<h1>Polymer Plant Scheduling — Dedicated PE &amp; PP Production Trains</h1>
<p class="subtitle">Manufacturing scheduling case study &middot; Julia + JuMP + HiGHS &middot; companion to <code>notebooks/polymer_pe_pp_scheduling.jl</code></p>

<p>
This report documents a <strong>production scheduling model</strong> for a
polymer plant that manufactures several grades of <strong>polyethylene
(PE)</strong> and <strong>polypropylene (PP)</strong>. PE and PP are made on
<strong>physically separate trains</strong>: different reactor technology
and catalyst systems (e.g. gas-phase/slurry PE reactors vs.
Ziegler-Natta/metallocene PP reactors) mean a PE line can never run a PP
grade or vice versa. So this is really <strong>two independent single-line
scheduling problems, running in parallel</strong> — one per train.
</p>
<p>
Within a train, switching between grades still costs time and money:
color/additive changes and partial purges. The scheduler below decides the
order in which each train runs its confirmed orders so as to minimize total
changeover cost plus tardiness cost, per train. Each train's sub-problem is
a mixed-integer linear program (MILP), built with
<a href="https://jump.dev">JuMP.jl</a> and solved with the open-source
<a href="https://highs.dev">HiGHS</a> solver.
</p>

<h2>1. Plant data: grades, families, and orders</h2>
<p>
The plant has <strong>$(n) confirmed orders</strong> across
<strong>5 grades</strong>: three PE grades (HDPE, LLDPE, LDPE) on the PE
train, and two PP grades (homopolymer and copolymer) on the PP train. Each
order has a quantity, a production rate on its train, and a customer due
date (in hours from the scheduling horizon start, t&nbsp;=&nbsp;0).
</p>
<table>
<thead><tr><th>Order</th><th>Grade</th><th>Train</th><th>Qty (t)</th><th>Rate (t/h)</th><th>Due (h)</th><th>Weight</th></tr></thead>
<tbody>
$(join(["<tr><td>$(o.id)</td><td>$(o.grade)</td><td>$(String(family[o.grade]))</td><td>$(@sprintf("%.0f", o.qty))</td><td>$(@sprintf("%.1f", production_rate[o.grade]))</td><td>$(@sprintf("%.0f", o.due))</td><td>$(o.weight)</td></tr>" for o in orders]))
</tbody>
</table>

<h2>2. Sequence-dependent changeovers (within a train)</h2>
<p>
Because a train only ever runs grades from one family, the only changeover
decision is <em>within</em> that family:
</p>
<table>
<thead><tr><th>Transition</th><th>Time (h)</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Same grade repeated</td><td>0.5</td><td>housekeeping only</td></tr>
<tr><td>Different grade, same family</td><td>2.5</td><td>color/additive change, partial purge</td></tr>
</tbody>
</table>
<p class="note">
There is no PE&harr;PP transition to model — that changeover never happens,
because it would require re-equipping the whole train.
</p>
<p>Cost rates used in this run: changeover
<strong>\$$(@sprintf("%.0f", changeover_cost_per_hour))/h</strong>, tardiness
<strong>\$$(@sprintf("%.0f", tardiness_cost_per_hour))/h</strong> per unit
order weight. (The companion Pluto notebook exposes both as live sliders.)</p>

<h2>3. MILP formulation (applied once per train)</h2>
<p>
Each train's orders are scheduled independently. Node 0 is a dummy "train
idle" start state; nodes 1..m are that train's orders. Binary variables
x<sub>ij</sub> indicate that order j runs immediately after node i.
Continuous variables C<sub>i</sub> and T<sub>i</sub> are the completion time
and tardiness of order i.
</p>
<pre>
minimize    c_co * sum_{i != j} s_ij * x_ij  +  c_tar * sum_i w_i * T_i

subject to  sum_j x_0j = 1
            sum_{i != j} x_ij = 1                       for all j
            sum_{j != i} x_ij &le; 1                       for all i
            C_j &ge; C_i + s_ij + p_j - M(1 - x_ij)         for all i != j
            T_i &ge; C_i - d_i,   T_i &ge; 0
            x_ij &isin; {0,1},   C_i, T_i &ge; 0
</pre>
<p class="note">
Completion times must strictly increase along any chain of active arcs, so
this "big-M timing" formulation rules out sub-tours on its own — no
separate subtour-elimination constraints (e.g. MTZ) are needed. The two
trains' sub-problems share no variables, so solving them independently with
HiGHS is exact, not a heuristic decomposition — both reach global optimality
in well under a second.
</p>

<h2>4. Optimal campaign sequence, per train</h2>
<table>
<thead><tr><th>Train</th><th>Order</th><th>Grade</th><th>Qty (t)</th><th>Start (h)</th><th>End (h)</th><th>Due (h)</th><th>Status</th></tr></thead>
<tbody>
$(rows_html)
</tbody>
</table>

<h2>5. Gantt chart</h2>
<img class="gantt" src="data:image/png;base64,$(gantt_b64)" alt="Gantt chart of the two parallel production trains">
<p>
Grade blocks are colored by polymer family (<span class="legend-pe">PE</span> vs.
<span class="legend-pp">PP</span>); triangles mark each order's due date (red = missed).
Because the two trains never compete for the same equipment, they run
<strong>simultaneously</strong>: overall makespan is the <em>slower</em> of
the two trains, not the sum of both.
</p>

<h2>6. Key performance indicators</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="label">PE train makespan</div><div class="value">$(@sprintf("%.1f", makespan_pe)) h</div></div>
  <div class="kpi"><div class="label">PP train makespan</div><div class="value">$(@sprintf("%.1f", makespan_pp)) h</div></div>
  <div class="kpi"><div class="label">Overall makespan</div><div class="value">$(@sprintf("%.1f", overall_makespan)) h</div></div>
  <div class="kpi"><div class="label">Total cost</div><div class="value">\$$(@sprintf("%.0f", total_cost))</div></div>
</div>
<p>
Tardiness: $(@sprintf("%.1f", total_tardy_h)) h across $(n_tardy_orders) of $(n) orders.
PE train cost: \$$(@sprintf("%.0f", pe.cost)); PP train cost: \$$(@sprintf("%.0f", pp.cost)).
</p>

<h2>7. Discussion &amp; extensions</h2>
<p>
Modeling PE and PP as two <strong>dedicated, parallel trains</strong> —
rather than one shared line that pays a family-changeover penalty — matters
economically: it removes a changeover that can never actually happen on
real equipment, and it lets both product families progress at once instead
of queueing behind each other. Natural next steps for a production-grade
version:
</p>
<ul>
  <li><strong>More than one train per family</strong> — e.g. two PE lines with different capability sets; extend the model to a proper parallel-machine assignment problem within a family.</li>
  <li><strong>Minimum/maximum campaign length</strong> — avoid uneconomically short runs by bounding the number of consecutive orders of the same grade.</li>
  <li><strong>Exact grade-pair changeover matrix</strong> — replace the flat "0.5 / 2.5 h" rule with real historical purge times per ordered grade pair.</li>
  <li><strong>Shared upstream/downstream utilities</strong> — if both trains draw from a common feedstock or packaging line, that shared resource needs its own capacity constraint linking the two schedules.</li>
  <li><strong>Rolling-horizon re-optimization</strong> — re-solve as new orders arrive or priorities change, warm-starting from the current sequence.</li>
  <li><strong>Robustness</strong> — sample uncertain processing rates / late raw-material arrivals and re-optimize, or solve a chance-constrained variant.</li>
</ul>
<p>
Because each train's model is a compact, independent JuMP program, all of
the above are incremental changes to the scheduling function above, not a
rewrite. An interactive version of this same model — with live cost-weight
sliders — is available as a Pluto.jl notebook at
<code>notebooks/polymer_pe_pp_scheduling.jl</code> in this repository.
</p>

<footer>Generated from the Scheduling repository (Julia $(VERSION), JuMP.jl, HiGHS solver).</footer>

</body>
</html>
"""

repo = normpath(joinpath(@__DIR__, ".."))
docs_dir = joinpath(repo, "docs")
mkpath(docs_dir)
outpath = joinpath(docs_dir, "polymer_pe_pp_scheduling.html")
write(outpath, html)
println("Wrote self-contained report: ", outpath, " (", length(html), " bytes, ", length(gantt_b64), " b64 chars for the chart)")
