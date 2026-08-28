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
proc_time = [orders[i].qty / production_rate[orders[i].grade] for i in 1:n]

function changeover_time(gi::String, gj::String)
    if gi == gj
        0.5
    elseif family[gi] == family[gj]
        2.5
    else
        7.0
    end
end
s = [changeover_time(orders[i].grade, orders[j].grade) for i in 1:n, j in 1:n]
startup_time = 1.0

changeover_cost_per_hour = 450.0
tardiness_cost_per_hour  = 300.0

# ---------------------------------------------------------------------------
# 2. MILP (identical formulation to the notebook)
# ---------------------------------------------------------------------------

model = Model(HiGHS.Optimizer)
set_silent(model)

N  = 1:n
N0 = 0:n

@variable(model, x[i in N0, j in N0; i != j && j != 0], Bin)
@variable(model, C[N] >= 0)
@variable(model, T[N] >= 0)

@constraint(model, sum(x[0, j] for j in N) == 1)
@constraint(model, [j in N], sum(x[i, j] for i in N0 if i != j) == 1)
@constraint(model, [i in N], sum(x[i, j] for j in N if j != i) <= 1)

bigM = startup_time + sum(proc_time) + sum(maximum(s[i, :]) for i in N)

@constraint(model, [j in N], C[j] >= startup_time + proc_time[j] - bigM * (1 - x[0, j]))
@constraint(model, [i in N, j in N; i != j],
    C[j] >= C[i] + s[i, j] + proc_time[j] - bigM * (1 - x[i, j]))
@constraint(model, [i in N], T[i] >= C[i] - orders[i].due)

@objective(model, Min,
    changeover_cost_per_hour * sum(s[i, j] * x[i, j] for i in N, j in N if i != j) +
    changeover_cost_per_hour * sum(startup_time * x[0, j] for j in N) +
    tardiness_cost_per_hour * sum(orders[i].weight * T[i] for i in N)
)

optimize!(model)
@assert termination_status(model) == MOI.OPTIMAL "Solver did not reach optimality"

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
    global seq
    while haskey(succ, cur)
        cur = succ[cur]
        push!(seq, cur)
    end
end

# ---------------------------------------------------------------------------
# 3. Gantt chart -> PNG -> base64 (so the HTML needs zero external requests)
# ---------------------------------------------------------------------------

family_color = Dict(:PE => RGB(0.20, 0.45, 0.85), :PP => RGB(0.90, 0.55, 0.10))

gantt = plot(
    size = (1000, 360), dpi = 150,
    xlabel = "Time (h)", ylabel = "", yticks = :none, ylims = (0, 2),
    legend = :outertop, legendcolumns = 2, framestyle = :box,
    title = "Single-line production campaign",
)
seen = Set{Symbol}()
for i in seq
    start_t = value(C[i]) - proc_time[i]
    fam = family[orders[i].grade]
    lbl = fam in seen ? "" : String(fam)
    push!(seen, fam)
    plot!(gantt, Shape([start_t, value(C[i]), value(C[i]), start_t], [0.4, 0.4, 1.6, 1.6]),
          color = family_color[fam], linecolor = :black, label = lbl)
    annotate!(gantt, (start_t + value(C[i])) / 2, 1.0,
              text(string(orders[i].grade, "\n#", orders[i].id), 7, :white, :center))
    due_color = value(T[i]) > 1e-6 ? :red : :black
    scatter!(gantt, [orders[i].due], [1.85], marker = :dtriangle, markersize = 6,
             color = due_color, label = "")
end

gantt_png_path = tempname() * ".png"
savefig(gantt, gantt_png_path)
gantt_b64 = base64encode(read(gantt_png_path))
rm(gantt_png_path)

# ---------------------------------------------------------------------------
# 4. KPIs and the results table
# ---------------------------------------------------------------------------

total_changeover_h = sum(s[i, j] * xval[i, j] for i in N, j in N if i != j) +
                      sum(startup_time * xval[0, j] for j in N)
total_tardy_h = sum(value.(T))
n_tardy_orders = count(i -> value(T[i]) > 1e-6, N)
makespan = maximum(value.(C))
total_cost = objective_value(model)

rows_html = IOBuffer()
for i in seq
    start_t = value(C[i]) - proc_time[i]
    tardy = value(T[i])
    tardy_str = tardy > 1e-6 ? @sprintf("%.1f h late", tardy) : "on time"
    tardy_class = tardy > 1e-6 ? " class=\"tardy\"" : ""
    println(rows_html, """
    <tr$tardy_class>
      <td>$(orders[i].id)</td>
      <td>$(orders[i].grade)</td>
      <td>$(String(family[orders[i].grade]))</td>
      <td>$(@sprintf("%.0f", orders[i].qty))</td>
      <td>$(@sprintf("%.1f", start_t))</td>
      <td>$(@sprintf("%.1f", value(C[i])))</td>
      <td>$(@sprintf("%.1f", orders[i].due))</td>
      <td>$tardy_str</td>
    </tr>""")
end

# ---------------------------------------------------------------------------
# 5. Assemble the self-contained HTML report
# ---------------------------------------------------------------------------

html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polymer Plant Scheduling — PE &amp; PP Production Campaigns</title>
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

<h1>Polymer Plant Scheduling — PE &amp; PP Production Campaigns</h1>
<p class="subtitle">Manufacturing scheduling case study &middot; Julia + JuMP + HiGHS &middot; companion to <code>notebooks/polymer_pe_pp_scheduling.jl</code></p>

<p>
This report documents a <strong>production scheduling model</strong> for a single
compounding / extrusion line that manufactures several grades of
<strong>polyethylene (PE)</strong> and <strong>polypropylene (PP)</strong>.
Switching the line between grades — and especially between the PE and PP
<strong>families</strong> — costs time and money: purging, catalyst/additive
changes, and off-spec transition material. The model below decides the order
in which to run a set of confirmed orders so as to minimize total changeover
cost plus tardiness cost. It is a mixed-integer linear program (MILP), built
with <a href="https://jump.dev">JuMP.jl</a> and solved with the open-source
<a href="https://highs.dev">HiGHS</a> solver.
</p>

<h2>1. Plant data: grades, families, and orders</h2>
<p>
The line currently has <strong>$(n) confirmed orders</strong> across
<strong>5 grades</strong>: three PE grades (HDPE, LLDPE, LDPE) and two PP
grades (homopolymer and copolymer). Each order has a quantity, a production
rate on this line, and a customer due date (in hours from the scheduling
horizon start, t&nbsp;=&nbsp;0).
</p>
<table>
<thead><tr><th>Order</th><th>Grade</th><th>Family</th><th>Qty (t)</th><th>Rate (t/h)</th><th>Due (h)</th><th>Weight</th></tr></thead>
<tbody>
$(join(["<tr><td>$(o.id)</td><td>$(o.grade)</td><td>$(String(family[o.grade]))</td><td>$(@sprintf("%.0f", o.qty))</td><td>$(@sprintf("%.1f", production_rate[o.grade]))</td><td>$(@sprintf("%.0f", o.due))</td><td>$(o.weight)</td></tr>" for o in orders]))
</tbody>
</table>

<h2>2. Sequence-dependent changeovers</h2>
<p>
Because PE and PP require different catalysts and purge procedures, the
changeover time depends on <em>what</em> the line just ran, not just on the
next grade:
</p>
<table>
<thead><tr><th>Transition</th><th>Time (h)</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Same grade repeated</td><td>0.5</td><td>housekeeping only</td></tr>
<tr><td>Same family, different grade</td><td>2.5</td><td>color/additive change, partial purge</td></tr>
<tr><td>PE &harr; PP family switch</td><td>7.0</td><td>full line/reactor purge + catalyst change</td></tr>
</tbody>
</table>
<p>Cost rates used in this run: changeover
<strong>\$$(@sprintf("%.0f", changeover_cost_per_hour))/h</strong>, tardiness
<strong>\$$(@sprintf("%.0f", tardiness_cost_per_hour))/h</strong> per unit
order weight. (The companion Pluto notebook exposes both as live sliders.)</p>

<h2>3. MILP formulation</h2>
<p>
Node 0 is a dummy "line idle" start state; nodes 1..n are the orders. Binary
variables x<sub>ij</sub> indicate that order j runs immediately after node i.
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
this "big-M timing" formulation rules out sub-tours on its own — no separate
subtour-elimination constraints (e.g. MTZ) are needed for a single open path.
With $(n) orders the model has $(sum(1 for i in N0, j in N0 if i != j && j != 0)) binary
variables and solves to global optimality with HiGHS in well under a second.
</p>

<h2>4. Optimal campaign sequence</h2>
<table>
<thead><tr><th>Order</th><th>Grade</th><th>Family</th><th>Qty (t)</th><th>Start (h)</th><th>End (h)</th><th>Due (h)</th><th>Status</th></tr></thead>
<tbody>
$(String(take!(rows_html)))
</tbody>
</table>

<h2>5. Gantt chart</h2>
<img class="gantt" src="data:image/png;base64,$(gantt_b64)" alt="Gantt chart of the optimal production campaign">
<p>
Grade blocks are colored by polymer family (<span class="legend-pe">PE</span> vs.
<span class="legend-pp">PP</span>); triangles mark each order's due date (red = missed).
The optimizer clusters same-family grades together to avoid the expensive
7-hour PE&harr;PP purge, trading a little lateness on lower-weight orders when
that is cheaper than an extra family switch.
</p>

<h2>6. Key performance indicators</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="label">Makespan</div><div class="value">$(@sprintf("%.1f", makespan)) h</div></div>
  <div class="kpi"><div class="label">Changeover time</div><div class="value">$(@sprintf("%.1f", total_changeover_h)) h</div></div>
  <div class="kpi"><div class="label">Tardiness</div><div class="value">$(@sprintf("%.1f", total_tardy_h)) h ($(n_tardy_orders) ord.)</div></div>
  <div class="kpi"><div class="label">Total cost</div><div class="value">\$$(@sprintf("%.0f", total_cost))</div></div>
</div>

<h2>7. Discussion &amp; extensions</h2>
<p>
This single-line, deterministic model already captures the dominant
economics of PE/PP campaign scheduling — family-driven changeovers and
due-date pressure — while staying small enough to solve to global optimality
in well under a second. Natural next steps for a production-grade version:
</p>
<ul>
  <li><strong>Multiple parallel lines</strong> — extend x to a 3-index x[i,j,line] and add line-capability restrictions (not every line can run every grade).</li>
  <li><strong>Minimum/maximum campaign length</strong> — avoid uneconomically short runs by bounding the number of consecutive orders of the same grade.</li>
  <li><strong>Exact grade-pair changeover matrix</strong> — replace the family-level matrix with real historical purge times per ordered grade pair.</li>
  <li><strong>Inventory &amp; storage costs</strong> — silo capacity limits often force a grade off the natural due-date order even when changeovers are cheap.</li>
  <li><strong>Rolling-horizon re-optimization</strong> — re-solve as new orders arrive or priorities change, warm-starting from the current sequence.</li>
  <li><strong>Robustness</strong> — sample uncertain processing rates / late raw-material arrivals and re-optimize, or solve a chance-constrained variant.</li>
</ul>
<p>
Because the model is expressed as a compact JuMP program, all of the above
are incremental changes to the constraints already built above, not a rewrite.
An interactive version of this same model — with live cost-weight sliders —
is available as a Pluto.jl notebook at
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
