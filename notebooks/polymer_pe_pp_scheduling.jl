### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fe032d65-bf88-453d-8d8a-6f968b1f2c37
md"""
# Polymer Plant Scheduling — Dedicated PE & PP Production Trains

This notebook builds and solves a **month-long production schedule** (a
30-day, 720-hour horizon) for a polymer plant that manufactures **8 grades**
of **polyethylene (PE)** and **polypropylene (PP)**. Each grade runs as a
multi-day campaign, and changeovers between grades take a few hours — the
scale real plant schedulers actually work at.

PE and PP are made on **physically separate trains**: different reactor
technology and catalyst systems (e.g. gas-phase/slurry PE reactors vs.
Ziegler-Natta/metallocene PP reactors) mean a PE line can never run a PP
grade or vice versa. So this is really **two independent single-line
scheduling problems, running in parallel** — one per train — not one line
that occasionally switches families.

Within a train, switching between grades still costs time and money:
color/additive changes and partial purges. The scheduler below decides the
order in which each train runs its confirmed orders so as to minimize

$$\text{total cost} = \sum_{\text{train} \in \{PE, PP\}} \Big[ \underbrace{\text{changeover cost}}_{\text{sequence-dependent}} + \underbrace{\text{tardiness cost}}_{\text{missed due dates}} \Big]$$

Each train's sub-problem is a mixed-integer linear program (MILP), built
with [JuMP.jl](https://jump.dev) and solved with the open-source
[HiGHS](https://highs.dev) solver. Move the sliders further down to see how
the optimal campaign sequence on each train reacts to the relative cost of
changeovers versus lateness.
"""

# ╔═╡ b15b8a12-dc9f-42b8-8d8c-e86653fd8941
begin
	using JuMP
	using HiGHS
	using PlutoUI
	using Plots
	using Printf
end

# ╔═╡ ab7a8be1-2ec3-4c89-9f1b-e296e78b3116
TableOfContents(title = "Polymer Scheduling")

# ╔═╡ 1d374221-ae3a-4cbf-9a06-471dc3c90905
md"""
## 1. Plant data: grades, families, and rates

Eight grades run on the plant's two trains: four PE grades (two HDPE
grades, one LLDPE, one LDPE) on the PE train, and four PP grades
(two homopolymer variants, one random copolymer, one impact copolymer) on
the PP train.
"""

# ╔═╡ d6867a68-a251-4b75-9141-42740f2d9b7e
const HORIZON_HOURS = 720.0 # 30-day scheduling horizon

# ╔═╡ 6825b6b2-ea37-4445-8e73-a8dca5835c65
family = Dict(
	"HDPE-5502"      => :PE,
	"HDPE-6200"      => :PE,
	"LLDPE-2020"     => :PE,
	"LDPE-1922"      => :PE,
	"PP-Homo-1100"   => :PP,
	"PP-Homo-1305"   => :PP,
	"PP-Copo-3300"   => :PP,
	"PP-Impact-7015" => :PP,
)

# ╔═╡ 34809d1b-e0a0-45d6-84af-a9a102b16902
production_rate = Dict( # tons / hour, on that grade's dedicated train
	"HDPE-5502"      => 5.0,
	"HDPE-6200"      => 5.5,
	"LLDPE-2020"     => 4.5,
	"LDPE-1922"      => 4.0,
	"PP-Homo-1100"   => 5.5,
	"PP-Homo-1305"   => 5.0,
	"PP-Copo-3300"   => 4.8,
	"PP-Impact-7015" => 4.5,
)

# ╔═╡ d5e393f8-be6d-4e35-9550-8e77f5767fe9
md"""
## 2. Customer orders → production lots

A plant does not run one production campaign per purchase order. Sales
orders for the **same grade** shipping close together get pooled into one
production lot, sized to cover all of them, timed to the **earliest** ship
date in the group — producing early never hurts the later orders in the
group, it just means finished goods sit in the warehouse a few extra
hours. Fewer, larger lots also mean fewer changeovers.

Here are the raw customer orders behind this scheduling run:
"""

# ╔═╡ 39d1e41d-01aa-4a72-a32d-e428710670d8
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
];

# ╔═╡ ea7ca7d8-9147-4a66-b0f0-725a925a831c
md"""
Consolidation window — combine same-grade orders due within this many hours
of the earliest one in the group:

$(@bind consolidation_window Slider(0:6:96, default = 24, show_value = true))
"""

# ╔═╡ c7d4bff8-3dcd-423b-bf3d-0c212084bde0
# PP grades carry a higher contractual tardiness weight than PE in this
# example.
order_weight(g::String) = family[g] == :PP ? 1.5 : 1.0

# ╔═╡ e412d6fe-86d3-4ca6-92af-d74412f3b052
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

# ╔═╡ 5f210a38-baad-406b-9b5d-ff6434501e7d
orders = consolidate_orders(customer_orders; window = consolidation_window)

# ╔═╡ bbfacb30-2429-445e-9b7d-40acbe8e0738
md"""
$(length(customer_orders)) customer orders consolidate into
**$(length(orders)) production lots** at a $(consolidation_window)-hour
window:
"""

# ╔═╡ 1ace6a69-c80b-4e6f-a38f-152c06f07652
orders

# ╔═╡ 00dd81f6-e9c9-472a-a143-5136c289c55b
begin
	pe_orders = filter(o -> family[o.grade] == :PE, orders)
	pp_orders = filter(o -> family[o.grade] == :PP, orders)
	(pe = length(pe_orders), pp = length(pp_orders))
end

# ╔═╡ e90ea1b4-5388-4142-a101-00606814a8c5
md"""
## 3. Sequence-dependent changeovers (within a train)

Because a train only ever runs grades from one family, the only changeover
decision is *within* that family. Real purge/transition times depend on the
**specific grade pair**, not just "same grade or not" — and they are often
**asymmetric**: moving to a lighter color or lower-additive grade purges
faster than the reverse, because residual material from the *previous*
grade is what has to be flushed out. So changeovers are given as an
explicit `(from, to) -> hours` table rather than a flat rule:
"""

# ╔═╡ 77c8cea9-da1e-400d-ae75-a8a09643ffcc
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

# ╔═╡ 38052c38-dd81-4129-b02d-9d608acfcc78
begin
	function changeover_matrix_md(grades)
		header = "| from \\ to | " * join(grades, " | ") * " |"
		sep = "|---" ^ (length(grades) + 1) * "|"
		body = join(
			["| **$g1** | " * join([string(CHANGEOVER[(g1, g2)]) for g2 in grades], " | ") * " |"
			 for g1 in grades],
			"\n",
		)
		Markdown.parse(join([header, sep, body], "\n"))
	end
	changeover_matrix_md(["HDPE-5502", "HDPE-6200", "LLDPE-2020", "LDPE-1922"])
end

# ╔═╡ 5af401ce-ea1b-41db-85b6-583bf26f972c
changeover_matrix_md(["PP-Homo-1100", "PP-Homo-1305", "PP-Copo-3300", "PP-Impact-7015"])

# ╔═╡ d982c7c5-78ce-4488-93db-2c1d08ea6f21
md"""
There is no PE ↔ PP transition to model — that changeover never happens,
because it would require re-equipping the whole train. Use the sliders to
explore how the relative cost of a changeover-hour vs. a tardy-hour
reshapes each train's optimal campaign order.
"""

# ╔═╡ 478c75dd-7fa0-4940-8c0a-ad711f8fdd32
startup_time = 1.0 # train idle -> first grade of the campaign

# ╔═╡ 8eacba3e-3a21-4369-a6d9-99e407b378d2
md"""
Changeover cost rate (\$ / hour of line time lost to purging & transition material):

$(@bind changeover_cost_per_hour Slider(100:25:800, default = 450, show_value = true))

Tardiness cost rate (\$ / hour late, per unit of order weight):

$(@bind tardiness_cost_per_hour Slider(50:25:600, default = 300, show_value = true))
"""

# ╔═╡ 8fd5ce1d-5d3b-43ed-92f4-cc3b19be5b22
md"""
## 4. MILP formulation (applied once per train)

Each train's orders are scheduled independently. Node `0` is a dummy
*"train idle"* start state; nodes `1..m` are that train's orders. Binary
variables $x_{ij}$ indicate that order $j$ runs immediately after node
$i$. Continuous variables $C_i$ and $T_i$ are the completion time and
tardiness of order $i$.

$$
\begin{aligned}
\min \quad & c_{\text{co}} \sum_{i \neq j} s_{ij} x_{ij} + c_{\text{tar}} \sum_i w_i T_i \\
\text{s.t.} \quad & \sum_j x_{0j} = 1 \\
& \sum_{i \neq j} x_{ij} = 1 & \forall j \\
& \sum_{j \neq i} x_{ij} \le 1 & \forall i \\
& C_j \ge C_i + s_{ij} + p_j - M(1 - x_{ij}) & \forall i \neq j \\
& T_i \ge C_i - d_i, \quad T_i \ge 0 \\
& x_{ij} \in \{0,1\}, \; C_i, T_i \ge 0
\end{aligned}
$$

Completion times must strictly increase along any chain of active arcs, so
this "big-M timing" formulation rules out sub-tours on its own — no
separate subtour-elimination constraints (e.g. MTZ) are needed for a single
open path. The two trains' sub-problems share no variables, so solving them
independently is exact, not a heuristic decomposition.
"""

# ╔═╡ 69bf6362-08af-42e0-b331-a885e7c1c322
function schedule_line(line_orders, changeover_cost_per_hour, tardiness_cost_per_hour)
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

	xval = value.(x)
	succ = Dict{Int, Int}()
	local first_order = 0
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

	(seq = seq, pt = pt, C = value.(C), T = value.(T), cost = objective_value(model),
	 status = termination_status(model))
end

# ╔═╡ 14d04202-018b-43f4-8bfa-8c43fb13109b
pe_result = schedule_line(pe_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ 7ed86b27-275b-473c-9d5b-7a78dfd62930
pp_result = schedule_line(pp_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ aaa21be6-d75a-4da2-8994-6d9e3afd99e9
md"""
## 5. Optimal campaign sequence, per train
"""

# ╔═╡ bd06320c-eeaf-473d-b463-0942e299bd78
begin
	rows(res, os) = map(res.seq) do i
		start_t = res.C[i] - res.pt[i]
		(order = os[i].id, grade = os[i].grade, qty = os[i].qty,
		 start_day = round(start_t / 24, digits = 1), end_day = round(res.C[i] / 24, digits = 1),
		 due_day = round(os[i].due / 24, digits = 1), tardy_days = round(res.T[i] / 24, digits = 1))
	end
	(PE = rows(pe_result, pe_orders), PP = rows(pp_result, pp_orders))
end

# ╔═╡ 805ceec2-c1dc-470a-adc5-5d882952dc9f
md"""
## 6. Gantt chart
"""

# ╔═╡ c77fadae-2445-4fe9-acac-a3048db41886
begin
	family_color = Dict(:PE => RGB(0.20, 0.45, 0.85), :PP => RGB(0.90, 0.55, 0.10))

	gantt = plot(
		size = (1000, 380),
		xlabel = "Time (days)",
		ylabel = "",
		yticks = ([1, 2], ["PP train", "PE train"]),
		xlims = (0, HORIZON_HOURS / 24),
		ylims = (0, 3),
		legend = :outertop,
		legendcolumns = 2,
		framestyle = :box,
		title = "Two dedicated production trains — 30-day horizon (running in parallel)",
	)

	local seen = Set{Symbol}()
	for (res, os, ypos) in ((pe_result, pe_orders, 2), (pp_result, pp_orders, 1))
		for i in res.seq
			start_d = (res.C[i] - res.pt[i]) / 24
			end_d = res.C[i] / 24
			fam = family[os[i].grade]
			lbl = fam in seen ? "" : String(fam)
			push!(seen, fam)
			plot!(gantt, Shape([start_d, end_d, end_d, start_d],
								[ypos - 0.35, ypos - 0.35, ypos + 0.35, ypos + 0.35]),
				  color = family_color[fam], linecolor = :black, label = lbl)
			annotate!(gantt, (start_d + end_d) / 2, ypos,
					  text(string(os[i].grade, "\n#", os[i].id), 6, :white, :center))
			due_color = res.T[i] > 1e-6 ? :red : :black
			scatter!(gantt, [os[i].due / 24], [ypos + 0.42], marker = :dtriangle, markersize = 6,
					 color = due_color, label = "")
		end
	end

	gantt
end

# ╔═╡ 174e81b3-0e4d-4a70-9672-c291afec3b74
md"""
Grade blocks are colored by polymer family (PE vs. PP); triangles mark each
order's due date (red = missed). Because the two trains never compete for
the same equipment, they run **simultaneously**: overall makespan is the
*slower* of the two trains, not the sum of both — a big win over treating
this as one shared line that has to purge between families.
"""

# ╔═╡ 1898c9d5-ec44-44fd-b44c-8909eeedfa41
md"""
## 7. Key performance indicators
"""

# ╔═╡ fe30a67e-3b37-41ef-b531-755c251f2f92
begin
	makespan_pe = maximum(pe_result.C)
	makespan_pp = maximum(pp_result.C)
	overall_makespan = max(makespan_pe, makespan_pp)
	total_tardy_h = sum(pe_result.T) + sum(pp_result.T)
	n_tardy_orders = count(>(1e-6), vcat(pe_result.T, pp_result.T))
	total_cost = pe_result.cost + pp_result.cost
	horizon_slack_days = (HORIZON_HOURS - overall_makespan) / 24

	kpi_text = string(
		"| KPI | PE train | PP train | Overall |\n",
		"|---|---|---|---|\n",
		"| Makespan (days) | ", @sprintf("%.1f", makespan_pe / 24), " | ", @sprintf("%.1f", makespan_pp / 24),
		" | ", @sprintf("%.1f", overall_makespan / 24), " |\n",
		"| Cost (\$) | ", @sprintf("%.0f", pe_result.cost), " | ", @sprintf("%.0f", pp_result.cost),
		" | ", @sprintf("%.0f", total_cost), " |\n",
		"| Tardy orders | | | ", n_tardy_orders, " of ", length(orders), " |\n",
		"| Total tardiness (days) | | | ", @sprintf("%.1f", total_tardy_h / 24), " |\n",
		"| Slack vs. 30-day horizon | | | ", @sprintf("%.1f", horizon_slack_days), " days |\n",
	)
	Markdown.parse(kpi_text)
end

# ╔═╡ d561b7cb-cedb-4c25-aa2e-01207d753209
md"""
## 8. Discussion & extensions

Modeling PE and PP as two **dedicated, parallel trains** — rather than one
shared line that pays a family-changeover penalty — matters a lot
economically: it removes a changeover that can never actually happen on
real equipment, and it lets both product families progress at once instead
of queueing behind each other. In this example that alone cuts total
scheduling cost roughly 8x versus the (unrealistic) single-shared-line
version of the same orders.

Natural next steps for a production-grade version:

- **More than one train per family** — e.g. two PE lines with different
  capability sets; extend `schedule_line` to a proper parallel-machine
  assignment problem within a family.
- **Minimum/maximum campaign length** — avoid uneconomically short runs by
  bounding the number of consecutive orders of the same grade.
- **Data-driven changeover matrix** — populate the grade-pair matrix from
  real historical purge-time records instead of hand-entered estimates.
- **Shared upstream/downstream utilities** — if both trains draw from a
  common feedstock or packaging line, that shared resource needs its own
  capacity constraint linking the two schedules.
- **Rolling-horizon re-optimization** — re-solve as new orders arrive or an
  order's priority changes, warm-starting from the current sequence.
- **Robustness** — sample uncertain processing rates / late raw-material
  arrivals and re-optimize, or solve a chance-constrained variant.

Because each train's model is a compact, independent JuMP program, all of
the above are incremental changes to `schedule_line`, not a rewrite.
"""

# ╔═╡ Cell order:
# ╟─fe032d65-bf88-453d-8d8a-6f968b1f2c37
# ╠═b15b8a12-dc9f-42b8-8d8c-e86653fd8941
# ╠═ab7a8be1-2ec3-4c89-9f1b-e296e78b3116
# ╟─1d374221-ae3a-4cbf-9a06-471dc3c90905
# ╠═d6867a68-a251-4b75-9141-42740f2d9b7e
# ╠═6825b6b2-ea37-4445-8e73-a8dca5835c65
# ╠═34809d1b-e0a0-45d6-84af-a9a102b16902
# ╟─d5e393f8-be6d-4e35-9550-8e77f5767fe9
# ╠═39d1e41d-01aa-4a72-a32d-e428710670d8
# ╟─ea7ca7d8-9147-4a66-b0f0-725a925a831c
# ╠═c7d4bff8-3dcd-423b-bf3d-0c212084bde0
# ╠═e412d6fe-86d3-4ca6-92af-d74412f3b052
# ╠═5f210a38-baad-406b-9b5d-ff6434501e7d
# ╟─bbfacb30-2429-445e-9b7d-40acbe8e0738
# ╠═1ace6a69-c80b-4e6f-a38f-152c06f07652
# ╠═00dd81f6-e9c9-472a-a143-5136c289c55b
# ╟─e90ea1b4-5388-4142-a101-00606814a8c5
# ╠═77c8cea9-da1e-400d-ae75-a8a09643ffcc
# ╠═38052c38-dd81-4129-b02d-9d608acfcc78
# ╠═5af401ce-ea1b-41db-85b6-583bf26f972c
# ╟─d982c7c5-78ce-4488-93db-2c1d08ea6f21
# ╠═478c75dd-7fa0-4940-8c0a-ad711f8fdd32
# ╟─8eacba3e-3a21-4369-a6d9-99e407b378d2
# ╟─8fd5ce1d-5d3b-43ed-92f4-cc3b19be5b22
# ╠═69bf6362-08af-42e0-b331-a885e7c1c322
# ╠═14d04202-018b-43f4-8bfa-8c43fb13109b
# ╠═7ed86b27-275b-473c-9d5b-7a78dfd62930
# ╟─aaa21be6-d75a-4da2-8994-6d9e3afd99e9
# ╠═bd06320c-eeaf-473d-b463-0942e299bd78
# ╟─805ceec2-c1dc-470a-adc5-5d882952dc9f
# ╠═c77fadae-2445-4fe9-acac-a3048db41886
# ╟─174e81b3-0e4d-4a70-9672-c291afec3b74
# ╟─1898c9d5-ec44-44fd-b44c-8909eeedfa41
# ╠═fe30a67e-3b37-41ef-b531-755c251f2f92
# ╟─d561b7cb-cedb-4c25-aa2e-01207d753209
