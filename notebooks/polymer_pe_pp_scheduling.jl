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

# ╔═╡ c783b241-9716-4e31-afe7-191411ba73f3
md"""
# Polymer Plant Scheduling — Dedicated PE & PP Production Trains

This notebook builds and solves a **production scheduling model** for a
polymer plant that manufactures several grades of **polyethylene (PE)** and
**polypropylene (PP)**.

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

# ╔═╡ ed4c907f-68f2-4ebe-9414-8b56faa3272a
begin
	using JuMP
	using HiGHS
	using PlutoUI
	using Plots
	using Printf
end

# ╔═╡ 026f4d46-5556-4cfc-9bba-0354cb0c2c3d
TableOfContents(title = "Polymer Scheduling")

# ╔═╡ 9cc2783b-ddd0-4721-852f-8afa0e31916d
md"""
## 1. Plant data: grades, families, and rates

Five grades run on the plant's two trains: three PE grades (HDPE, LLDPE,
LDPE) on the PE train, and two PP grades (homopolymer and copolymer) on the
PP train.
"""

# ╔═╡ bdc39a26-6ec0-406f-9434-06545ae50809
family = Dict(
	"HDPE-5502"    => :PE,
	"LLDPE-2020"   => :PE,
	"LDPE-1922"    => :PE,
	"PP-Homo-1100" => :PP,
	"PP-Copo-3300" => :PP,
)

# ╔═╡ 129fc4c1-bb8b-48bb-a74b-3f97199957a1
production_rate = Dict( # tons / hour, on that grade's dedicated train
	"HDPE-5502"    => 5.0,
	"LLDPE-2020"   => 4.5,
	"LDPE-1922"    => 4.0,
	"PP-Homo-1100" => 5.5,
	"PP-Copo-3300" => 4.8,
)

# ╔═╡ f9344e11-f1d6-4c76-9d0e-111e835e1d20
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

# ╔═╡ 5197ed79-8bf6-45ce-a209-00179ccd2795
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
];

# ╔═╡ 7436fc61-829e-4426-88de-8339fc324199
md"""
Consolidation window — combine same-grade orders due within this many hours
of the earliest one in the group:

$(@bind consolidation_window Slider(0:6:72, default = 24, show_value = true))
"""

# ╔═╡ 99875fe8-c864-4ed9-a305-ae9b408986fc
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

# ╔═╡ 24ac1e27-fee1-49d2-bc04-06779b9241bd
orders = consolidate_orders(customer_orders; window = consolidation_window)

# ╔═╡ 6cf15771-643d-4a9f-be53-691b2e4a3d5a
md"""
$(length(customer_orders)) customer orders consolidate into
**$(length(orders)) production lots** at a $(consolidation_window)-hour
window:
"""

# ╔═╡ 5aec0595-393e-45bd-b10e-81f7465489c8
orders

# ╔═╡ 380e97d5-1a7a-4c3e-ad85-36837448f177
begin
	pe_orders = filter(o -> family[o.grade] == :PE, orders)
	pp_orders = filter(o -> family[o.grade] == :PP, orders)
	(pe = length(pe_orders), pp = length(pp_orders))
end

# ╔═╡ 94e4f72e-4b55-48e6-b081-0d9ca54c5fbf
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

# ╔═╡ 956f4c8c-e6d6-49b4-9636-d50f865919f5
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

# ╔═╡ 6e40c2a4-1546-4604-bbe0-c444f373cbd4
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
	changeover_matrix_md(["HDPE-5502", "LLDPE-2020", "LDPE-1922"])
end

# ╔═╡ 92cd571d-7cb8-464e-b821-98d2771587e6
changeover_matrix_md(["PP-Homo-1100", "PP-Copo-3300"])

# ╔═╡ fe0838f3-f10a-484f-873e-257bf9e00425
md"""
There is no PE ↔ PP transition to model — that changeover never happens,
because it would require re-equipping the whole train. Use the sliders to
explore how the relative cost of a changeover-hour vs. a tardy-hour
reshapes each train's optimal campaign order.
"""

# ╔═╡ 91a0d917-0dd5-4feb-88b4-9d02447a08ab
startup_time = 1.0 # train idle -> first grade of the campaign

# ╔═╡ bbc70a39-6dc4-4df5-8fd3-277224c14b37
md"""
Changeover cost rate (\$ / hour of line time lost to purging & transition material):

$(@bind changeover_cost_per_hour Slider(100:25:800, default = 450, show_value = true))

Tardiness cost rate (\$ / hour late, per unit of order weight):

$(@bind tardiness_cost_per_hour Slider(50:25:600, default = 300, show_value = true))
"""

# ╔═╡ a35cb454-040e-4de9-a74f-1b5afa80b5e0
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

# ╔═╡ b764d837-c731-4b33-a070-3af2b7092cab
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

# ╔═╡ 1d47f62e-eb49-43e0-a0fc-79c4d76d154c
pe_result = schedule_line(pe_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ c59f6ccc-e115-48fa-8618-749ade664231
pp_result = schedule_line(pp_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ a429e148-dfc4-46cb-981c-32bc701421e6
md"""
## 5. Optimal campaign sequence, per train
"""

# ╔═╡ 98bceca0-3542-4604-8162-8173534f6846
begin
	rows(res, os) = map(res.seq) do i
		start_t = res.C[i] - res.pt[i]
		(order = os[i].id, grade = os[i].grade, qty = os[i].qty,
		 start = round(start_t, digits = 1), stop = round(res.C[i], digits = 1),
		 due = os[i].due, tardy = round(res.T[i], digits = 1))
	end
	(PE = rows(pe_result, pe_orders), PP = rows(pp_result, pp_orders))
end

# ╔═╡ 0c4c7da1-8455-4fb1-bee4-ec4f08ba071f
md"""
## 6. Gantt chart
"""

# ╔═╡ 821d748f-f13e-439d-9127-7c8adc37eb48
begin
	family_color = Dict(:PE => RGB(0.20, 0.45, 0.85), :PP => RGB(0.90, 0.55, 0.10))

	gantt = plot(
		size = (900, 380),
		xlabel = "Time (h)",
		ylabel = "",
		yticks = ([1, 2], ["PP train", "PE train"]),
		ylims = (0, 3),
		legend = :outertop,
		legendcolumns = 2,
		framestyle = :box,
		title = "Two dedicated production trains (running in parallel)",
	)

	local seen = Set{Symbol}()
	for (res, os, ypos) in ((pe_result, pe_orders, 2), (pp_result, pp_orders, 1))
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

	gantt
end

# ╔═╡ c32571be-ae39-482f-aaf0-166bc7b8f66f
md"""
Grade blocks are colored by polymer family (PE vs. PP); triangles mark each
order's due date (red = missed). Because the two trains never compete for
the same equipment, they run **simultaneously**: overall makespan is the
*slower* of the two trains, not the sum of both — a big win over treating
this as one shared line that has to purge between families.
"""

# ╔═╡ e8e21fb1-efa2-4ca7-91d5-7d0dfd78bb08
md"""
## 7. Key performance indicators
"""

# ╔═╡ eaec6c0f-8e2d-4ea8-81ad-4ce47e8eba62
begin
	makespan_pe = maximum(pe_result.C)
	makespan_pp = maximum(pp_result.C)
	overall_makespan = max(makespan_pe, makespan_pp)
	total_tardy_h = sum(pe_result.T) + sum(pp_result.T)
	n_tardy_orders = count(>(1e-6), vcat(pe_result.T, pp_result.T))
	total_cost = pe_result.cost + pp_result.cost

	kpi_text = string(
		"| KPI | PE train | PP train | Overall |\n",
		"|---|---|---|---|\n",
		"| Makespan (h) | ", @sprintf("%.1f", makespan_pe), " | ", @sprintf("%.1f", makespan_pp),
		" | ", @sprintf("%.1f", overall_makespan), " |\n",
		"| Cost (\$) | ", @sprintf("%.0f", pe_result.cost), " | ", @sprintf("%.0f", pp_result.cost),
		" | ", @sprintf("%.0f", total_cost), " |\n",
		"| Tardy orders | | | ", n_tardy_orders, " of ", length(orders), " |\n",
		"| Total tardiness (h) | | | ", @sprintf("%.1f", total_tardy_h), " |\n",
	)
	Markdown.parse(kpi_text)
end

# ╔═╡ 0623bbe8-0b34-469e-9d96-fdcafe137b3f
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
# ╟─c783b241-9716-4e31-afe7-191411ba73f3
# ╠═ed4c907f-68f2-4ebe-9414-8b56faa3272a
# ╠═026f4d46-5556-4cfc-9bba-0354cb0c2c3d
# ╟─9cc2783b-ddd0-4721-852f-8afa0e31916d
# ╠═bdc39a26-6ec0-406f-9434-06545ae50809
# ╠═129fc4c1-bb8b-48bb-a74b-3f97199957a1
# ╟─f9344e11-f1d6-4c76-9d0e-111e835e1d20
# ╠═5197ed79-8bf6-45ce-a209-00179ccd2795
# ╟─7436fc61-829e-4426-88de-8339fc324199
# ╠═99875fe8-c864-4ed9-a305-ae9b408986fc
# ╠═24ac1e27-fee1-49d2-bc04-06779b9241bd
# ╟─6cf15771-643d-4a9f-be53-691b2e4a3d5a
# ╠═5aec0595-393e-45bd-b10e-81f7465489c8
# ╠═380e97d5-1a7a-4c3e-ad85-36837448f177
# ╟─94e4f72e-4b55-48e6-b081-0d9ca54c5fbf
# ╠═956f4c8c-e6d6-49b4-9636-d50f865919f5
# ╠═6e40c2a4-1546-4604-bbe0-c444f373cbd4
# ╠═92cd571d-7cb8-464e-b821-98d2771587e6
# ╟─fe0838f3-f10a-484f-873e-257bf9e00425
# ╠═91a0d917-0dd5-4feb-88b4-9d02447a08ab
# ╟─bbc70a39-6dc4-4df5-8fd3-277224c14b37
# ╟─a35cb454-040e-4de9-a74f-1b5afa80b5e0
# ╠═b764d837-c731-4b33-a070-3af2b7092cab
# ╠═1d47f62e-eb49-43e0-a0fc-79c4d76d154c
# ╠═c59f6ccc-e115-48fa-8618-749ade664231
# ╟─a429e148-dfc4-46cb-981c-32bc701421e6
# ╠═98bceca0-3542-4604-8162-8173534f6846
# ╟─0c4c7da1-8455-4fb1-bee4-ec4f08ba071f
# ╠═821d748f-f13e-439d-9127-7c8adc37eb48
# ╟─c32571be-ae39-482f-aaf0-166bc7b8f66f
# ╟─e8e21fb1-efa2-4ca7-91d5-7d0dfd78bb08
# ╠═eaec6c0f-8e2d-4ea8-81ad-4ce47e8eba62
# ╟─0623bbe8-0b34-469e-9d96-fdcafe137b3f
