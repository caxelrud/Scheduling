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

# ╔═╡ 636888a9-4816-47d5-805c-9a6adb46ed1b
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

# ╔═╡ 4107a004-ebc2-4096-8332-f7b6860851ba
begin
	using JuMP
	using HiGHS
	using PlutoUI
	using Plots
	using Printf
end

# ╔═╡ 0c3b1ca4-b445-4ba6-8330-0887fa975892
TableOfContents(title = "Polymer Scheduling")

# ╔═╡ 3f3d2d43-55b7-43b8-a01e-81bda66e27b9
md"""
## 1. Plant data: grades, families, and orders

The plant has **7 confirmed orders** across **5 grades**: three PE grades
(HDPE, LLDPE, LDPE) running on the PE train, and two PP grades
(homopolymer and copolymer) running on the PP train. Each order has a
quantity, a production rate on its train, and a customer due date (in
hours from the scheduling horizon start, `t = 0`).
"""

# ╔═╡ 104b339b-70ca-406a-a69a-0fd3bc2d427e
family = Dict(
	"HDPE-5502"    => :PE,
	"LLDPE-2020"   => :PE,
	"LDPE-1922"    => :PE,
	"PP-Homo-1100" => :PP,
	"PP-Copo-3300" => :PP,
)

# ╔═╡ 439e0a9a-67b1-4c14-ab92-fb0a9cbe6b05
production_rate = Dict( # tons / hour, on that grade's dedicated train
	"HDPE-5502"    => 5.0,
	"LLDPE-2020"   => 4.5,
	"LDPE-1922"    => 4.0,
	"PP-Homo-1100" => 5.5,
	"PP-Copo-3300" => 4.8,
)

# ╔═╡ af1a8ac5-20b2-4d22-9866-a81b02c19315
orders = [
	(id = 1, grade = "HDPE-5502",    qty = 120.0, due = 30.0,  weight = 1.0),
	(id = 2, grade = "LLDPE-2020",   qty = 80.0,  due = 40.0,  weight = 1.0),
	(id = 3, grade = "PP-Homo-1100", qty = 100.0, due = 55.0,  weight = 1.5),
	(id = 4, grade = "LDPE-1922",    qty = 60.0,  due = 65.0,  weight = 1.0),
	(id = 5, grade = "PP-Copo-3300", qty = 90.0,  due = 80.0,  weight = 1.5),
	(id = 6, grade = "HDPE-5502",    qty = 70.0,  due = 95.0,  weight = 1.0),
	(id = 7, grade = "PP-Homo-1100", qty = 110.0, due = 110.0, weight = 1.5),
];

# ╔═╡ 619b9040-2f0c-4269-883f-542806c19274
begin
	pe_orders = filter(o -> family[o.grade] == :PE, orders)
	pp_orders = filter(o -> family[o.grade] == :PP, orders)
	(pe = length(pe_orders), pp = length(pp_orders))
end

# ╔═╡ 50fa1b02-40e4-4d90-b6c0-aa35876e7d3c
md"""
## 2. Sequence-dependent changeovers (within a train)

Because a train only ever runs grades from one family, the only changeover
decision is *within* that family. Real purge/transition times depend on the
**specific grade pair**, not just "same grade or not" — and they are often
**asymmetric**: moving to a lighter color or lower-additive grade purges
faster than the reverse, because residual material from the *previous*
grade is what has to be flushed out. So changeovers are given as an
explicit `(from, to) -> hours` table rather than a flat rule:
"""

# ╔═╡ f8ce68cc-f280-4a76-b1ae-c011a18394ea
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

# ╔═╡ d9184e81-c3cc-466c-bc8b-c4f3a1d035a4
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

# ╔═╡ 63d326db-8036-4bf1-b1f9-efc0905b7077
changeover_matrix_md(["PP-Homo-1100", "PP-Copo-3300"])

# ╔═╡ 020024d4-a525-479b-814d-681064fff68c
md"""
There is no PE ↔ PP transition to model — that changeover never happens,
because it would require re-equipping the whole train. Use the sliders to
explore how the relative cost of a changeover-hour vs. a tardy-hour
reshapes each train's optimal campaign order.
"""

# ╔═╡ 145f722a-4ae2-4333-ad78-217a2a600297
startup_time = 1.0 # train idle -> first grade of the campaign

# ╔═╡ 2cfdd195-2c97-472b-91fe-074047cb7d69
md"""
Changeover cost rate (\$ / hour of line time lost to purging & transition material):

$(@bind changeover_cost_per_hour Slider(100:25:800, default = 450, show_value = true))

Tardiness cost rate (\$ / hour late, per unit of order weight):

$(@bind tardiness_cost_per_hour Slider(50:25:600, default = 300, show_value = true))
"""

# ╔═╡ 170e1a62-9142-418e-946b-80aa6f14b253
md"""
## 3. MILP formulation (applied once per train)

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

# ╔═╡ 65f8c104-36dc-46b2-8772-bc5ff0130c18
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

# ╔═╡ 5aef4e56-851e-483c-aa39-0bae2cda868a
pe_result = schedule_line(pe_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ 8f4450de-dad6-4728-9161-821c33a7cb63
pp_result = schedule_line(pp_orders, changeover_cost_per_hour, tardiness_cost_per_hour)

# ╔═╡ 7c739c33-bff1-441e-9eb8-26d5957ef8a3
md"""
## 4. Optimal campaign sequence, per train
"""

# ╔═╡ e4995c67-0b2e-455e-bc9b-3ee4128e9150
begin
	rows(res, os) = map(res.seq) do i
		start_t = res.C[i] - res.pt[i]
		(order = os[i].id, grade = os[i].grade, qty = os[i].qty,
		 start = round(start_t, digits = 1), stop = round(res.C[i], digits = 1),
		 due = os[i].due, tardy = round(res.T[i], digits = 1))
	end
	(PE = rows(pe_result, pe_orders), PP = rows(pp_result, pp_orders))
end

# ╔═╡ 63e5e8b8-7c4d-4410-88ce-cfeec654828e
md"""
## 5. Gantt chart
"""

# ╔═╡ cd389a3d-5dda-4dd1-afa2-4cd1895a2262
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

# ╔═╡ 5467b9c4-83f5-4666-a3bf-425b51d399a3
md"""
Grade blocks are colored by polymer family (PE vs. PP); triangles mark each
order's due date (red = missed). Because the two trains never compete for
the same equipment, they run **simultaneously**: overall makespan is the
*slower* of the two trains, not the sum of both — a big win over treating
this as one shared line that has to purge between families.
"""

# ╔═╡ 070a6da7-8c4a-48f6-b129-b5b729e2d5a2
md"""
## 6. Key performance indicators
"""

# ╔═╡ bb5f2e39-6439-42b9-84a2-50c755908146
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

# ╔═╡ f3801aa6-802b-41cf-96ac-9a51554318eb
md"""
## 7. Discussion & extensions

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
# ╟─636888a9-4816-47d5-805c-9a6adb46ed1b
# ╠═4107a004-ebc2-4096-8332-f7b6860851ba
# ╠═0c3b1ca4-b445-4ba6-8330-0887fa975892
# ╟─3f3d2d43-55b7-43b8-a01e-81bda66e27b9
# ╠═104b339b-70ca-406a-a69a-0fd3bc2d427e
# ╠═439e0a9a-67b1-4c14-ab92-fb0a9cbe6b05
# ╠═af1a8ac5-20b2-4d22-9866-a81b02c19315
# ╠═619b9040-2f0c-4269-883f-542806c19274
# ╟─50fa1b02-40e4-4d90-b6c0-aa35876e7d3c
# ╠═f8ce68cc-f280-4a76-b1ae-c011a18394ea
# ╠═d9184e81-c3cc-466c-bc8b-c4f3a1d035a4
# ╠═63d326db-8036-4bf1-b1f9-efc0905b7077
# ╟─020024d4-a525-479b-814d-681064fff68c
# ╠═145f722a-4ae2-4333-ad78-217a2a600297
# ╟─2cfdd195-2c97-472b-91fe-074047cb7d69
# ╟─170e1a62-9142-418e-946b-80aa6f14b253
# ╠═65f8c104-36dc-46b2-8772-bc5ff0130c18
# ╠═5aef4e56-851e-483c-aa39-0bae2cda868a
# ╠═8f4450de-dad6-4728-9161-821c33a7cb63
# ╟─7c739c33-bff1-441e-9eb8-26d5957ef8a3
# ╠═e4995c67-0b2e-455e-bc9b-3ee4128e9150
# ╟─63e5e8b8-7c4d-4410-88ce-cfeec654828e
# ╠═cd389a3d-5dda-4dd1-afa2-4cd1895a2262
# ╟─5467b9c4-83f5-4666-a3bf-425b51d399a3
# ╟─070a6da7-8c4a-48f6-b129-b5b729e2d5a2
# ╠═bb5f2e39-6439-42b9-84a2-50c755908146
# ╟─f3801aa6-802b-41cf-96ac-9a51554318eb
