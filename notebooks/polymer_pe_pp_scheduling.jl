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

# ╔═╡ 46cd7256-ad7a-41da-a6d3-952823f9dc39
md"""
# Polymer Plant Scheduling — PE & PP Production Campaigns

This notebook builds and solves a **production scheduling model** for a
single compounding / extrusion line that manufactures several grades of
**polyethylene (PE)** and **polypropylene (PP)**.

Switching the line between grades — and especially between the PE and PP
**families** — costs time and money: purging, catalyst/additive changes,
and off-spec transition material. The scheduler below decides the order in
which to run a set of confirmed orders so as to minimize

$$\text{total cost} = \underbrace{\text{changeover cost}}_{\text{sequence-dependent}} + \underbrace{\text{tardiness cost}}_{\text{missed due dates}}$$

The model is a mixed-integer linear program (MILP), built with
[JuMP.jl](https://jump.dev) and solved with the open-source
[HiGHS](https://highs.dev) solver. Move the sliders further down to see how
the optimal campaign sequence reacts to the relative cost of changeovers
versus lateness.
"""

# ╔═╡ 6b84d682-60c4-4fc4-a098-e5f0a4bccc24
begin
	using JuMP
	using HiGHS
	using PlutoUI
	using Plots
	using Printf
end

# ╔═╡ 236f1801-e89b-4a47-877b-acac88c8fa63
TableOfContents(title = "Polymer Scheduling")

# ╔═╡ da954f7e-5afd-4fbb-acb6-05746a55d110
md"""
## 1. Plant data: grades, families, and orders

The line currently has **7 confirmed orders** across **5 grades**: three PE
grades (HDPE, LLDPE, LDPE) and two PP grades (homopolymer and copolymer).
Each order has a quantity, a production rate on this line, and a customer
due date (in hours from the scheduling horizon start, `t = 0`).
"""

# ╔═╡ 16bc81ee-27e1-46fc-8354-3b7774557bff
family = Dict(
	"HDPE-5502"    => :PE,
	"LLDPE-2020"   => :PE,
	"LDPE-1922"    => :PE,
	"PP-Homo-1100" => :PP,
	"PP-Copo-3300" => :PP,
)

# ╔═╡ 29f0aa3d-9951-43ca-988a-9625653d5ecb
production_rate = Dict( # tons / hour, on this line
	"HDPE-5502"    => 5.0,
	"LLDPE-2020"   => 4.5,
	"LDPE-1922"    => 4.0,
	"PP-Homo-1100" => 5.5,
	"PP-Copo-3300" => 4.8,
)

# ╔═╡ 662ecaf1-5a61-4aad-ab89-788834401db8
orders = [
	(id = 1, grade = "HDPE-5502",    qty = 120.0, due = 30.0,  weight = 1.0),
	(id = 2, grade = "LLDPE-2020",   qty = 80.0,  due = 40.0,  weight = 1.0),
	(id = 3, grade = "PP-Homo-1100", qty = 100.0, due = 55.0,  weight = 1.5),
	(id = 4, grade = "LDPE-1922",    qty = 60.0,  due = 65.0,  weight = 1.0),
	(id = 5, grade = "PP-Copo-3300", qty = 90.0,  due = 80.0,  weight = 1.5),
	(id = 6, grade = "HDPE-5502",    qty = 70.0,  due = 95.0,  weight = 1.0),
	(id = 7, grade = "PP-Homo-1100", qty = 110.0, due = 110.0, weight = 1.5),
];

# ╔═╡ f6e6faca-f942-4809-b7c4-6bfd20cbb0c1
n = length(orders)

proc_time = [orders[i].qty / production_rate[orders[i].grade] for i in 1:n]

# ╔═╡ 370b8e93-37f0-434e-a7ba-4cc893dfb700
md"""
## 2. Sequence-dependent changeovers

Because PE and PP require different catalysts and purge procedures, the
changeover time depends on *what* the line just ran, not just on the next
grade:

| Transition | Time (h) | Why |
|---|---|---|
| Same grade repeated | 0.5 | housekeeping only |
| Same family, different grade | 2.5 | color/additive change, partial purge |
| PE ↔ PP family switch | 7.0 | full line/reactor purge + catalyst change |

Use the sliders to explore how the relative cost of a changeover-hour vs. a
tardy-hour reshapes the optimal campaign order.
"""

# ╔═╡ c2bc08ff-bb90-4c8b-9848-7446fb6280b0
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

startup_time = 1.0 # line idle -> first grade of the campaign

# ╔═╡ 44fa3759-362f-4f1d-a213-2d608b13d378
md"""
Changeover cost rate (\$ / hour of line time lost to purging & transition material):

$(@bind changeover_cost_per_hour Slider(100:25:800, default = 450, show_value = true))

Tardiness cost rate (\$ / hour late, per unit of order weight):

$(@bind tardiness_cost_per_hour Slider(50:25:600, default = 300, show_value = true))
"""

# ╔═╡ e499d937-6f36-4a62-8b9b-f0fb2954c864
md"""
## 3. MILP formulation

Node `0` is a dummy *"line idle"* start state; nodes `1..n` are the orders.
Binary variables $x_{ij}$ indicate that order $j$ runs immediately after
node $i$. Continuous variables $C_i$ and $T_i$ are the completion time
and tardiness of order $i$.

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
open path.
"""

# ╔═╡ 1139f9f0-b225-4ea6-bf19-ee25c9102584
begin
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

	termination_status(model)
end

# ╔═╡ c0c1e0de-1302-4ca3-b6db-337f42fd730d
md"""
## 4. Optimal campaign sequence
"""

# ╔═╡ 0072ae1f-ef79-4d16-8b0d-8252c97adca7
begin
	xval = value.(x)
	succ = Dict{Int, Int}()
	local first_order = 0
	for j in N
		if xval[0, j] > 0.5
			first_order = j
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
	seq
end

# ╔═╡ 86e678c7-e4ea-4ca3-aff3-127e96cde7d3
begin
	rows = map(seq) do i
		start_t = value(C[i]) - proc_time[i]
		(order = orders[i].id, grade = orders[i].grade, qty = orders[i].qty,
		 start = round(start_t, digits = 1), stop = round(value(C[i]), digits = 1),
		 due = orders[i].due, tardy = round(value(T[i]), digits = 1))
	end
	rows
end

# ╔═╡ ab2e8453-2737-489b-a1a1-6466f3b9f9c9
md"""
## 5. Gantt chart
"""

# ╔═╡ 64e3b564-98cb-430d-bcdf-ab36deb218ca
begin
	family_color = Dict(:PE => RGB(0.20, 0.45, 0.85), :PP => RGB(0.90, 0.55, 0.10))

	gantt = plot(
		size = (900, 340),
		xlabel = "Time (h)",
		ylabel = "",
		yticks = :none,
		ylims = (0, 2),
		legend = :outertop,
		legendcolumns = 2,
		framestyle = :box,
		title = "Single-line production campaign",
	)

	local seen = Set{Symbol}()
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

	gantt
end

# ╔═╡ 1bb67a1c-d173-4e50-bae7-bae5f0d655fc
md"""
Grade blocks are colored by polymer family (PE vs. PP); triangles mark each
order's due date (red = missed). Notice how the optimizer clusters
same-family grades together to avoid the expensive 7-hour PE↔PP purge,
trading a little lateness on lower-weight orders when that is cheaper than
an extra family switch.
"""

# ╔═╡ 8c91f6fd-850e-4a10-a13c-dfb4c2807082
md"""
## 6. Key performance indicators
"""

# ╔═╡ 8f385433-3fd0-4c61-8f41-169179ba5580
begin
	total_changeover_h = sum(s[i, j] * xval[i, j] for i in N, j in N if i != j) +
						  sum(startup_time * xval[0, j] for j in N)
	total_tardy_h = sum(value.(T))
	n_tardy_orders = count(i -> value(T[i]) > 1e-6, N)
	makespan = maximum(value.(C))
	total_cost = objective_value(model)

	kpi_text = string(
		"| KPI | Value |\n",
		"|---|---|\n",
		"| Makespan | ", @sprintf("%.1f", makespan), " h |\n",
		"| Total changeover time | ", @sprintf("%.1f", total_changeover_h), " h |\n",
		"| Total tardiness | ", @sprintf("%.1f", total_tardy_h), " h across ", n_tardy_orders, " order(s) |\n",
		"| Total cost | \$", @sprintf("%.2f", total_cost), " |\n",
	)
	Markdown.parse(kpi_text)
end

# ╔═╡ 49771591-5524-4b9d-b861-3d886445d093
md"""
## 7. Discussion & extensions

This single-line, deterministic model already captures the dominant
economics of PE/PP campaign scheduling — family-driven changeovers and
due-date pressure — while staying small enough (7 orders → ~50 binaries)
for HiGHS to solve to global optimality in well under a second.

Natural next steps for a production-grade version:

- **Multiple parallel lines** — extend `x` to a 3-index `x[i,j,line]` and
  add line-capability restrictions (not every line can run every grade).
- **Minimum/maximum campaign length** — avoid uneconomically short runs by
  bounding the number of consecutive orders of the same grade.
- **Exact grade-pair changeover matrix** — replace the family-level `s`
  matrix with real historical purge times per ordered grade pair.
- **Inventory & storage costs** — silo capacity limits often force a grade
  off the natural due-date order even when changeovers are cheap.
- **Rolling-horizon re-optimization** — re-solve as new orders arrive or an
  order's priority changes, warm-starting from the current sequence.
- **Robustness** — sample uncertain processing rates / late raw-material
  arrivals and re-optimize, or solve a chance-constrained variant.

Because the model is expressed as a compact JuMP program, all of the above
are incremental changes to the constraints already built above, not a
rewrite.
"""

# ╔═╡ Cell order:
# ╟─46cd7256-ad7a-41da-a6d3-952823f9dc39
# ╠═6b84d682-60c4-4fc4-a098-e5f0a4bccc24
# ╠═236f1801-e89b-4a47-877b-acac88c8fa63
# ╟─da954f7e-5afd-4fbb-acb6-05746a55d110
# ╠═16bc81ee-27e1-46fc-8354-3b7774557bff
# ╠═29f0aa3d-9951-43ca-988a-9625653d5ecb
# ╠═662ecaf1-5a61-4aad-ab89-788834401db8
# ╠═f6e6faca-f942-4809-b7c4-6bfd20cbb0c1
# ╟─370b8e93-37f0-434e-a7ba-4cc893dfb700
# ╠═c2bc08ff-bb90-4c8b-9848-7446fb6280b0
# ╟─44fa3759-362f-4f1d-a213-2d608b13d378
# ╟─e499d937-6f36-4a62-8b9b-f0fb2954c864
# ╠═1139f9f0-b225-4ea6-bf19-ee25c9102584
# ╟─c0c1e0de-1302-4ca3-b6db-337f42fd730d
# ╠═0072ae1f-ef79-4d16-8b0d-8252c97adca7
# ╠═86e678c7-e4ea-4ca3-aff3-127e96cde7d3
# ╟─ab2e8453-2737-489b-a1a1-6466f3b9f9c9
# ╠═64e3b564-98cb-430d-bcdf-ab36deb218ca
# ╟─1bb67a1c-d173-4e50-bae7-bae5f0d655fc
# ╟─8c91f6fd-850e-4a10-a13c-dfb4c2807082
# ╠═8f385433-3fd0-4c61-8f41-169179ba5580
# ╟─49771591-5524-4b9d-b861-3d886445d093
