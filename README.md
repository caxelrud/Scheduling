# Scheduling

Manufacturing scheduling applications built in **Julia**, using
[**Pluto.jl**](https://plutojl.org) reactive notebooks for interactive
modeling, and [JuMP.jl](https://jump.dev) + [HiGHS](https://highs.dev) for
optimization.

## Cases

### 1. Polymer plant scheduling (PE / PP campaigns)

`notebooks/polymer_pe_pp_scheduling.jl` — a Pluto notebook that schedules a
**month-long (30-day, 720-hour) production schedule** for a polymer plant's
**two dedicated production trains**: one for 4 polyethylene (PE) grades, one
for 4 polypropylene (PP) grades. PE and PP use different reactor/catalyst
technology, so they never share a line — the notebook solves two independent
single-line MILPs (one per train) in parallel, each grade running a
multi-day campaign with changeovers taking a few hours, the scale real
plant schedulers actually work at.

Raw customer orders (many small, separately-negotiated quantities per
grade) are first **consolidated**: same-grade orders due within a
configurable time window get pooled into one production lot, sized to the
total and timed to the earliest ship date in the group — matching how
plants actually plan, and cutting the number of campaigns (and therefore
changeovers) without hurting service. Each lot then goes through an
explicit, often-asymmetric grade-to-grade changeover matrix *within* its
family, due dates, and tardiness costs. Cost-weight and consolidation-window
sliders let you explore the trade-offs interactively, and the Gantt chart
shows both trains running side by side.

A rendered PDF snapshot is at `docs/polymer_pe_pp_scheduling.pdf`.

## Getting started

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pluto; Pluto.run(notebook="notebooks/polymer_pe_pp_scheduling.jl")'
```

This opens the notebook in your browser with live sliders and an editable
model — change orders, rates, or costs and the schedule re-optimizes.

## Regenerating the PDF documentation

The recommended path solves the model directly in Julia and renders a
self-contained report (no external network access needed to view/print it):

```bash
julia --project=. scripts/generate_report.jl
# -> docs/polymer_pe_pp_scheduling.html (charts embedded as base64 PNG)

# then rasterize that HTML to PDF with any headless Chromium/Chrome:
chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf=docs/polymer_pe_pp_scheduling.pdf \
  --no-pdf-header-footer docs/polymer_pe_pp_scheduling.html
```

Alternatively, `scripts/export_pdf.jl` uses `PlutoSliderServer.jl` to run the
*actual* Pluto notebook headlessly and export it with the real Pluto reader
UI (sliders included, read-only). That export fetches Pluto's frontend
assets from a CDN, so it needs normal internet access from the machine doing
the export.

## Repository layout

```
Project.toml             Julia environment (JuMP, HiGHS, Pluto, Plots, ...)
notebooks/                Pluto notebooks (one manufacturing case per file)
scripts/                  Model prototype, notebook builder, report/export tooling
docs/                     Exported HTML/PDF documentation
```
