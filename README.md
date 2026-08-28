# Scheduling

Manufacturing scheduling applications built in **Julia**, using
[**Pluto.jl**](https://plutojl.org) reactive notebooks for interactive
modeling, and [JuMP.jl](https://jump.dev) + [HiGHS](https://highs.dev) for
optimization.

## Cases

### 1. Polymer plant scheduling (PE / PP campaigns)

`notebooks/polymer_pe_pp_scheduling.jl` — a Pluto notebook that schedules a
single compounding/extrusion line producing multiple polyethylene (PE) and
polypropylene (PP) grades. It models sequence-dependent changeover times
(driven by PE↔PP family switches), due dates, and tardiness costs as a
mixed-integer linear program, solves it with HiGHS, and visualizes the
resulting campaign sequence as a Gantt chart. Cost-weight sliders let you
explore the changeover-vs-tardiness trade-off interactively.

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
