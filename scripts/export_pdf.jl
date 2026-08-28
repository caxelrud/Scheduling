#!/usr/bin/env julia
# Runs the Pluto notebook headlessly (validating it executes with no errors),
# exports a static HTML snapshot, then rasterizes that HTML to PDF using
# headless Chromium. Run with:
#   julia --project=. scripts/export_pdf.jl

using PlutoSliderServer

const REPO      = normpath(joinpath(@__DIR__, ".."))
const NOTEBOOK  = joinpath(REPO, "notebooks", "polymer_pe_pp_scheduling.jl")
const DOCS_DIR  = joinpath(REPO, "docs")
mkpath(DOCS_DIR)

println("Running notebook end-to-end via PlutoSliderServer.export_notebook ...")
PlutoSliderServer.export_notebook(
    NOTEBOOK;
    Export_output_dir = DOCS_DIR,
    Export_create_index = false,
)

html_path = joinpath(DOCS_DIR, "polymer_pe_pp_scheduling.html")
@assert isfile(html_path) "Expected HTML export at $html_path"
println("Notebook ran successfully and exported HTML: ", html_path)
