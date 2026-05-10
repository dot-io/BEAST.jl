using CUDA
using BEAST
using LinearAlgebra
using CompScienceMeshes
using Statistics
using Plots

const H = haskey(ENV, "BEAST_H") ? parse(Float64, ENV["BEAST_H"]) : 0.25
const K = 2π / 200.0   # long wavelength → well-separated far-field blocks

const BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
@assert BEASTCUDAExt !== nothing "BEASTCUDAExt failed to load."

using .BEASTCUDAExt: assembleblock_gpu, assembleblock_primer_gpu,
    assembleblock_body_gpu!, CuMatrixStore, gpu_scatter_contention


"""
    run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel, variant)

Run a GPU block assembly. Primer and body deliberately kept separate because
ACA code may want to call primer once and body on multiple indices.
"""
function run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:gather_tile_coop, variant=:direct)
    ZT = BEAST.scalartype(biop, tfs, bfs)
    Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
    store = CuMatrixStore(Z_dev)
    ctx = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel, variant=variant)
    return Array(Z_dev)
end

"""
    benchmark_gpu(biop, tfs, test_ids, bfs, trial_ids; warmup=true)

Benchmark all GPU kernel versions with CUDA.@elapsed.
Runs a warmup pass first (to exclude JIT compilation) according to best practice in
https://guillesanbri.com/CUDA-Benchmarks/ (TODO check if any more reputable sources)
"""
function benchmark_gpu(biop, tfs, test_ids, bfs, trial_ids; warmup=true)
    n_test = length(test_ids)
    n_trial = length(trial_ids)
    kernels = (:scatter, :gather_entry, :gather_tile, :gather_tile_coop, :sparse)

    if warmup
        @info "  Warmup w/ JIT compilation..."
        for kernel in kernels
            try
                run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
            catch e
                @warn "  Warmup failed for $kernel: $e . Returning"
            end
        end
        try
            run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:scatter, variant=:tiled)
        catch e
            @warn "Warmup failed for scatter/tiled: $e"
        end
    end

    results = Dict{Symbol,Float64}()

    for kernel in kernels
        @info "Benchmarking kernel=$kernel..."
        try
            # Measure total wall-clock time (primer + body + transfer). I think this makes
            # it the best method to benchmark against @elapsed
            t = CUDA.@elapsed run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
            results[kernel] = t
            @info "$(round(t; digits=4)) s."
        catch e
            @warn "Failed: $e"
            results[kernel] = NaN #clearly report failed kernel exec
        end

        if kernel === :scatter
            @info "Benchmarking kernel=scatter (tiled)..."
            try
                t = CUDA.@elapsed run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:scatter, variant=:tiled)
                results[:scatter_tiled] = t
                @info "    → $(round(t; digits=4)) s"
            catch e
                @warn "Failed: $e"
                results[:scatter_tiled] = NaN
            end
        end
    end

    return results
end

function main()
    @info "Generating mesh with h=$H..."
    sphere = meshsphere(radius=1.0, h=H)
    sphere2 = CompScienceMeshes.translate(sphere, [0.0, 0.0, 4.0])

    op = Maxwell3D.singlelayer(wavenumber=K)
    X = raviartthomas(sphere)
    X2 = raviartthomas(sphere2)
    n = numfunctions(X)

    n_test_el = length(sphere)
    n_trial_el = length(sphere2)
    n_pairs = n_test_el * n_trial_el

    @info "Problem size:"
    @info "  Test elements:   $n_test_el"
    @info "  Trial elements:  $n_trial_el"
    @info "  Element pairs:   $n_pairs"
    @info "  Test DOFs:       $n"
    @info "  Trial DOFs:      $(numfunctions(X2))"
    @info "  Output matrix:   $n × $(numfunctions(X2)) = $(n * numfunctions(X2)) entries"

    @info "CPU assembly..."
    cpu_time_default = @elapsed A_cpu_default = assemble(op, X, X2)
    @info "  CPU time (default): $(round(cpu_time_default; digits=3)) s"

    @info ""
    @info "CPU assembly (DoubleNumQStrat — far-field only)..."
    far_qs = BEAST.DoubleNumQStrat(2, 3)  # same outer/inner rules as GPU
    cpu_time_far = @elapsed A_cpu_far = assemble(op, X, X2; quadstrat=far_qs)
    @info "  CPU time (far-only): $(round(cpu_time_far; digits=3)) s"

    # Use the fair baseline for speedup calculations
    cpu_time = cpu_time_far
    A_cpu = A_cpu_far

    @info "GPU assembly..."
    results = benchmark_gpu(op, X, 1:n, X2, 1:n; warmup=true)

    @info "Results  (h=$H, $n_test_el elements, $n DOFs)"
    @info "  CPU (default qs):    $(round(cpu_time_default; digits=3)) s"
    @info "  CPU (far-only qs):   $(round(cpu_time; digits=3)) s (used as speedup baseline)"
    for kernel in (:scatter, :scatter_tiled, :gather_entry, :gather_tile, :gather_tile_coop, :sparse)
        if haskey(results, kernel)
            t = results[kernel]
            speedup = isnan(t) ? NaN : round(cpu_time / t; digits=1)
            @info "  $(rpad(string(kernel), 20)) $(round(t; digits=4)) s   ($(speedup)x)"
        end
    end

    _plot_speedup(results, cpu_time, n, n_test_el)

    return results
end


"""
    _plot_speedup(results, cpu_time, n_dofs, n_elements)

Generate a bar chart of GPU speedup vs CPU for all kernel variants.
Saves the plot to `bench/speedup.png` and also displays it.
"""
function _plot_speedup(results, cpu_time, n_dofs, n_elements)
    kernel_order = [:scatter, :scatter_tiled, :gather_entry, :gather_tile, :gather_tile_coop, :sparse]
    labels = String[]
    speedups = Float64[]

    for k in kernel_order
        if haskey(results, k)
            t = results[k]
            push!(labels, string(k))
            push!(speedups, isnan(t) ? 0.0 : cpu_time / t)
        end
    end

    if isempty(labels)
        @warn "No benchmark results to plot."
        return
    end

    p = bar(
        labels,
        speedups,
        title="GPU Speedup vs CPU (far-field only)  (h=$H, $n_elements elems, $n_dofs DOFs)",
        ylabel="Speedup",
        xlabel="Kernel",
        label="GPU / CPU",
        color=:steelblue,
        legend=:topright,
        minorgrid=true,
        size=(900, 500),
        left_margin=5Plots.mm,
        bottom_margin=10Plots.mm,
    )
    hline!([1.0], label="CPU baseline", linestyle=:dash, color=:red)

    # Annotate each bar with its speedup value
    for (i, s) in enumerate(speedups)
        annotate!(i, s + 0.3, Plots.text("$(round(s; digits=1))×", 8, :center))
    end

    savefig(p, joinpath(@__DIR__, "speedup.png"))
    @info "Speedup plot saved to bench/speedup.png"
    return p
end

main()
