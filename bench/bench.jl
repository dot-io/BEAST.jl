using CUDA
using BEAST
using LinearAlgebra
using CompScienceMeshes
using Statistics
using Plots
using BenchmarkTools

const K = 2π / 200.0
const BEASTCUDA = Base.get_extension(BEAST, :BEASTCUDA)
@assert BEASTCUDA !== nothing "BEASTCUDA failed to load."

using .BEASTCUDA: assembleblock_gpu, assembleblock_primer_gpu,
    assembleblock_body_gpu!, CuMatrixStore

const KERNEL_ORDER = (:gather_tile, :pair_scatter, :sparse, :hybrid_global, :hybrid_shared)

const h_dof_configs = [
    (0.24, "1e3"),
    (0.077, "1e4"),
    # (0.03, "1e5"),
]


"""
    run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel, variant)

Run a full primer + body GPU block assembly and return the result matrix.
Primer and body are kept separate internally so ACA code can reuse the primer.
"""
function run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:pair_scatter, variant=:direct)
    ZT = BEAST.scalartype(biop, tfs, bfs)
    Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
    store = CuMatrixStore(Z_dev)
    ctx = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel, variant=variant)
    return Array(Z_dev)
end


"""
    benchmark_gpu(biop, tfs, test_ids, bfs, trial_ids; warmup, n_repeats)

Benchmark all GPU kernel variants with CUDA.@elapsed.

Runs a warmup pass first (to exclude JIT compilation), then runs each kernel
`n_repeats` times and reports the median.  Median is more robust to scheduling
spikes than mean; minimum would capture best-case hardware throughput.
"""
function benchmark_gpu(biop, tfs, test_ids, bfs, trial_ids; warmup=true, n_repeats=1)
    kernels = (:gather_tile, :pair_scatter, :sparse, :hybrid_global, :hybrid_shared)

    if warmup
        @info "  Warmup (JIT compilation)..."
        for kernel in kernels
            try
                run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
            catch e
                @warn "  Warmup failed for $kernel: $e"
            end
        end
    end

    results = Dict{Symbol,Float64}()
    y_err = Dict{Symbol,Float64}()

    for kernel in kernels
        @info "  Benchmarking kernel=$kernel (n=$n_repeats runs)..."
        try
            times = []
            for _ in 1:n_repeats
                timetuple = @btimed run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
                push!(times, timetuple.time)
            end
            mean_time = mean(times)
            standard_error = stdm(times, mean_time, corrected=true)
            t = median(times)
            ci_95_one_side = 1.645 * standard_error # for error bars
            results[kernel] = t
            y_err[kernel] = ci_95_one_side
            @info "    median=$(round(t; digits=4)) s  min=$(round(minimum(times); digits=4)) s  max=$(round(maximum(times); digits=4)) s"
        catch e
            @warn "  Failed: $e"
            results[kernel] = NaN
            y_err[kernel] = NaN
        end
    end

    return results, y_err
end

"""
    benchmark_basis(label, op, X, X2; warmup, n_repeats)

Run the full CPU + GPU benchmark suite for a single (operator, basis) pair.

Skips the CPU benchmark for large n (> 20K DOFs per side) since the runtime
would be prohibitive.  Skips the GPU full-block benchmark if the output matrix
would exceed half of available GPU memory.

Returns `(cpu_time_far, gpu_results, n_dofs, n_test_el)`.
`cpu_time_far` is NaN when the CPU benchmark was skipped.
`gpu_results` is an empty Dict when the GPU benchmark was skipped.
"""
function benchmark_basis(label, op, X, X2; warmup=true, n_repeats=5)
    n = numfunctions(X)
    n2 = numfunctions(X2)
    tgeo = geometry(X)
    n_test_el = length(tgeo)
    ZT = BEAST.scalartype(op, X, X2)
    matrix_bytes = Int(n) * Int(n2) * sizeof(ZT)
    avail_bytes = CUDA.available_memory()

    @info "Benchmarking $label"
    @info "Test DOFs $n"
    @info "Trial DOFs $n2"
    @info "Output matrix $n x $n2 ($(round(matrix_bytes / 1e9; digits=2)) GB)"
    @info "GPU free memory $(round(avail_bytes / 1e9; digits=2)) GB"

    # Skip for large problems where runtime would be hours.
    cpu_time_far = NaN
    if n <= 20_000
        @info "  CPU assembly (DoubleNumQStrat, far-field only)..."
        far_qs = BEAST.DoubleNumQStrat(2, 3)
        timetuple = @btimed assemble(op, X, X2; quadstrat=far_qs)
        cpu_time_far = timetuple.time
        @info " $(round(cpu_time_far; digits=3)) s"
    else
        @info " CPU benchmark skipped (n=$n > 20K; extrapolated runtime would be hours)"
    end

    #gpu benchmark
    gpu_results = Dict{Symbol,Float64}()
    if matrix_bytes > avail_bytes
        @warn "GPU full-block skipped: matrix $(round(matrix_bytes/1e9;digits=1)) GB > $(round(avail_bytes/1e9;digits=1)) GB"
    else
        @info "GPU full-block assembly (n_repeats=$n_repeats, reporting median)..."
        gpu_results, y_err = benchmark_gpu(op, X, 1:n, X2, 1:n; warmup=true, n_repeats)
    end

    @info "Results for $label"
    if !isnan(cpu_time_far)
        @info "CPU (far-only): $(round(cpu_time_far; digits=3))s  (speedup baseline)"
    end
    for kernel in KERNEL_ORDER
        haskey(gpu_results, kernel) || continue
        t = gpu_results[kernel]
        isnan(t) && (@info "    $(rpad(string(kernel), 20)) FAILED"; continue)
        if !isnan(cpu_time_far)
            @info "    $(rpad(string(kernel), 20)) $(round(t; digits=4)) s  ($(round(cpu_time_far/t; digits=1))x)"
        else
            @info "    $(rpad(string(kernel), 20)) $(round(t; digits=4)) s"
        end
    end

    return (cpu_time_far, gpu_results, n, n_test_el, y_err)
end



"""
    _plot_clustered_speedup(rt_cpu, rt_gpu, lg_cpu, lg_gpu, n_elements, rt_n, lg_n;
                            h_val, suffix)
"""
function _plot_clustered_speedup(rt_cpu, rt_gpu, y_err_rt, lg_cpu, lg_gpu, y_err_lg, n_elements, rt_n, lg_n;
    h_val=0.25, suffix="")

    present_kernels = [k for k in KERNEL_ORDER if haskey(rt_gpu, k) || haskey(lg_gpu, k)]
    isempty(present_kernels) && (@warn "No benchmark results to plot."; return)

    n_kernels = length(present_kernels)
    labels = string.(present_kernels)

    _speedup(cpu_time, gpu_time) = (isnan(cpu_time) || isnan(gpu_time)) ? 0.0 : cpu_time / gpu_time

    rt_speedups = [_speedup(rt_cpu, get(rt_gpu, k, NaN)) for k in present_kernels]
    lg_speedups = [_speedup(lg_cpu, get(lg_gpu, k, NaN)) for k in present_kernels]

    bar_width = 0.35
    x = collect(1:n_kernels)
    x_rt = x .- bar_width / 2
    x_lg = x .+ bar_width / 2

    p = bar(x_rt, rt_speedups;
        bar_width=bar_width, label="RT / Maxwell3D", color=:steelblue,
        title="GPU Speedup vs CPU (far-field only)  (h=$h_val, $n_elements elems)",
        ylabel="Speedup", xlabel="Kernel",
        legend=:topright, minorgrid=true,
        size=(1100, 550),
        left_margin=5Plots.mm, bottom_margin=12Plots.mm,
        xticks=(x, labels), xrotation=30,
        yerror=collect(values(y_err_rt))
    )
    bar!(x_lg, lg_speedups; bar_width=bar_width, label="LagrangeC0D1 / Helmholtz3D", color=:coral, yerror=collect(values(y_err_lg)))
    hline!([1.0]; label="CPU baseline", linestyle=:dash, color=:red)

    for i in 1:n_kernels
        rt_speedups[i] > 0 && annotate!(x_rt[i], rt_speedups[i] + 0.3,
            Plots.text("$(round(rt_speedups[i]; digits=1))×", 7, :center))
        lg_speedups[i] > 0 && annotate!(x_lg[i], lg_speedups[i] + 0.3,
            Plots.text("$(round(lg_speedups[i]; digits=1))×", 7, :center))
    end

    outfile = joinpath(@__DIR__, "speedup$(suffix).png")
    savefig(p, outfile)
    @info "Speedup plot saved to $outfile"
    return p
end


function main()
    for (h_val, dof_label) in h_dof_configs
        @info "Mesh with size h=$h_val, approx. $dof_label RT DOFs"

        sphere = meshsphere(radius=1.0, h=h_val)
        sphere2 = CompScienceMeshes.translate(sphere, [0.0, 0.0, 4.0])
        @info "  Elements: $(length(sphere))"
        # rt basis
        rt_op = Maxwell3D.singlelayer(wavenumber=K)
        X_rt = raviartthomas(sphere)
        X2_rt = raviartthomas(sphere2)
        rt_cpu, rt_gpu, rt_n, rt_nel, y_err_rt = benchmark_basis(
            "RT / Maxwell3D ($dof_label DOFs)", rt_op, X_rt, X2_rt;
            warmup=true, n_repeats=3)
        #lagrange basis
        hh_op = Helmholtz3D.singlelayer(gamma=im * K)
        X_lg = lagrangec0d1(sphere; dirichlet=false)
        X2_lg = lagrangec0d1(sphere2; dirichlet=false)
        lg_cpu, lg_gpu, lg_n, lg_nel, y_err_lg = benchmark_basis(
            "LagrangeC0D1 / Helmholtz3D ($dof_label DOFs)", hh_op, X_lg, X2_lg;
            warmup=true, n_repeats=1)

        _plot_clustered_speedup(rt_cpu, rt_gpu, y_err_rt, lg_cpu, lg_gpu, y_err_lg, rt_nel, rt_n, lg_n;
            h_val, suffix="_$(dof_label)_dofs")
    end
end


"""
    rowcol_scenarios(n_test, n_trial) -> Vector{Tuple{String,Vector{Int},Vector{Int}}}

Return the four H-matrix-style index subsets:
1. One wide row  (1 test DOF  × 320 trial DOFs)
2. One tall col  (320 test DOFs × 1 trial DOF)
3. Four discontiguous rows × 80 trial DOFs
4. 80 test DOFs × four discontiguous trial cols
"""
function rowcol_scenarios(n_test::Int, n_trial::Int)
    row_w  = min(320, n_trial)
    col_h  = min(320, n_test)
    multi_w = min(80, n_trial)
    multi_h = min(80, n_test)

    # four evenly-spaced, discontiguous DOF ids
    discontig_test  = [clamp(round(Int, i * n_test  / 5), 1, n_test)  for i in 1:4]
    discontig_trial = [clamp(round(Int, i * n_trial / 5), 1, n_trial) for i in 1:4]

    return [
        ("1×$(row_w) (wide row)",        [n_test ÷ 2],    collect(1:row_w)),
        ("$(col_h)×1 (tall col)",        collect(1:col_h), [n_trial ÷ 2]),
        ("4×$(multi_w) (discontig rows)", discontig_test,  collect(1:multi_w)),
        ("$(multi_h)×4 (discontig cols)", collect(1:multi_h), discontig_trial),
    ]
end


"""
    benchmark_dof_scenarios(label, op, X, X2; scenarios, n_repeats, kernels)

Build one GPU primer per kernel (reusable across blocks), then time the body-only
call for each (test_ids, trial_ids) scenario.

Returns `Dict{String, Dict{Symbol, Float64}}` mapping scenario name → kernel → median body time (s).
"""
function benchmark_dof_scenarios(label, op, X, X2;
    scenarios,
    n_repeats::Int=5,
    kernels=KERNEL_ORDER)

    @info "Row/column scenario benchmark: $label"
    ZT = BEAST.scalartype(op, X, X2)

    primers = Dict{Symbol,Any}()
    for kernel in kernels
        try
            primers[kernel] = assembleblock_primer_gpu(op, X, X2; kernel)
        catch e
            @warn "  Primer build failed for $kernel: $e"
        end
    end

    results = Dict{String,Dict{Symbol,Float64}}()

    for (name, test_ids, trial_ids) in scenarios
        @info "  Scenario: \"$name\"  ($(length(test_ids)) × $(length(trial_ids)))"
        results[name] = Dict{Symbol,Float64}()

        for kernel in kernels
            haskey(primers, kernel) || continue
            ctx = primers[kernel]
            try
                # warmup (excludes JIT from timings)
                Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
                assembleblock_body_gpu!(op, X, test_ids, X2, trial_ids, ctx,
                    CuMatrixStore(Z_dev); kernel)
                CUDA.synchronize()

                times = Float64[]
                for _ in 1:n_repeats
                    Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
                    t = CUDA.@elapsed begin
                        assembleblock_body_gpu!(op, X, test_ids, X2, trial_ids, ctx,
                            CuMatrixStore(Z_dev); kernel)
                        CUDA.synchronize()
                    end
                    push!(times, t)
                end

                med = median(times)
                results[name][kernel] = med
                @info "    $(rpad(string(kernel), 18)) $(round(med * 1e3; digits=3)) ms"
            catch e
                @warn "    $kernel failed: $e"
                results[name][kernel] = NaN
            end
        end
    end

    return results
end


"""
    _plot_dof_scenarios(label_rt, label_lg, scenarios, rt_results, lg_results; suffix)

2×2 subplot grid showing body-only time (ms) per scenario and kernel.
RT and Lagrange bars are side by side within each subplot.
"""
function _plot_dof_scenarios(label_rt, label_lg, scenarios, rt_results, lg_results; suffix="")
    scenario_names = [s[1] for s in scenarios]
    present_kernels = [k for k in KERNEL_ORDER
        if any(!isnan(get(get(rt_results, n, Dict()), k, NaN)) ||
               !isnan(get(get(lg_results, n, Dict()), k, NaN))
           for n in scenario_names)]
    isempty(present_kernels) && (@warn "No scenario results to plot."; return)

    n_kernels = length(present_kernels)
    klabels = string.(present_kernels)
    x = collect(1:n_kernels)
    bar_width = 0.35

    subplots = []
    for name in scenario_names
        rt_ms = [get(get(rt_results, name, Dict()), k, NaN) * 1e3 for k in present_kernels]
        lg_ms = [get(get(lg_results, name, Dict()), k, NaN) * 1e3 for k in present_kernels]

        p = bar(x .- bar_width/2, rt_ms;
            bar_width=bar_width, label=label_rt, color=:steelblue,
            title=name, ylabel="Body time (ms)", xlabel="Kernel",
            legend=:topright, minorgrid=true,
            left_margin=5Plots.mm, bottom_margin=14Plots.mm,
            xticks=(x, klabels), xrotation=30)
        bar!(x .+ bar_width/2, lg_ms; bar_width=bar_width, label=label_lg, color=:coral)
        push!(subplots, p)
    end

    combined = plot(subplots...; layout=(2, 2), size=(1600, 900))
    outfile = joinpath(@__DIR__, "rowcol_scenarios$(suffix).png")
    savefig(combined, outfile)
    @info "Scenario plot saved to $outfile"
    return combined
end


function main_rowcol()
    for (h_val, dof_label) in h_dof_configs
        @info "=== Row/col scenarios  h=$h_val (~$dof_label DOFs) ==="
        sphere  = meshsphere(radius=1.0, h=h_val)
        sphere2 = CompScienceMeshes.translate(sphere, [0.0, 0.0, 4.0])

        rt_op  = Maxwell3D.singlelayer(wavenumber=K)
        X_rt   = raviartthomas(sphere)
        X2_rt  = raviartthomas(sphere2)

        hh_op  = Helmholtz3D.singlelayer(gamma=im * K)
        X_lg   = lagrangec0d1(sphere; dirichlet=false)
        X2_lg  = lagrangec0d1(sphere2; dirichlet=false)

        scenarios_rt = rowcol_scenarios(numfunctions(X_rt),  numfunctions(X2_rt))
        scenarios_lg = rowcol_scenarios(numfunctions(X_lg), numfunctions(X2_lg))

        rt_results = benchmark_dof_scenarios(
            "RT / Maxwell3D ($dof_label DOFs)", rt_op, X_rt, X2_rt;
            scenarios=scenarios_rt, n_repeats=5)
        lg_results = benchmark_dof_scenarios(
            "Lagrange / Helmholtz3D ($dof_label DOFs)", hh_op, X_lg, X2_lg;
            scenarios=scenarios_lg, n_repeats=5)

        # Plot using RT scenario names (same structure as Lagrange)
        _plot_dof_scenarios(
            "RT ($(numfunctions(X_rt)) DOFs)",
            "Lagrange ($(numfunctions(X_lg)) DOFs)",
            scenarios_rt, rt_results, lg_results;
            suffix="_$(dof_label)_dofs")
    end
end


main()
main_rowcol()
