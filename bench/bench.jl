using CUDA
using BEAST

"""
    run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
Run a gpu block assembly. primer and body deliberately kept separate because
ACA code may want to call primer once and body on multiple indices.
   """
function run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:gather_tile_coop)
    ZT = BEAST.scalartype(biop, tfs, bfs)
    Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
    store = CuMatrixStore(Z_dev)
    ctx = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel)
    return Array(Z_dev)
end

"""
    test_all_kernels(biop, tfs, test_ids, bfs, trial_ids, A_ref; atol)
Run all four kernel versions for a dof subset and @test each against
a CPU-assembled matrix.
"""
function total_time(biop, tfs, test_ids, bfs, trial_ids)
    @info "Computing total execution type of matrix block assembly..."
    @time assemble(biop, tfs, bfs)

    for kernel in (:scatter, :gather_entry, :gather_tile, :gather_tile_coop)
            @time run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
        end
    end

function gpu_time(biop, tfs, test_ids, bfs, trial_ids; kernel=:gather_tile_coop)
    @info "Computing GPU time of GPU block assembly with kernel $kernel..."
    CUDA.@time run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
end

function main()
    sphere = readmesh(SPHERE, T=Float64)
            sphere2 = translate(sphere, [0.0, 0.0, 4.0])
            op = Maxwell3D.singlelayer(wavenumber=K)
            X = raviartthomas(sphere)
            X2 = raviartthomas(sphere2)
            A = assemble(op, X, X2)
            n = numfunctions(X)
    total_time(op, (X, X2), 1:n, (X, X2), 1:n)
end
