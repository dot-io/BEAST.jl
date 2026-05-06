# Kernel-agnostic orchestration for GPU block assembly.
#
# Three entry points:
#   - `assembleblock_primer_gpu(biop, tfs, bfs; kernel=…)`
#         CPU-side primer + upload of all device-resident data common to a
#         given (test space, trial space) pair. Returns a `ctx::NamedTuple`.
#   - `assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel=…)`
#         Launches the kernel(s) appropriate for `kernel` to assemble the dof
#         subset (`test_ids`, `trial_ids`) into `store.data`. This is where the
#         dispatch on the four implementations happens.
#   - `assembleblock_gpu(biop, tfs, bfs, store; kernel=…)`
#         Convenience wrapper that runs primer + body for the full dof set.
#
# Valid `kernel` values:
#   :scatter           → element-stationary scatter        (gpu_v1)
#   :gather_entry      → entry-stationary gather, 1 block / output entry (gpu_v2)
#   :gather_tile       → tile-stationary gather Layer 1    (gpu_v3)
#   :gather_tile_coop  → tile-stationary gather Layer 2    (gpu_v4)
#
# Each kernel file (`gpu_v$N.jl`) only defines its kernel(s) and any kernel-
# specific helpers (e.g. `build_tile_pairs` for v4). All launch logic lives
# here so the kernel files remain pure compute.

const _VALID_KERNELS = (:scatter, :gather_entry, :gather_tile, :gather_tile_coop)

# Each kernel needs the assembly data in either scatter or gather direction.
_loop_order(kernel::Symbol) = kernel === :scatter ? :scatter : :gather


"""
    assembleblock_primer_gpu(biop, tfs, bfs; kernel=:gather_tile_coop) -> ctx

Run the CPU primer, build the per-element charts and assembly data, flatten the
quadrature data, and upload everything to the device. The exact set of device
buffers depends on `kernel` — scatter needs `FlattenedAssemblyData`, gather
needs `InvAssemblyData`. The chosen `kernel` is stored in `ctx.kernel` so the
body can later validate consistency.
"""
function assembleblock_primer_gpu(biop, tfs, bfs; kernel::Symbol = :gather_tile_coop)
    kernel ∈ _VALID_KERNELS || error("kernel=$kernel not in $_VALID_KERNELS")
    loop_order = _loop_order(kernel)

    qs = BEAST.defaultquadstrat(biop, tfs, bfs)
    test_elements, tad, trial_elements, bad, qd, _ =
        BEAST.assembleblock_primer(biop, tfs, bfs; quadstrat = qs)

    ZT = scalartype(biop, tfs, bfs)
    num_tfs = numfunctions(tfs)
    num_bfs = numfunctions(bfs)

    tgeo = geometry(tfs); bgeo = geometry(bfs)
    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))
    num_tshapes = numfunctions(refspace(tfs), tdom)
    num_bshapes = numfunctions(refspace(bfs), bdom)

    test_shapes  = refspace(tfs)
    trial_shapes = refspace(bfs)

    test_elements_dev  = CUDA.cu(test_elements)
    trial_elements_dev = CUDA.cu(trial_elements)

    if loop_order === :gather
        tad_gpu = InvAssemblyData(tad, length(test_elements),  num_tshapes, num_tfs, ZT)
        bad_gpu = InvAssemblyData(bad, length(trial_elements), num_bshapes, num_bfs, ZT)
    else  # :scatter
        tad_gpu = FlattenedAssemblyData(tad, length(test_elements),  num_tshapes, ZT)
        bad_gpu = FlattenedAssemblyData(bad, length(trial_elements), num_bshapes, ZT)
    end

    quaddata_gpu = flatten_quaddata_gpu(qd, length(test_elements), length(trial_elements))

    return (;
        kernel,
        qs, ZT,
        num_tfs, num_bfs,
        num_tshapes, num_bshapes,
        test_elements, trial_elements,
        tad, bad, qd,
        test_elements_dev, trial_elements_dev,
        tad_gpu, bad_gpu, quaddata_gpu,
        test_shapes, trial_shapes,
    )
end


"""
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel=ctx.kernel)

Launch the kernel(s) for the requested implementation. `ctx` must come from
`assembleblock_primer_gpu(...; kernel=…)` with a compatible loop direction
(scatter primer ↔ scatter kernel; gather primer ↔ any gather kernel).
"""
function assembleblock_body_gpu!(
    biop, tfs, test_ids, bfs, trial_ids,
    ctx, store::DeviceStore;
    kernel::Symbol = ctx.kernel,
)
    kernel ∈ _VALID_KERNELS || error("kernel=$kernel not in $_VALID_KERNELS")
    _loop_order(kernel) === _loop_order(ctx.kernel) ||
        error("Primer prepared for $(ctx.kernel) (loop_order=$(_loop_order(ctx.kernel))) " *
              "but kernel=$kernel requires $(_loop_order(kernel))")

    if kernel === :scatter
        _launch_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :gather_entry
        _launch_gather_entry!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :gather_tile
        _launch_gather_tile!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :gather_tile_coop
        _launch_gather_tile_coop!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    end
    CUDA.synchronize()
    return store
end


"""
    assembleblock_gpu(biop, tfs, bfs, store; kernel=:gather_tile_coop)

Full-block convenience function: run the primer and body for the entire dof set of
`tfs x bfs`. The result is accumulated into `store.data`.
"""
function assembleblock_gpu(biop, tfs, bfs, store; kernel::Symbol = :gather_tile_coop)
    ctx       = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
    test_ids  = collect(1:numfunctions(tfs))
    trial_ids = collect(1:numfunctions(bfs))
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel)
end




# v1 element-stationary scatter (two-kernel pipeline: integrand → scatter)
function _launch_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_id_dev, trial_id_dev =
        filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)
    num_test  = Int32(length(test_id_dev))
    num_trial = Int32(length(trial_id_dev))
    num_pairs = Int(num_test) * Int(num_trial)

    zlocals_all_dev = CUDA.zeros(ctx.ZT, ctx.num_tshapes, ctx.num_bshapes, num_pairs)

    # Kernel 1 per-pair integrand evaluation
    kernel = @cuda launch = false momintegrals!(
        zlocals_all_dev,
        biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        test_id_dev, trial_id_dev,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
        Int32(ctx.num_tshapes), Int32(ctx.num_bshapes), num_test, num_trial,
    )
    config  = launch_configuration(kernel.fun)
    threads = min(num_pairs, config.threads)
    blocks  = cld(num_pairs, threads)
    kernel(
        zlocals_all_dev,
        biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        test_id_dev, trial_id_dev,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
        Int32(ctx.num_tshapes), Int32(ctx.num_bshapes), num_test, num_trial;
        threads, blocks,
    )

    # Kernel 2 scatter into the output via shared-memory tiles
    test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids)
    gpu_scatter!(
        store.data, zlocals_all_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_id_dev, trial_id_dev,
        test_id_map_dev, trial_id_map_dev,
        ctx.num_tshapes, ctx.num_bshapes,
    )
    return
end

# v2 entry-stationary gather (one BLOCK per output entry, in-block reduction)
function _launch_gather_entry!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids)
    M_block = Int32(length(test_ids))
    N_block = Int32(length(trial_ids))

    threads = 256                     # tunable; warp-multiple recommended
    blocks  = (M_block, N_block)
    @cuda threads = threads blocks = blocks gather_reduce_kernel!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_id_map_dev, trial_id_map_dev,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
    )
    return
end

# v3 tile-stationary gather, Layer 1 (one thread per output entry, no atomics)
function _launch_gather_tile!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids)
    M_block = Int32(length(test_ids))
    N_block = Int32(length(trial_ids))
    M_tile  = TILE_SIZE                # from utils.jl
    N_tile  = TILE_SIZE
    blocks  = (cld(M_block, M_tile), cld(N_block, N_tile))

    @cuda threads = (M_tile, N_tile) blocks = blocks tile_gather_kernel!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_id_map_dev, trial_id_map_dev,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
        M_block, N_block,
    )
    return
end

# v4 tile-stationary gather with cooperative integrand evaluation in shared memory
# (prevents recomputing integrands)
function _launch_gather_tile_coop!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids)
    M_block = Int32(length(test_ids))
    N_block = Int32(length(trial_ids))
    M_tile  = TILE_SIZE
    N_tile  = TILE_SIZE
    n_tiles_m = cld(M_block, M_tile)
    n_tiles_n = cld(N_block, N_tile)

    pair_flat, pair_off = build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)

    @cuda threads = (M_tile, N_tile) blocks = (n_tiles_m, n_tiles_n) tile_gather_cooperative_kernel!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_id_map_dev, trial_id_map_dev,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
        pair_flat, pair_off,
        M_block, N_block, n_tiles_m,
        Val(Int(ctx.num_tshapes)), Val(Int(ctx.num_bshapes)),
    )
    return
end
