const _VALID_KERNELS = (:gather_tile, :pair_scatter, :sparse, :hybrid_global, :hybrid_shared)

# Each kernel needs the assembly data in either scatter or gather direction.
_loop_order(kernel::Symbol) =  (kernel === :sparse ? :sparse : (kernel === :pair_scatter ? :scatter : (kernel === :hybrid_global ? :hybrid : (kernel === :hybrid_shared ? :hybrid : :gather))))


"""
    assembleblock_primer_gpu(biop, tfs, bfs; kernel) -> ctx

"""
function assembleblock_primer_gpu(biop, tfs, bfs;qs=BEAST.defaultquadstrat(biop, tfs, bfs), kernel::Symbol=:pair_scatter)
    kernel ∈ _VALID_KERNELS || error("kernel=$kernel not in $_VALID_KERNELS")
    loop_order = _loop_order(kernel)

    test_elements, tad, trial_elements, bad, qd, _ =
        BEAST.assembleblock_primer(biop, tfs, bfs; quadstrat=qs)

    ZT = scalartype(biop, tfs, bfs)
    num_tfs = numfunctions(tfs)
    num_bfs = numfunctions(bfs)

    tgeo = geometry(tfs)
    bgeo = geometry(bfs)
    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))
    num_tshapes = numfunctions(refspace(tfs), tdom)
    num_bshapes = numfunctions(refspace(bfs), bdom)

    test_shapes = refspace(tfs)
    trial_shapes = refspace(bfs)

    # ── Sparse kernel: use the third-party implementation's primer ──
    if kernel === :sparse
        return _assembleblock_primer_gpu_sparse(biop, tfs, bfs; qs, ZT,
            num_tfs, num_bfs, num_tshapes, num_bshapes,
            test_shapes, trial_shapes)
    end

    test_elements_dev = CUDA.cu(test_elements)
    trial_elements_dev = CUDA.cu(trial_elements)

    if loop_order === :gather
        tad_gpu = InvAssemblyData(tad, length(test_elements), num_tshapes, num_tfs, ZT)
        bad_gpu = InvAssemblyData(bad, length(trial_elements), num_bshapes, num_bfs, ZT)
    elseif loop_order === :scatter
        tad_gpu = FlattenedAssemblyData(tad, length(test_elements), num_tshapes, ZT)
        bad_gpu = FlattenedAssemblyData(bad, length(trial_elements), num_bshapes, ZT)
    elseif loop_order === :hybrid
        # Build both forward (with contributor index k) and inverse assembly data
        tad_gpu = HybridAssemblyData(tad, bad,
            length(test_elements), length(trial_elements),
            num_tshapes, num_bshapes,
            num_tfs, num_bfs, ZT)
        bad_gpu = nothing  # packed inside tad_gpu (HybridAssemblyData)
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
"""
function assembleblock_body_gpu!(
    biop, tfs, test_ids, bfs, trial_ids,
    ctx, store::DeviceStore;
    kernel::Symbol=ctx.kernel, variant::Symbol=:direct,
)
    kernel ∈ _VALID_KERNELS || error("kernel=$kernel not in $_VALID_KERNELS")
    _loop_order(kernel) === _loop_order(ctx.kernel) ||
        error("Primer prepared for $(ctx.kernel) (loop_order=$(_loop_order(ctx.kernel))) " *
              "but kernel=$kernel requires $(_loop_order(kernel))")

    if kernel === :scatter
        _launch_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; variant=variant)
    elseif kernel === :gather_entry
        _launch_gather_entry!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :gather_tile
        _launch_gather_tile!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :gather_tile_coop
        _launch_gather_tile_coop!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :pair_scatter
        _launch_pair_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :sparse
        _launch_sparse!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :hybrid_global
        _launch_hybrid_global!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :hybrid_shared
        _launch_hybrid_shared!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    elseif kernel === :warp_scatter
        _launch_warp_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    end
    CUDA.synchronize()
    return store
end


"""
    assembleblock_gpu(biop, tfs, bfs, store; kernel=:gather_tile_coop)
"""
function assembleblock_gpu(biop, tfs, bfs, store; kernel::Symbol=:gather_tile_coop)
    ctx = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
    test_ids = collect(1:numfunctions(tfs))
    trial_ids = collect(1:numfunctions(bfs))
    assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel)
end




# # v1  scatter (two-kernel pipeline: integrand → scatter)
# function _launch_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; variant::Symbol=:direct)
#     test_id_dev, trial_id_dev =
#         filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)
#     num_test = Int32(length(test_id_dev))
#     num_trial = Int32(length(trial_id_dev))
#     num_pairs = Int(num_test) * Int(num_trial)

#     zlocals_all_dev = CUDA.zeros(ctx.ZT, ctx.num_tshapes, ctx.num_bshapes, num_pairs)

#     # Kernel 1 per-pair integrand evaluation
#     kernel = @cuda launch = false momintegrals!(
#         zlocals_all_dev,
#         biop,
#         ctx.test_shapes, ctx.trial_shapes,
#         ctx.test_elements_dev, ctx.trial_elements_dev,
#         test_id_dev, trial_id_dev,
#         ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
#         ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
#         ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
#         ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
#         Int32(ctx.num_tshapes), Int32(ctx.num_bshapes), num_test, num_trial,
#     )
#     config = launch_configuration(kernel.fun)
#     threads = min(num_pairs, config.threads)
#     blocks = cld(num_pairs, threads)
#     kernel(
#         zlocals_all_dev,
#         biop,
#         ctx.test_shapes, ctx.trial_shapes,
#         ctx.test_elements_dev, ctx.trial_elements_dev,
#         test_id_dev, trial_id_dev,
#         ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
#         ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
#         ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
#         ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
#         Int32(ctx.num_tshapes), Int32(ctx.num_bshapes), num_test, num_trial;
#         threads, blocks,
#     )

#     test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids, ctx.num_tfs, ctx.num_bfs)
#     gpu_scatter!(
#         store.data, zlocals_all_dev,
#         ctx.tad_gpu, ctx.bad_gpu,
#         test_id_dev, trial_id_dev,
#         test_id_map_dev, trial_id_map_dev,
#         ctx.num_tshapes, ctx.num_bshapes;
#         variant=variant,
#     )
#     return
# end

#scatter (one thread per element pair, full zlocal + direct scatter)
function _launch_pair_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_id_dev, trial_id_dev =
        filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)
    test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids, ctx.num_tfs, ctx.num_bfs)

    gpu_pair_scatter!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu, ctx.bad_gpu,
        test_id_dev, trial_id_dev,
        test_id_map_dev, trial_id_map_dev,
        ctx.quaddata_gpu,
        ctx.num_tshapes, ctx.num_bshapes,
    )
    return
end

# v2 entry-stationary gather (one BLOCK per output entry, in-block reduction)
function _launch_gather_entry!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    # Local→global maps: local_index -> global_dof_id
    # The gather kernels need to convert per-thread local indices into global
    # DOF ids for looking up InvAssemblyData offsets/lengths.
    test_l2g = CUDA.cu(Int32.(test_ids))
    trial_l2g = CUDA.cu(Int32.(trial_ids))
    M_block = Int32(length(test_ids))
    N_block = Int32(length(trial_ids))

    threads = 256                     # tunable; warp-multiple recommended
    blocks = (M_block, N_block)
    @cuda threads = threads blocks = blocks gather_reduce_kernel!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_l2g, trial_l2g,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
    )
    return
end

# naive gather
function _launch_gather_tile!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    test_l2g = CUDA.cu(Int32.(test_ids))
    trial_l2g = CUDA.cu(Int32.(trial_ids))
    M_block = Int32(length(test_ids))
    N_block = Int32(length(trial_ids))
    M_tile = TILE_SIZE                # from utils.jl
    N_tile = TILE_SIZE
    blocks = (cld(M_block, M_tile), cld(N_block, N_tile))

    @cuda threads = (M_tile, N_tile) blocks = blocks tile_gather_kernel!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
        ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
        test_l2g, trial_l2g,
        ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
        ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
        M_block, N_block,
    )
    return
end

# function _launch_gather_tile_coop!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
#     test_l2g = CUDA.cu(Int32.(test_ids))
#     trial_l2g = CUDA.cu(Int32.(trial_ids))
#     M_block = Int32(length(test_ids))
#     N_block = Int32(length(trial_ids))
#     M_tile = TILE_SIZE
#     N_tile = TILE_SIZE
#     n_tiles_m = cld(M_block, M_tile)
#     n_tiles_n = cld(N_block, N_tile)

#     pair_flat, pair_off = build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)

#     @cuda threads = (M_tile, N_tile) blocks = (n_tiles_m, n_tiles_n) tile_gather_cooperative_kernel!(
#         store.data, biop, ctx.test_shapes, ctx.trial_shapes,
#         ctx.test_elements_dev, ctx.trial_elements_dev,
#         ctx.tad_gpu.flat, ctx.tad_gpu.offsets, ctx.tad_gpu.lengths,
#         ctx.bad_gpu.flat, ctx.bad_gpu.offsets, ctx.bad_gpu.lengths,
#         test_l2g, trial_l2g,
#         ctx.quaddata_gpu.tqp_flat, ctx.quaddata_gpu.tqp_offsets, ctx.quaddata_gpu.tqp_lengths,
#         ctx.quaddata_gpu.bqp_flat, ctx.quaddata_gpu.bqp_offsets, ctx.quaddata_gpu.bqp_lengths,
#         pair_flat, pair_off,
#         M_block, N_block, n_tiles_m,
#         Int32(ctx.num_tshapes), Int32(ctx.num_bshapes),
#         Val(M_tile), Val(N_tile),
#     )
#     return
# end


"""
    _assembleblock_primer_gpu_sparse(biop, tfs, bfs; ...) -> ctx

Sparse-kernel primer. Calls `SparseImpl.assemble_primer_gpu` for both test and
trial spaces, uploads the CommonVertex rule, and packages everything into a
context NamedTuple compatible with the assembly interface.
"""
function _assembleblock_primer_gpu_sparse(biop, tfs, bfs;
    qs, ZT, num_tfs, num_bfs, num_tshapes, num_bshapes,
    test_shapes, trial_shapes)

    SI = SparseImpl

    # Extract the far-field quadrature rule from the strategy.
    # DoubleNumWiltonSauterQStrat uses outer_rule_far / inner_rule_far,
    # DoubleNumSauterQstrat uses outer_rule / inner_rule.
    outer_rule = _far_outer_rule(qs)
    inner_rule = _far_inner_rule(qs)

    # Run the sparse primer for test and trial spaces
    test_l2g, test_ad_qd = SI.assemble_primer_gpu(biop, tfs, outer_rule)
    trial_l2g, trial_ad_qd = SI.assemble_primer_gpu(biop, bfs, inner_rule)

    # Unpack the sparse primer results
    (test_el_d, test_ad_d), test_qd = test_ad_qd
    (trial_el_d, trial_ad_d), trial_qd = trial_ad_qd

    # Build CommonVertex rule on the GPU (needed by Sauter-Schwab kernels)
    cv_rule = CompScienceMeshes.legendre(qs.sauter_schwab_common_vert, 0.0, 1.0)
    q = Array{Tuple{Float64,Float64}}(undef, length(cv_rule[2]))
    for (i, a) in enumerate(zip(cv_rule[1], cv_rule[2]))
        q[i] = a
    end
    cvrule_d = CUDA.cu(q)

    # Package quad data in the format expected by assemble_block_gpu
    qd_d = (test_qd, trial_qd, cvrule_d)

    # We also need the CPU-side sparse assembly data for subsetting in the body
    # Reconstruct it from the function spaces
    _, test_ad_cpu, _ = BEAST.assemblydata(tfs)
    _, trial_ad_cpu, _ = BEAST.assemblydata(bfs)

    # Build the sparse CPU-side assembly data matrices (dof × (element×shape))
    test_ad_sparse_cpu = _build_sparse_ad_cpu(tfs, test_ad_cpu, num_tshapes)
    trial_ad_sparse_cpu = _build_sparse_ad_cpu(bfs, trial_ad_cpu, num_bshapes)

    return (;
        kernel=:sparse,
        qs, ZT,
        num_tfs, num_bfs,
        num_tshapes, num_bshapes,
        test_shapes, trial_shapes,
        # Sparse-specific fields
        test_l2g, trial_l2g,
        test_el_d, trial_el_d,
        test_ad_d, trial_ad_d,               # CuSparseMatrixCSC on device
        test_ad_sparse_cpu, trial_ad_sparse_cpu,  # SparseMatrixCSC on CPU
        test_qd, trial_qd,
        qd_d,
    )
end


"""
    _build_sparse_ad_cpu(X, ad_cpu, num_shapes)

Build a `SparseMatrixCSC` from BEAST `AssemblyData` in the format expected
by the sparse implementation: rows = DOF IDs, columns = (element-1)*num_shapes + shape_id.
"""
function _build_sparse_ad_cpu(X, ad_cpu, num_shapes)
    num_dofs = numfunctions(X)
    el, _, _ = BEAST.assemblydata(X)
    num_el = length(el)

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    ax = axes(ad_cpu.data)
    for i in ax[1]  # shape functions
        for j in ax[2]  # element-local shape index
            for k in ax[3]  # element index
                dof = ad_cpu.data[i, j, k][1]
                if dof > 0
                    push!(rows, dof)
                    push!(cols, num_shapes * (k - 1) + j)
                    push!(vals, ad_cpu.data[i, j, k][2])
                end
            end
        end
    end

    return SparseArrays.sparse(rows, cols, vals, num_dofs, num_shapes * num_el)
end


"""
    _launch_sparse!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)

Sparse-kernel body. Converts DOF IDs to element IDs, calls
`SparseImpl.assemble_block_gpu`, and writes the result into `store.data`.
"""
function _launch_sparse!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    SI = SparseImpl

    # For the full-block case (all DOFs), all elements are active.
    # Convert DOF IDs to element IDs by looking at which elements support
    # the requested DOFs.
    test_el_ids = _dofs_to_elements(tfs, test_ids)
    trial_el_ids = _dofs_to_elements(bfs, trial_ids)

    # Call the sparse implementation's block assembly, keeping the result
    # on the GPU to avoid a costly GPU→CPU→GPU round-trip.
    matrix_d, test_active_rows, trial_active_rows = SI.assemble_block_gpu(
        biop,
        ctx.test_shapes, ctx.test_el_d, ctx.test_ad_sparse_cpu,
        ctx.trial_shapes, ctx.trial_el_d, ctx.trial_ad_sparse_cpu,
        ctx.qd_d,
        test_el_ids, trial_el_ids;
        numshapes_test=ctx.num_tshapes,
        numshapes_trial=ctx.num_bshapes,
        keep_on_gpu=true,
    )
end

#     # Scatter results from matrix_d into store.data entirely on the GPU.
#     #
#     # matrix_d is sized (length(test_active_rows), length(trial_active_rows))
#     # store.data is sized (length(test_ids), length(trial_ids))
#     #
#     # We need to map active_rows[i] → position of that DOF in test_ids/trial_ids.
#     # Build the mapping arrays on the CPU (they're small), upload, then launch
#     # a GPU kernel that does the indexed addition.

#     # Build row/col mapping: for each entry in active_rows, find its index in ids.
#     # If an active row is not in the requested ids, map it to 0 (kernel will skip).
#     test_row_map = _build_row_map(test_active_rows, test_ids)
#     trial_row_map = _build_row_map(trial_active_rows, trial_ids)

#     test_row_map_d = CuArray(test_row_map)
#     trial_row_map_d = CuArray(trial_row_map)

#     # Launch GPU scatter kernel
#     _gpu_scatter_add!(store.data, matrix_d, test_row_map_d, trial_row_map_d)

#     return
# end


# ===========================================================================
# Hybrid Global — launch wrapper
# ===========================================================================

"""
    _launch_hybrid_global!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)

Launch the two-phase hybrid kernel with Z_padded in global memory.
Requires ctx.tad_gpu to be a HybridAssemblyData (built when kernel ∈ (:hybrid_global, :hybrid_shared)).
"""
function _launch_hybrid_global!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    had = ctx.tad_gpu  # HybridAssemblyData{T}

    active_test_ids, active_trial_ids =
        filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)
    test_id_map, trial_id_map =
        create_id_maps(test_ids, trial_ids, ctx.num_tfs, ctx.num_bfs)

    # Local row/col → global DOF. Phase 2 needs this to index coeff_*_padded
    # (global-DOF-indexed) correctly when test_ids/trial_ids are arbitrary subsets.
    test_dof_ids_dev = CUDA.cu(Int32.(test_ids))
    trial_dof_ids_dev = CUDA.cu(Int32.(trial_ids))

    gpu_hybrid_global!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        active_test_ids, active_trial_ids,
        had,
        test_id_map, trial_id_map,
        ctx.quaddata_gpu,
        ctx.num_tshapes, ctx.num_bshapes,
        test_dof_ids_dev, trial_dof_ids_dev,
    )
    return
end


# ===========================================================================
# Hybrid Shared — launch wrapper
# ===========================================================================

"""
    _launch_hybrid_shared!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)

Launch the fused biphasic kernel with Z_tile in shared memory.
Requires ctx.tad_gpu to be a HybridAssemblyData (built when kernel ∈ (:hybrid_global, :hybrid_shared)).
"""
function _launch_hybrid_shared!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
    had = ctx.tad_gpu  # HybridAssemblyData{T}

    test_id_map, trial_id_map =
        create_id_maps(test_ids, trial_ids, ctx.num_tfs, ctx.num_bfs)

    test_dof_ids_dev = CUDA.cu(Int32.(test_ids))
    trial_dof_ids_dev = CUDA.cu(Int32.(trial_ids))

    gpu_hybrid_shared!(
        store.data, biop, ctx.test_shapes, ctx.trial_shapes,
        ctx.test_elements_dev, ctx.trial_elements_dev,
        had,
        test_id_map, trial_id_map,
        ctx.quaddata_gpu,
        ctx.num_tshapes, ctx.num_bshapes,
        tfs, bfs, test_ids, trial_ids,
        test_dof_ids_dev, trial_dof_ids_dev,
    )
    return
end


# #Warp scatter launch

# function _launch_warp_scatter!(biop, tfs, test_ids, bfs, trial_ids, ctx, store)
#     test_id_dev, trial_id_dev =
#         filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)
#     test_id_map_dev, trial_id_map_dev = create_id_maps(test_ids, trial_ids, ctx.num_tfs, ctx.num_bfs)

#     datatype = scalartype(biop, tfs, bfs)
#     num_test_elements = length(ctx.test_elements)
#     num_trial_elements = length(ctx.trial_elements)

#     plan = build_warp_scatter_plan(test_ids,
#         trial_ids, ctx.tad_gpu, ctx.bad_gpu,
#         num_test_elements, num_trial_elements,
#         ctx.num_tshapes, ctx.num_bshapes,
#         datatype)

#         gpu_warp_scatter!(
#         store,
#             biop, ctx.test_shapes, ctx.trial_shapes,
#             ctx.test_elements_dev, ctx.trial_elements_dev,
#             plan,
#             ctx.quaddata_gpu,
#             ctx.num_tshapes, ctx.num_bshapes
#         )
#     return
# end

# """
#     _build_row_map(active_rows, ids) -> Vector{Int32}

# For each entry `active_rows[i]`, find its position in `ids`.
# Returns a vector of the same length as `active_rows` where entry `i`
# contains the index of `active_rows[i]` in `ids`, or 0 if not found.
# """
# function _build_row_map(active_rows, ids)
#     id_set = Dict{Int,Int32}()
#     for (i, id) in enumerate(ids)
#         id_set[id] = Int32(i)
#     end
#     return Int32[get(id_set, r, Int32(0)) for r in active_rows]
# end


# """
#     _gpu_scatter_add!(dst, src, row_map, col_map)

# GPU kernel: for each (i,j) in `src`, if `row_map[i] > 0` and `col_map[j] > 0`,
# add `src[i,j]` into `dst[row_map[i], col_map[j]]`.

# Uses atomicAdd to avoid race conditions when multiple (i,j) map to the same
# `dst` entry (unlikely in practice but possible for overlapping DOFs).
# """
# function _gpu_scatter_add!(dst::CuMatrix{T}, src::CuMatrix{T},
#     row_map::CuVector{Int32}, col_map::CuVector{Int32}) where {T}

#     M, N = size(src)
#     threads = (16, 16)
#     blocks = (ceil(Int, M / threads[1]), ceil(Int, N / threads[2]))
#     @cuda blocks = blocks threads = threads _gpu_scatter_add_kernel!(
#         dst, src, row_map, col_map, M, N)
#     return
# end

# function _gpu_scatter_add_kernel!(dst, src, row_map, col_map, M, N)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     if i <= M && j <= N
#         ri = row_map[i]
#         cj = col_map[j]
#         if ri > 0 && cj > 0
#             v = src[i, j]

#             dst_f = reinterpret(Float64, dst)
#             nrows = size(dst, 1)
#             flat = ri + (cj - 1) * nrows #colmajor
#             CUDA.@atomic dst_f[2*flat-1] += v.re
#             CUDA.@atomic dst_f[2*flat] += v.im
#         end
#     end
#     return nothing
# end

function _dofs_to_elements(X, dof_ids)
    el_ids = Int[]
    for m in dof_ids
        for sh in X.fns[m]
            push!(el_ids, sh.cellid)
        end
    end
    return unique!(sort!(el_ids))
end


# # Different quad strategy types use different field names for the far-field
# # quadrature rule. These helpers provide a uniform interface.
 _far_outer_rule(qs) = hasproperty(qs, :outer_rule_far) ? qs.outer_rule_far : qs.outer_rule

 _far_inner_rule(qs) = hasproperty(qs, :inner_rule_far) ? qs.inner_rule_far : qs.inner_rule
