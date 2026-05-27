"""
Gathering based implementation 1
Issue with this impl: only one output entry per block, many non-full blocks scheduled
"""


function block_reduce_sum(x::T) where T
    # warp-level reduction cf. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-primitives
    # this is only for intra-warp. intra-block / inter-warp we need shared memory cf. next lines
    for offset in (Int32(16), Int32(8), Int32(4), Int32(2), Int32(1))
        x += shfl_down_sync(0xffffffff, x, offset)
    end
    # one value per warp survives in lane 0; reduce across warps via shared memory
    shared = @cuStaticSharedMem(T, 32)        # max 32 warps per block
    lane = (threadIdx().x - Int32(1)) & Int32(31) # 0 if this is the warp's 0 thread TODO check if I can write this more simply
    warp = (threadIdx().x - Int32(1)) >> 5
    if lane == 0
        shared[warp+Int32(1)] = x # Julia 1-indexing
    end
    sync_threads() # sync threads within blocks: all warps have written their reduced value to shared memory now
    nwarps = div(blockDim().x + Int32(31), Int32(32))
    if warp == 0
        # This if is to prevent out of bounds access to shared memory if the block has fewer than 32 warps.
        # let's say lane = 31 and nwarps = 20, we would try to access shared[32] at some point which contains nonsense.
        x = (lane < nwarps) ? shared[lane+Int32(1)] : zero(T)
        for offset in (Int32(16), Int32(8), Int32(4), Int32(2), Int32(1)) # Same treelike reduction, still log(n) worst case
            x += shfl_down_sync(0xffffffff, x, offset)
        end
    end
    return x
end

function gather_reduce_kernel!(
    output,                # CuMatrix{T}, output rows × cols
    op, test_shapes, trial_shapes,
    test_elements, bsis_elements,
    inv_tad_flat::CuDeviceVector{Tuple{Int32,Int32,T},1}, inv_tad_offsets, inv_tad_lengths,
    inv_bad_flat::CuDeviceVector{Tuple{Int32,Int32,T},1}, inv_bad_offsets, inv_bad_lengths,
    test_id_map, trial_id_map,     # block-local → global dof
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
) where {T}
    # block to grid coords: which (m, n) does this block compute?
    m_local = blockIdx().x
    n_local = blockIdx().y

    # TODO; should i add inbounds calls?
    # Get global dof indices for this block's output entry
    m_global = test_id_map[m_local]
    n_global = trial_id_map[n_local]

    # Number of contributing (p, i, a) and (q, j, b)
    # TODO; should i add inbounds calls?
    t_off, t_len = inv_tad_offsets[m_global], inv_tad_lengths[m_global]
    b_off, b_len = inv_bad_offsets[n_global], inv_bad_lengths[n_global]

    # Each thread takes a slice of the (t_len × b_len) cross product. TODO: huh?
    tid = threadIdx().x
    nthreads = blockDim().x
    total = t_len * b_len

    acc = zero(T)
    k = tid - Int32(1)
    while k < total
        ti = div(k, b_len) + Int32(1)
        bi = mod(k, b_len) + Int32(1)
        @inbounds (p, i, a) = inv_tad_flat[t_off+ti-Int32(1)]
        @inbounds (q, j, b) = inv_bad_flat[b_off+bi-Int32(1)]

        # Compute z_{ij}^{(p,q)} on the fly — full quadrature loop
        z_ij = compute_pair_entry(T, op, test_shapes, trial_shapes,
            test_elements[p], bsis_elements[q],
            i, j,
            tqp_flat, tqp_offsets[p], tqp_lengths[p],
            bqp_flat, bqp_offsets[q], bqp_lengths[q])

        acc += a * z_ij * b
        k += nthreads # TODO: think about the stride conceptually: is it actually good for CMA?
    end

    # ─── reduction ───
    acc = block_reduce_sum(acc)            # tree reduction in shared memory or via warp shuffles

    if tid == Int32(1)
        @inbounds output[m_local, n_local] = acc     # one writer per block, no race conditions
    end
    return
end
