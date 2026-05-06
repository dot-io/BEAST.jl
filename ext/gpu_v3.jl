# Gathering-based implementation 2: only one thread per matrix output entry

function tile_gather_kernel!(
    output,                     # CuMatrix{T} of size (M_block, N_block)
    op, test_shapes, trial_shapes,
    test_elements, bsis_elements,
    inv_tad::InvAssemblyData{T},   # m → list of (p, i, a)
    inv_bad::InvAssemblyData{T},   # n → list of (q, j, b)
    test_id_map, trial_id_map,
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    M_block::Int32, N_block::Int32,
) where {T}
    # Tile this block owns (m + ..., n + ...)
    tile_m_start = (blockIdx().x - Int32(1)) * blockDim().x + Int32(1)
    tile_n_start = (blockIdx().y - Int32(1)) * blockDim().y + Int32(1)

    # Output entry this thread owns
    m_local = tile_m_start + threadIdx().x - Int32(1)
    n_local = tile_n_start + threadIdx().y - Int32(1)

    # prevent out-of-bounds accesses
    (m_local > M_block || n_local > N_block) && return

    @inbounds m_global = test_id_map[m_local]
    @inbounds n_global = trial_id_map[n_local]

    @inbounds t_off = inv_tad.offsets[m_global]
    @inbounds t_len = inv_tad.lengths[m_global]
    @inbounds b_off = inv_bad.offsets[n_global]
    @inbounds b_len = inv_bad.lengths[n_global]

    acc = zero(T)
    ti = Int32(0)
    while ti < t_len
        @inbounds (p, i, a) = inv_tad.flat[t_off + ti] # (test element, test shape fn. index, weight)
        bi = Int32(0)
        while bi < b_len
            @inbounds (q, j, b) = inv_bad.flat[b_off + bi] # (trial element, basis shape fn. index, weight)
            z_ij = compute_pair_entry(T,
                op, test_shapes, trial_shapes,
                test_elements[p], bsis_elements[q],
                i, j,
                tqp_flat, tqp_offsets[p], tqp_lengths[p],
                bqp_flat, bqp_offsets[q], bqp_lengths[q],
            )
            acc += a * z_ij * b
            bi += Int32(1)
        end
        ti += Int32(1)
    end

    @inbounds output[m_local, n_local] = acc   # TODO: prove no race condition
    return
end
