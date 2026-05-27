using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @cuStaticSharedMem,
    @inbounds, synchronize


"""
    hybrid_shared_kernel!(...)

Fused biphasic kernel with Z_tile in shared memory.
"""
function hybrid_shared_kernel!(
    output::CuDeviceMatrix{T},
    op, test_shapes, trial_shapes,
    test_elements, trial_elements,
    fwd_tad_flat, fwd_tad_offsets, fwd_tad_lengths,
    fwd_bad_flat, fwd_bad_offsets, fwd_bad_lengths,
    coeff_test_padded::CuDeviceMatrix{CT},
    coeff_trial_padded::CuDeviceMatrix{CB},
    test_id_map::CuDeviceVector{Int32},
    trial_id_map::CuDeviceVector{Int32},
    test_dof_ids::CuDeviceVector{Int32},     # local row m → global DOF
    trial_dof_ids::CuDeviceVector{Int32},    # local col n → global DOF
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    pair_flat, pair_off,
    M_block::Int32, N_block::Int32, n_tiles_m::Int32,
    ::Val{NS}, ::Val{MS},
    ::Val{TILE}, ::Val{K_max_t}, ::Val{K_max_b},
) where {T,CT,CB,NS,MS,TILE,K_max_t,K_max_b}

    LD = TILE + Int32(1)
    Z_tile = @cuStaticSharedMem(T, (LD, K_max_t, K_max_b, TILE))

    m_loc = threadIdx().x   # 1..TILE
    n_loc = threadIdx().y   # 1..TILE
    tid_linear = (n_loc - Int32(1)) * Int32(TILE) + m_loc
    num_threads = Int32(TILE) * Int32(TILE)

    # Global output indices for this thread
    m = (blockIdx().x - Int32(1)) * Int32(TILE) + m_loc
    n = (blockIdx().y - Int32(1)) * Int32(TILE) + n_loc

    in_bounds = (m <= M_block) & (n <= N_block)

    total_tile_entries = Int32(TILE) * Int32(K_max_t) * Int32(K_max_b) * Int32(TILE)
    idx = tid_linear
    while idx <= total_tile_entries
        m_z = mod(idx - Int32(1), Int32(TILE)) + Int32(1)
        rem1 = div(idx - Int32(1), Int32(TILE))
        k_z = mod(rem1, Int32(K_max_t)) + Int32(1)
        rem2 = div(rem1, Int32(K_max_t))
        l_z = mod(rem2, Int32(K_max_b)) + Int32(1)
        n_z = div(rem2, Int32(K_max_b)) + Int32(1)
        @inbounds Z_tile[m_z, k_z, l_z, n_z] = zero(T)
        idx += num_threads
    end
    CUDA.sync_threads()

    tile_idx = blockIdx().x + (blockIdx().y - Int32(1)) * n_tiles_m
    @inbounds pair_lo = pair_off[tile_idx]
    @inbounds pair_hi = pair_off[tile_idx+Int32(1)] - Int32(1)
    npairs = pair_hi - pair_lo + Int32(1)

    m_start = (blockIdx().x - Int32(1)) * Int32(TILE) + Int32(1)
    n_start = (blockIdx().y - Int32(1)) * Int32(TILE) + Int32(1)

    pair_iter = tid_linear
    while pair_iter <= npairs
        @inbounds current_pair = pair_lo + pair_iter - Int32(1)
        @inbounds (p, q) = pair_flat[current_pair]

        @inbounds tcell = test_elements[p]
        @inbounds bcell = trial_elements[q]

        @inbounds t_off = tqp_offsets[p]
        @inbounds t_len = tqp_lengths[p]
        @inbounds b_off = bqp_offsets[q]
        @inbounds b_len = bqp_lengths[q]

        z = accumulate_zlocal_ntuple(T, op, test_shapes, trial_shapes,
            tcell, bcell,
            tqp_flat, t_off, t_len,
            bqp_flat, b_off, b_len,
            Val(NS), Val(MS))

        @inbounds for i in Int32(1):Int32(NS)
            t_ad_off = fwd_tad_offsets[p, i]
            t_ad_len = fwd_tad_lengths[p, i]
            for j in Int32(1):Int32(MS)
                flat_idx = (j - Int32(1)) * Int32(NS) + i
                zval = z[flat_idx]

                b_ad_off = fwd_bad_offsets[q, j]
                b_ad_len = fwd_bad_lengths[q, j]

                ti = Int32(0)
                while ti < t_ad_len
                    (m_global, k, a_coeff) = fwd_tad_flat[t_ad_off+ti]
                    m_local = test_id_map[m_global]

                    if m_local != Int32(0)
                        m_l = m_local - m_start + Int32(1)
                        if m_l >= Int32(1) && m_l <= Int32(TILE)
                            bi = Int32(0)
                            while bi < b_ad_len
                                (n_global, l, b_coeff) = fwd_bad_flat[b_ad_off+bi]
                                n_local = trial_id_map[n_global]

                                if n_local != Int32(0)
                                    n_l = n_local - n_start + Int32(1)
                                    if n_l >= Int32(1) && n_l <= Int32(TILE)
                                        # No atomic: unique (m,k,n,l) per (p,i,q,j)
                                        Z_tile[m_l, k, l, n_l] = zval
                                    end
                                end

                                bi += Int32(1)
                            end
                        end
                    end

                    ti += Int32(1)
                end
            end
        end

        pair_iter += num_threads
    end

    CUDA.sync_threads()

    if in_bounds
        @inbounds m_global = test_dof_ids[m]
        @inbounds n_global = trial_dof_ids[n]
        acc = zero(T)

        @inbounds for k in Int32(1):Int32(K_max_t)
            a = coeff_test_padded[m_global, k]
            for l in Int32(1):Int32(K_max_b)
                b = coeff_trial_padded[n_global, l]
                acc += a * Z_tile[m_loc, k, l, n_loc] * b
            end
        end

        @inbounds output[m, n] = acc
    end

    return nothing
end


"""
    _pick_hybrid_shared_tile(K_max_t, K_max_b, sizeof_T; limit_bytes=167936) -> Int

Choose the largest supported tile size that fits within `limit_bytes` of static
shared memory for the given problem.

Returns the chosen tile from `{16, 8, 4}` (powers of 2 → clean warp packing),
or `0` if even the smallest tile does not fit (caller should fall back to
:hybrid_global).
"""
function _pick_hybrid_shared_tile(K_max_t::Int, K_max_b::Int, sizeof_T::Int;
                                  limit_bytes::Int=167936)
    for tile in (16, 8, 4)
        if (tile + 1) * K_max_t * K_max_b * tile * sizeof_T <= limit_bytes
            return tile
        end
    end
    return 0
end


"""
    gpu_hybrid_shared!(output, biop, test_shapes, trial_shapes,
        test_elements_dev, trial_elements_dev, had,
        test_id_map, trial_id_map, quaddata_gpu,
        num_tshapes, num_bshapes, tfs, bfs, test_ids, trial_ids,
        test_dof_ids, trial_dof_ids)

Launch the fused biphasic kernel with an adaptively chosen tile size.
"""
function gpu_hybrid_shared!(
    output::CuMatrix{T},
    biop, test_shapes, trial_shapes,
    test_elements_dev, trial_elements_dev,
    had::HybridAssemblyData{T},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    quaddata_gpu,
    num_tshapes::Int, num_bshapes::Int,
    tfs, bfs, test_ids, trial_ids,
    test_dof_ids::CuVector{Int32},
    trial_dof_ids::CuVector{Int32},
) where {T}

    K_max_t = Int(had.K_max_test)
    K_max_b = Int(had.K_max_trial)

    tile = _pick_hybrid_shared_tile(K_max_t, K_max_b, sizeof(T))
    if tile == 0
        smallest = (4 + 1) * K_max_t * K_max_b * 4 * sizeof(T)
        error("hybrid_shared: even tile=4 needs $(smallest) bytes of shared memory " *
              "(> 49152 limit). K_max_t=$K_max_t, K_max_b=$K_max_b, sizeof(T)=$(sizeof(T)). " *
              "Use :hybrid_global instead.")
    end

    # Dispatch to a specialized launch with Val(tile) so kernel's static
    # shared-memory declaration sees compile-time constant. Each branch
    # compiles a separate kernel specialization.
    if tile == 16
        _launch_hybrid_shared_tile!(output, biop, test_shapes, trial_shapes,
            test_elements_dev, trial_elements_dev, had,
            test_id_map, trial_id_map, quaddata_gpu,
            num_tshapes, num_bshapes,
            tfs, bfs, test_ids, trial_ids,
            test_dof_ids, trial_dof_ids, Val(16))
    elseif tile == 8
        _launch_hybrid_shared_tile!(output, biop, test_shapes, trial_shapes,
            test_elements_dev, trial_elements_dev, had,
            test_id_map, trial_id_map, quaddata_gpu,
            num_tshapes, num_bshapes,
            tfs, bfs, test_ids, trial_ids,
            test_dof_ids, trial_dof_ids, Val(8))
    else  # tile == 4
        _launch_hybrid_shared_tile!(output, biop, test_shapes, trial_shapes,
            test_elements_dev, trial_elements_dev, had,
            test_id_map, trial_id_map, quaddata_gpu,
            num_tshapes, num_bshapes,
            tfs, bfs, test_ids, trial_ids,
            test_dof_ids, trial_dof_ids, Val(4))
    end

    return output
end


# Tile-specialized launch one / Val(TILE) specialization
@inline function _launch_hybrid_shared_tile!(
    output::CuMatrix{T},
    biop, test_shapes, trial_shapes,
    test_elements_dev, trial_elements_dev,
    had::HybridAssemblyData{T},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    quaddata_gpu,
    num_tshapes::Int, num_bshapes::Int,
    tfs, bfs, test_ids, trial_ids,
    test_dof_ids::CuVector{Int32},
    trial_dof_ids::CuVector{Int32},
    ::Val{TILE},
) where {T,TILE}

    M_block = Int32(size(output, 1))
    N_block = Int32(size(output, 2))
    K_max_t = Int(had.K_max_test)
    K_max_b = Int(had.K_max_trial)
    n_tiles_m = cld(Int(M_block), TILE)
    n_tiles_n = cld(Int(N_block), TILE)

    pair_flat, pair_off = build_tile_pairs(test_ids, trial_ids, tfs, bfs, TILE, TILE)

    @cuda threads = (TILE, TILE) blocks = (n_tiles_m, n_tiles_n) hybrid_shared_kernel!(
        output, biop, test_shapes, trial_shapes,
        test_elements_dev, trial_elements_dev,
        had.fwd_tad.flat, had.fwd_tad.offsets, had.fwd_tad.lengths,
        had.fwd_bad.flat, had.fwd_bad.offsets, had.fwd_bad.lengths,
        had.coeff_test_padded, had.coeff_trial_padded,
        test_id_map, trial_id_map,
        test_dof_ids, trial_dof_ids,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        pair_flat, pair_off,
        M_block, N_block, Int32(n_tiles_m),
        Val(num_tshapes), Val(num_bshapes),
        Val(TILE), Val(K_max_t), Val(K_max_b),
    )

    CUDA.synchronize()
    return output
end
