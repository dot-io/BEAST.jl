
# ===========================================================================
# v1.5 — Tiled shared-memory scatter
# ===========================================================================
#
# The output matrix is partitioned into tiles of TILE_SIZE × TILE_SIZE.
# Each block owns one tile and accumulates into shared memory, then writes
# the tile back to global memory with a single atomic per element.
#
# Shared memory atomics on ComplexF64 are NOT supported by CUDA, so we
# split into real/imag Float64 arrays (same pattern as gpu_v4.jl).
#
# IMPORTANT — when does tiling actually help?
#
#   The tiled version is only beneficial when many zlocal entries contribute
#   to the SAME tile, i.e. there is significant atomic contention on a
#   small region of the output matrix.  For RT basis the fan-out per zlocal
#   is typically 1 (each local shape on element p contributes to exactly one
#   global DOF), so contention is near-zero and v1.0 is usually faster.
#
#   The tiled version also does MORE total work: every block scans ALL
#   zlocals but only accumulates those that fall in its tile.  Total work
#   is O(num_tiles × total_zlocals) vs O(total_zlocals) for v1.0.
#
#   Bottom line: use v1.0 by default.  Switch to v1.5 only if profiling
#   shows high global-memory atomic contention for your problem.

"""
    scatter_kernel_tiled!(output_re, output_im, zlocals_all, tad_flat, tad_offsets, tad_lengths,
                          bad_flat, bad_offsets, bad_lengths,
                          active_test_ids, active_trial_ids,
                          test_id_map, trial_id_map,
                          num_tshapes, num_bshapes,
                          num_test, num_trial,
                          output_rows, output_cols)

Tiled scatter kernel: each block owns one output tile, accumulates into
shared memory (split real/imag for atomic compatibility), then writes back
to split real/imag global output buffers.
"""
function scatter_kernel_tiled!(
    output_re::CuDeviceMatrix{F},
    output_im::CuDeviceMatrix{F},
    zlocals_all::CuDeviceArray{T,3},
    tad_flat, tad_offsets, tad_lengths,
    bad_flat, bad_offsets, bad_lengths,
    active_test_ids::CuDeviceVector{Int32},
    active_trial_ids::CuDeviceVector{Int32},
    test_id_map::CuDeviceVector{Int32},
    trial_id_map::CuDeviceVector{Int32},
    num_tshapes::Int32,
    num_bshapes::Int32,
    num_test::Int32,
    num_trial::Int32,
    output_rows::Int32,
    output_cols::Int32,
) where {T,F<:AbstractFloat}

    # Split shared memory into real and imaginary parts so that
    # CUDA.@atomic can be used (atomics on ComplexF64 are unsupported
    # in shared memory).
    tile_re = @cuStaticSharedMem F (32, 32)
    tile_im = @cuStaticSharedMem F (32, 32)

    tid = threadIdx().x

    # Map linear thread id to a position within the TILE_SIZE × TILE_SIZE tile
    tile_row = mod(tid - Int32(1), TILE_SIZE) + Int32(1)
    tile_col = div(tid - Int32(1), TILE_SIZE) + Int32(1)

    # Initialise shared tile to zero
    if tid <= TILE_SIZE_SQ
        @inbounds tile_re[tile_row, tile_col] = zero(F)
        @inbounds tile_im[tile_row, tile_col] = zero(F)
    end
    sync_threads()

    # Global output indices covered by this block's tile
    tile_m_start = (blockIdx().x - Int32(1)) * TILE_SIZE + Int32(1)
    tile_n_start = (blockIdx().y - Int32(1)) * TILE_SIZE + Int32(1)

    total_pairs = num_test * num_trial
    MN = num_tshapes * num_bshapes
    total_zlocals = total_pairs * MN

    stride = blockDim().x

    linear_zlocal = tid

    # Grid-stride loop over all zlocals; only accumulate contributions
    # that fall inside this block's tile.
    while linear_zlocal <= total_zlocals
        pair_idx = div(linear_zlocal - Int32(1), MN) + Int32(1)
        rem_ij = mod(linear_zlocal - Int32(1), MN)
        i = mod(rem_ij, num_tshapes) + Int32(1)
        j = div(rem_ij, num_tshapes) + Int32(1)

        p_local = mod(pair_idx - Int32(1), num_test) + Int32(1)
        q_local = div(pair_idx - Int32(1), num_test) + Int32(1)

        @inbounds p = active_test_ids[p_local]
        @inbounds q = active_trial_ids[q_local]

        @inbounds zval = zlocals_all[i, j, pair_idx]

        if zval != zero(T)
            t_off = tad_offsets[p, i]
            t_len = tad_lengths[p, i]
            b_off = bad_offsets[q, j]
            b_len = bad_lengths[q, j]

            ti = Int32(0)
            while ti < t_len
                @inbounds (m_global, a_coeff) = tad_flat[t_off+ti]
                @inbounds m_local = test_id_map[m_global]

                if m_local != Int32(0)
                    # Check whether this DOF falls inside our tile
                    m_tile = m_local - tile_m_start + Int32(1)
                    if m_tile >= Int32(1) && m_tile <= TILE_SIZE
                        bi = Int32(0)
                        while bi < b_len
                            @inbounds (n_global, b_coeff) = bad_flat[b_off+bi]
                            @inbounds n_local = trial_id_map[n_global]

                            if n_local != Int32(0)
                                n_tile = n_local - tile_n_start + Int32(1)
                                if n_tile >= Int32(1) && n_tile <= TILE_SIZE
                                    contribution = a_coeff * zval * b_coeff
                                    CUDA.@atomic tile_re[m_tile, n_tile] += real(contribution)
                                    CUDA.@atomic tile_im[m_tile, n_tile] += imag(contribution)
                                end
                            end

                            bi += Int32(1)
                        end
                    end
                end

                ti += Int32(1)
            end
        end

        linear_zlocal += stride
    end

    sync_threads()

    # Write back: each thread writes one tile element to global output
    if tid <= TILE_SIZE_SQ
        m_global = tile_m_start + tile_row - Int32(1)
        n_global = tile_n_start + tile_col - Int32(1)

        if m_global <= output_rows && n_global <= output_cols
            @inbounds val_re = tile_re[tile_row, tile_col]
            @inbounds val_im = tile_im[tile_row, tile_col]
            if val_re != zero(F) || val_im != zero(F)
                CUDA.@atomic output_re[m_global, n_global] += val_re
                CUDA.@atomic output_im[m_global, n_global] += val_im
            end
        end
    end

    return nothing
end

function _launch_scatter_tiled!(
    output_re, output_im, zlocals_all, tad, bad,
    active_test_ids, active_trial_ids,
    test_id_map, trial_id_map,
    num_tshapes, num_bshapes,
    num_test, num_trial,
    output_rows, output_cols,
)
    threads_per_block = TILE_SIZE * TILE_SIZE
    num_tiles_m = cld(output_rows, TILE_SIZE)
    num_tiles_n = cld(output_cols, TILE_SIZE)

    @cuda threads = threads_per_block blocks = (num_tiles_m, num_tiles_n) scatter_kernel_tiled!(
        output_re, output_im, zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Int32(output_rows), Int32(output_cols),
    )
end
