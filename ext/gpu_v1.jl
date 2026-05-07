# Scattering-based implementation
#
# Two kernel variants:
#   v1.0  scatter_kernel_direct!   — atomic add directly into global output
#   v1.5  scatter_kernel_tiled!    — shared-memory tile reduction before
#                                    global write-back
#
# Both kernels share the same scatter logic: each thread owns one zlocal entry
# (or strides over several) and expands it into contributions to the output
# matrix via the AssemblyData maps.  The difference is *where* the atomic
# accumulation happens.

using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @cuStaticSharedMem, @inbounds,
    @atomic, synchronize


# ===========================================================================
# v1.0 — Direct global-memory scatter
# ===========================================================================
#
# Each thread processes one zlocal[i, j, pair_idx] entry (grid-stride loop)
# and atomically adds its contribution directly to the global output matrix.
#
# Pros:
#   • Simple, easy to verify correct
#   • Each zlocal is read exactly once → minimal memory traffic
#   • For RT basis (fan-out ≈ 1) there is negligible atomic contention
#
# Cons:
#   • Every contribution hits global memory via atomic → higher latency
#     per atomic than shared-memory atomic
#   • If many zlocals map to the same output entry the atomics serialise
#
# For typical BEM problems with RT basis each zlocal contributes to at most
# one output entry, so contention is near-zero and v1.0 is hard to beat.

"""
    scatter_kernel_direct!(output_re, output_im, zlocals_all, tad_flat, tad_offsets, tad_lengths,
                           bad_flat, bad_offsets, bad_lengths,
                           active_test_ids, active_trial_ids,
                           test_id_map, trial_id_map,
                           num_tshapes, num_bshapes,
                           num_test, num_trial,
                           output_rows, output_cols)

Direct scatter kernel: each thread expands one zlocal entry and atomically
adds its contributions into the split real/imag output buffers.
"""
function scatter_kernel_direct!(
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

    total_pairs = num_test * num_trial
    MN = num_tshapes * num_bshapes
    total_zlocals = total_pairs * MN

    # Grid-stride loop: each thread walks zlocals with stride = total threads
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    linear_zlocal = tid

    while linear_zlocal <= total_zlocals
        # Decode linear index → (pair_idx, i, j)
        pair_idx = div(linear_zlocal - Int32(1), MN) + Int32(1)
        rem_ij = mod(linear_zlocal - Int32(1), MN)
        i = mod(rem_ij, num_tshapes) + Int32(1)
        j = div(rem_ij, num_tshapes) + Int32(1)

        # Decode pair_idx → (p_local, q_local) → element ids (p, q)
        # Must match momintegrals! encoding: idx = p_local + (q_local-1)*num_test
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
                    bi = Int32(0)
                    while bi < b_len
                        @inbounds (n_global, b_coeff) = bad_flat[b_off+bi]
                        @inbounds n_local = trial_id_map[n_global]

                        if n_local != Int32(0)
                            contribution = a_coeff * zval * b_coeff
                            CUDA.@atomic output_re[m_local, n_local] += real(contribution)
                            CUDA.@atomic output_im[m_local, n_local] += imag(contribution)
                        end

                        bi += Int32(1)
                    end
                end

                ti += Int32(1)
            end
        end

        linear_zlocal += stride
    end

    return nothing
end


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


# ===========================================================================
# High-level launch wrapper
# ===========================================================================

"""
    gpu_scatter!(output, zlocals_all, tad, bad, active_test_ids, active_trial_ids,
                 test_id_map, trial_id_map, num_tshapes, num_bshapes;
                 variant=:direct)

High-level interface for GPU scatter operation.

# Arguments
- `output`:       output matrix block (CuMatrix{T})
- `zlocals_all`:  local integrals (CuArray{T,3}, size num_tshapes × num_bshapes × num_pairs)
- `tad`, `bad`:   FlattenedAssemblyData{T} for test/trial
- `active_test_ids`, `active_trial_ids`: active element IDs
- `test_id_map`, `trial_id_map`: global → local-in-block index mapping
- `num_tshapes`, `num_bshapes`: number of local shape functions

# Keyword
- `variant::Symbol`: `:direct` (default, v1.0) or `:tiled` (v1.5)

Use `:direct` for low-contention problems (typical for RT basis).
Use `:tiled` only if profiling shows high global-memory atomic contention.
"""
function gpu_scatter!(
    output::CuMatrix{T},
    zlocals_all::CuArray{T,3},
    tad::FlattenedAssemblyData{T},
    bad::FlattenedAssemblyData{T},
    active_test_ids::CuVector{Int32},
    active_trial_ids::CuVector{Int32},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    num_tshapes::Int,
    num_bshapes::Int;
    variant::Symbol=:direct,
) where {T}

    F = real(T)
    num_test = length(active_test_ids)
    num_trial = length(active_trial_ids)
    output_rows = size(output, 1)
    output_cols = size(output, 2)

    # Allocate split real/imag output buffers for atomic compatibility
    # (CUDA does not support atomicAdd on ComplexF64)
    output_re = CUDA.zeros(F, output_rows, output_cols)
    output_im = CUDA.zeros(F, output_rows, output_cols)

    if variant === :direct
        _launch_scatter_direct!(
            output_re, output_im, zlocals_all, tad, bad,
            active_test_ids, active_trial_ids,
            test_id_map, trial_id_map,
            num_tshapes, num_bshapes,
            num_test, num_trial,
            output_rows, output_cols,
        )
    elseif variant === :tiled
        _launch_scatter_tiled!(
            output_re, output_im, zlocals_all, tad, bad,
            active_test_ids, active_trial_ids,
            test_id_map, trial_id_map,
            num_tshapes, num_bshapes,
            num_test, num_trial,
            output_rows, output_cols,
        )
    else
        error("gpu_scatter!: unknown variant :$variant — use :direct or :tiled")
    end

    # Combine real and imaginary parts into the complex output
    output .= complex.(output_re, output_im)

    CUDA.synchronize()
    return output
end


function _launch_scatter_direct!(
    output_re, output_im, zlocals_all, tad, bad,
    active_test_ids, active_trial_ids,
    test_id_map, trial_id_map,
    num_tshapes, num_bshapes,
    num_test, num_trial,
    output_rows, output_cols,
)
    total_pairs = Int(num_test) * Int(num_trial)
    MN = Int(num_tshapes) * Int(num_bshapes)
    total_zlocals = total_pairs * MN

    # Use launch_configuration to pick a good thread count
    kernel = @cuda launch = false scatter_kernel_direct!(
        output_re, output_im, zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Int32(output_rows), Int32(output_cols),
    )
    config = launch_configuration(kernel.fun)
    threads = min(total_zlocals, config.threads)
    blocks = cld(total_zlocals, threads)

    kernel(
        output_re, output_im, zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Int32(output_rows), Int32(output_cols);
        threads, blocks,
    )
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


# ===========================================================================
# Helper: extract active element ids and upload to device
# ===========================================================================

function filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)

    active_test_el_ids = Int32[]
    active_trial_el_ids = Int32[]

    test_id_in_blk = Dict{Int,Int}()
    trial_id_in_blk = Dict{Int,Int}()

    for (i, m) in enumerate(test_ids)
        test_id_in_blk[m] = i
    end
    for (i, m) in enumerate(trial_ids)
        trial_id_in_blk[m] = i
    end

    for m in test_ids, sh in tfs.fns[m]
        push!(active_test_el_ids, Int32(sh.cellid))
    end
    for m in trial_ids, sh in bfs.fns[m]
        push!(active_trial_el_ids, Int32(sh.cellid))
    end

    active_test_el_ids = unique!(sort!(active_test_el_ids))
    active_trial_el_ids = unique!(sort!(active_trial_el_ids))

    (isempty(active_test_el_ids) || isempty(active_trial_el_ids)) && return


    # If I remove these assertions I don't have to pass the CPU arrays anymore. This doesnt cost anything however
    # @assert maximum(active_test_el_ids) <= length(test_elements)
    # @assert maximum(active_trial_el_ids) <= length(bsis_elements)

    # Transfer active element id lists to GPU
    active_test_ids_dev = CUDA.cu(active_test_el_ids)
    active_trial_ids_dev = CUDA.cu(active_trial_el_ids)

    return active_test_ids_dev, active_trial_ids_dev
end


# ===========================================================================
# Kernel 1: per-pair integrand evaluation (shared by both scatter variants)
# ===========================================================================

function momintegrals!(
    output::CuDeviceArray{T,3},
    op,
    test_shapes,
    trial_shapes,
    test_elements,
    bsis_elements,
    active_test_ids,
    active_trial_ids,
    tad_flat, tad_offsets, tad_lengths,
    bad_flat, bad_offsets, bad_lengths,
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    num_tshapes::Int32,
    num_bshapes::Int32,
    num_test::Int32,
    num_trial::Int32,
) where {T}

    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total_pairs = num_test * num_trial
    idx > total_pairs && return

    # One thread per (p, q) element pair
    p_local = mod(idx - Int32(1), num_test) + Int32(1)
    q_local = div(idx - Int32(1), num_test) + Int32(1)

    @inbounds p = active_test_ids[p_local]
    @inbounds q = active_trial_ids[q_local]

    tcell = test_elements[p]
    bcell = bsis_elements[q]
    igd = Integrand(op, test_shapes, trial_shapes, tcell, bcell)

    o_off = tqp_offsets[p]
    o_len = tqp_lengths[p]
    i_off = bqp_offsets[q]
    i_len = bqp_lengths[q]

    # Compute each zlocal[i, j] entry independently
    i = Int32(1)
    while i <= num_tshapes
        j = Int32(1)
        while j <= num_bshapes
            acc = zero(T)

            oi = Int32(0)
            while oi < o_len
                @inbounds womp = tqp_flat[o_off+oi]
                tgeo = womp.point
                tvals = womp.value
                jx = womp.weight

                ii = Int32(0)
                while ii < i_len
                    @inbounds wimp = bqp_flat[i_off+ii]
                    bgeo = wimp.point
                    bvals = wimp.value
                    jy = wimp.weight

                    z1 = igd(tgeo, bgeo, tvals, bvals)
                    acc += (jx * jy) * z1[i, j]

                    ii += Int32(1)
                end

                oi += Int32(1)
            end

            @inbounds output[i, j, idx] = acc
            j += Int32(1)
        end
        i += Int32(1)
    end

    return nothing
end

export scatter_kernel_direct!, scatter_kernel_tiled!
