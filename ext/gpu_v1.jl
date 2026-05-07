# Scattering-based implementation

using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @cuStaticSharedMem, @inbounds,
    @atomic, synchronize


"""
    scatter_kernel!(...)

Tiled scatter kernel with shared memory reduction.
Grid consists of num_tiles_m x num_tiles_n blocks, where each block handles one output tile
- Block: blockDim.x threads, where blockDim.x = TILE_SIZE² (if possible)
- TILE_SIZE: Derived from blockDim.x as floor(sqrt(blockDim.x))
1. Each thread initializes one element of shared tile to zero
2. All threads cooperatively scan zlocals (grid-stride loop):
    - Decode (pair, i, j) → (p, q, i, j)
    - Look up AssemblyData for (p,i) and (q,j)
    - For each (m', n') contribution:
        - If (m', n') falls in this block's tile:
            - Atomic add to shared tile[m_tile, n_tile]

Phase 3 (Write back):
  Sync threads
  Each thread writes one tile element to global output (atomic)
```

shared memory atomic ops are faster than global memory atomics because:
- They dont waste cycles accessing global memory
- Better coalescing for final global write
"""
function scatter_kernel!(
    output::CuDeviceMatrix{T},
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
) where {T}
    """
    This tile is instantiated with a maximum size of 32x32 = 1024. The reason for this is twofold:
    1. CUDA shared memory needs to be defined at compile-time, at which point TILE_SIZE being derived from blockDim is yet unknown.
    2. Having CUDA determine threads per block is optimal and hardware-independent, so we can incur a small penalty in the storage department if this means faster execution time.

    But in practice launching the max of 1024 threads should be fine.
    """

    # Split shared memory into real and imaginary parts so that
    # CUDA.@atomic can be used (atomics on ComplexF64 are unsupported)
    F = real(T)
    tile_re = @cuStaticSharedMem F (32, 32)
    tile_im = @cuStaticSharedMem F (32, 32)

    tid = threadIdx().x

    tile_row = mod(tid - Int32(1), TILE_SIZE) + Int32(1)
    tile_col = div(tid - Int32(1), TILE_SIZE) + Int32(1)

    """
    First initialization to 0 in order to be able to start accumulating
    """

    if tid <= TILE_SIZE_SQ
        @inbounds tile_re[tile_row, tile_col] = zero(F)
        @inbounds tile_im[tile_row, tile_col] = zero(F)
    end
    sync_threads()

    tile_m_start = (blockIdx().x - Int32(1)) * TILE_SIZE + Int32(1)
    tile_n_start = (blockIdx().y - Int32(1)) * TILE_SIZE + Int32(1)

    total_pairs = num_test * num_trial
    MN = num_tshapes * num_bshapes
    total_zlocals = total_pairs * MN

    stride = blockDim().x

    linear_zlocal = tid

    """
    I.e. while the thread is still within bounds of the "zlocals" array. The
    reason why I am using the strided access pattern is because of Contiguous
    Memory Access: threads within a warp accessing neighbouring data, which the
    SM can do in a coalesced fashion.

    c.f. https://stackoverflow.com/questions/19505404/performance-of-atomic-operations-on-shared-memory

    for a detailed explanation on how atomics operate on shared memory. TLDR:
    if there is intra-block / intra-tile (idem.) contention for a memory
    location, warp divergence happens hindering performance.
    """
    while linear_zlocal <= total_zlocals
        """
        Index conversion from thread index within block (corresponding to flattened row-wise 'zlocals' index) --> to the pair_idx (element pair index) and i,j (zlocals 2D index)
        """
        pair_idx = div(linear_zlocal - Int32(1), MN) + Int32(1)
        rem_ij = mod(linear_zlocal - Int32(1), MN)
        i = mod(rem_ij, num_tshapes) + Int32(1)
        j = div(rem_ij, num_tshapes) + Int32(1)

        p_local = div(pair_idx - Int32(1), num_trial) + Int32(1)
        q_local = mod(pair_idx - Int32(1), num_trial) + Int32(1)

        """
        Gets the test and basis function indices from the pair index.
        """
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

    if tid <= TILE_SIZE_SQ
        m_global = tile_m_start + tile_row - Int32(1)
        n_global = tile_n_start + tile_col - Int32(1)

        if m_global <= output_rows && n_global <= output_cols
            @inbounds val_re = tile_re[tile_row, tile_col]
            @inbounds val_im = tile_im[tile_row, tile_col]
            if val_re != zero(F) || val_im != zero(F)
                CUDA.@atomic output[m_global, n_global] += complex(val_re, val_im)
            end
        end
    end

    return nothing
end

"""
    gpu_scatter!(output, zlocals_all, tad, bad, active_test_ids, active_trial_ids,
                 test_id_map, trial_id_map, num_tshapes, num_bshapes; use_tiled=true)

High-level interface for GPU scatter operation.
- output: output matrix block to fill (TODO: how should i instantiate this? device memory copy is a good start but full instantiation on GPU is gold standard)
- zlocals_all: CuArray{T,3} - local integrals (M x N x num_pairs)
- tad, bad: FlattenedAssemblyData{T} - test/trial assembly data
- active_test_ids, active_trial_ids: active element IDs for this block
- test_id_map, trial_id_map: global → local-in-block index mapping
- num_tshapes, num_bshapes: number of local shape functions
- use_tiled: if true, use tiled kernel for large outputs (default: true)
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
) where {T}

    num_test = length(active_test_ids)
    num_trial = length(active_trial_ids)
    output_rows = size(output, 1)
    output_cols = size(output, 2)

    threads_per_block = TILE_SIZE * TILE_SIZE
    num_tiles_m = cld(output_rows, TILE_SIZE)
    num_tiles_n = cld(output_cols, TILE_SIZE)

    @cuda threads = threads_per_block blocks = (num_tiles_m, num_tiles_n) scatter_kernel!(
        output,
        zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids,
        active_trial_ids,
        test_id_map,
        trial_id_map,
        Int32(num_tshapes),
        Int32(num_bshapes),
        Int32(num_test),
        Int32(num_trial),
        Int32(output_rows),
        Int32(output_cols),
    )
    CUDA.synchronize()
    return output
end

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

function momintegrals!(
    output::CuDeviceArray{T, 3},
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
                @inbounds womp = tqp_flat[o_off + oi]
                tgeo  = womp.point
                tvals = womp.value
                jx    = womp.weight

                ii = Int32(0)
                while ii < i_len
                    @inbounds wimp = bqp_flat[i_off + ii]
                    bgeo  = wimp.point
                    bvals = wimp.value
                    jy    = wimp.weight

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

export scatter_kernel
