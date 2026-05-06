# Scattering-based implementation

using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @cuStaticSharedMem, @inbounds,
    @atomic, synchronize


"""
GPU scatter kernel for assembling local moment integrals into global matrix blocks.

This module provides:
- `FlattenedAssemblyData`: GPU-compatible representation of AssemblyData
- `scatter_kernel!`: CUDA kernel for local→global scatter with shared memory reduction
- `gpu_scatter!`: High-level interface for GPU-based scatter operation

## Design Rationale

The scatter phase maps local shape function integrals (computed per element pair)
to global basis function indices. For higher-order bases, one local shape can
contribute to multiple global functions.

We use a tiled approach with shared memory reduction:
1. Output matrix partitioned into tiles (TILE_SIZE × TILE_SIZE)
2. Each thread block handles contributions destined for one tile
3. All threads cooperatively scan zlocals, filtering for their tile's contributions
4. Shared memory accumulates tile contributions (reduces global memory traffic)
5. Final write to global memory (one atomic per tile element)

Advantages:
- Parallel I/O: multiple blocks work on different tiles simultaneously
- Memory hierarchy: shared memory is much faster than global memory
- Reduced contention: only one atomic per output element (instead of many)
"""

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
    output::CuMatrix{T},
    zlocals_all::CuArray{T,3},
    tad::FlattenedAssemblyData{T},
    bad::FlattenedAssemblyData{T},
    active_test_ids::CuVector{Int32},
    active_trial_ids::CuVector{Int32},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
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

    But in practice launching the max of 1024 thread should be fine.
    """

    tile = @cuStaticSharedMem T (32, 32)

    tid = threadIdx().x

    tile_row = mod(tid - Int32(1), TILE_SIZE) + Int32(1)
    tile_col = div(tid - Int32(1), TILE_SIZE) + Int32(1)

    """
    First initialization to 0 in order to be able to start accumulating
    """

    if tid <= TILE_SIZE_SQ
        @inbounds tile[tile_row, tile_col] = zero(T)
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
            t_off = tad.offsets[p, i]
            t_len = tad.lengths[p, i]
            b_off = bad.offsets[q, j]
            b_len = bad.lengths[q, j]

            ti = Int32(0)
            while ti < t_len
                @inbounds (m_global, a_coeff) = tad.flat[t_off+ti]
                @inbounds m_local = test_id_map[m_global]

                if m_local != Int32(0)
                    m_tile = m_local - tile_m_start + Int32(1)
                    if m_tile >= Int32(1) && m_tile <= TILE_SIZE
                        bi = Int32(0)
                        while bi < b_len
                            @inbounds (n_global, b_coeff) = bad.flat[b_off+bi]
                            @inbounds n_local = trial_id_map[n_global]

                            if n_local != Int32(0)
                                n_tile = n_local - tile_n_start + Int32(1)
                                if n_tile >= Int32(1) && n_tile <= TILE_SIZE
                                    contribution = a_coeff * zval * b_coeff
                                    CUDA.@atomic tile[m_tile, n_tile] +=
                                        contribution
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
            @inbounds val = tile[tile_row, tile_col]
            if val != zero(T)
                CUDA.@atomic output[m_global, n_global] += val
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
        tad,
        bad,
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

export gpu_scatter!

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

function assembleblock_body_gpu!(
    biop,
    tfs,
    test_ids,
    test_elements_dev,   # CuArray{Simplex} for the CUDA kernel
    test_assembly_dev,   # CuArray (tad.data) reserved for future GPU scatter
    bfs,
    trial_ids,
    bsis_elements_dev,   # CuArray{Simplex}
    trial_assembly_dev,  # CuArray (bad.data)
    quaddata_gpu,        # NamedTuple of flattened quad-point CuArrays
    zlocals,             # CuMatrix (M×N scratch, used for eltype/size)
    num_tshapes,
    num_bshapes,
    store::DeviceStore;
    quadstrat,
)


    test_shapes = refspace(tfs)
    trial_shapes = refspace(bfs)

    test_id_dev, trial_id_dev = filter_and_copy_dev(tfs, bfs, test_ids, trial_ids)

    num_test = Int32(length(test_id_dev))
    num_trial = Int32(length(trial_id_dev))
    num_pairs = num_test * num_trial

    # Attempt at single kernel launch for all test trial pairs
    #
    # replaces the CPU loops:
    #   for p in active_test_el_ids
    #     for q in active_trial_el_ids
    #       qrule = quadrule(biop, ..., p, tcell, q, bcell, qd, qs)
    #       @assert qrule isa DoubleQuadRule
    #       igd = Integrand(biop, test_shapes, trial_shapes, tcell, bcell)
    #       for womp in qrule.outer_quad_points
    #         for wimp in qrule.inner_quad_points
    #           zlocals[i,j] += (jx*jy) * igd(tgeo, bgeo, tvals, bvals)[i,j]
    #
    # Instead of calling quadrule() on device, the kernel indexes the
    # pre-flattened quadrature point arrays by element id.  This is valid
    # because for DoubleQuadRule (far-field) interactions:
    #   quadrule(p, q, etc.) = DoubleQuadRule(tpoints[1,p], bpoints[1,q])
    # And we assume doublequadrule because the interactions processed by ACA are far-field

    ZT = eltype(zlocals)

    # zlocals instantiated straight on GPU

    # total amount of threads to be instantiated.
    total_work = Int(num_pairs)

    zlocals_all_dev = CUDA.zeros(ZT, num_tshapes, num_bshapes, num_pairs)

    kernel = @cuda launch = false momintegrals!(
        zlocals_all_dev,
        biop,
        test_shapes,
        trial_shapes,
        test_elements_dev,
        bsis_elements_dev,
        test_id_dev,
        trial_id_dev,
        quaddata_gpu.tqp_flat,
        quaddata_gpu.tqp_offsets,
        quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat,
        quaddata_gpu.bqp_offsets,
        quaddata_gpu.bqp_lengths,
        ZT,
        Int32(num_tshapes),
        Int32(num_bshapes),
        num_test,
        num_trial,
    )

    # Optimize thread block size
    config = launch_configuration(kernel.fun)
    threads = min(total_work, config.threads)
    blocks = cld(total_work, threads)

    kernel(
        zlocals_all_dev,
        biop,
        test_shapes,
        trial_shapes,
        test_elements_dev,
        bsis_elements_dev,
        active_test_ids_dev,
        active_trial_ids_dev,
        quaddata_gpu.tqp_flat,
        quaddata_gpu.tqp_offsets,
        quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat,
        quaddata_gpu.bqp_offsets,
        quaddata_gpu.bqp_lengths,
        Int32(num_tshapes),
        Int32(num_bshapes),
        num_test,
        num_trial;
        threads,
        blocks,
    )

    #synchronize so that all threads have finished for this iteration
    CUDA.synchronize()

    # scatter from local to global, this is executed on CPU
    # Copy per-pair local integrals back to host, then expand into global
    # matrix positions using the AssemblyData mapping and store() callback.

    # TODO: instantiate global (H-)Matrix and perform scatter-gather on GPU

    # Device to host memory copying

    # Build the per-block global → local-in-block dof maps that scatter_kernel! needs.
    # (test_id_dev / trial_id_dev hold *element* ids; these maps hold *dof* ids.)
    test_id_map, trial_id_map = create_id_maps(test_ids, trial_ids)

    # GPU scatter: each block accumulates contributions destined for one TILE_SIZE×TILE_SIZE
    # output tile in shared memory, then writes that tile into store.data with atomics.
    gpu_scatter!(
        store.data,
        zlocals_all_dev,
        test_assembly_dev,
        trial_assembly_dev,
        test_id_dev,
        trial_id_dev,
        test_id_map,
        trial_id_map,
        Int(num_tshapes),
        Int(num_bshapes),
    )
end

function momintegrals!(
    output::CuMatrix{T},
    op,
    tad::FlattenedAssemblyData{T},
    bad::FlattenedAssemblyData{T},
    test_shapes,
    trial_shapes,
    test_elements,
    bsis_elements,
    active_test_ids,
    active_trial_ids,
    tqp_flat,
    tqp_offsets,
    tqp_lengths,
    bqp_flat,
    bqp_offsets,
    bqp_lengths,
    num_tshapes::Int32,
    num_bshapes::Int32,
    num_test::Int32,
    num_trial::Int32,
) where {T}

    # Instantiate a "tile" which will later contain a copy of the slice of the matrix that is being handled by this SM
    tile = @cuStaticSharedMem T (BlockDim().x)
    idx = (BlockIdx().x - Int32(1)) * BlockDim().x + ThreadIdx().x

    p_local = div(idx - Int32(1), num_trial) + Int32(1)
    q_local = div(idx - Int32(1), num_test) + Int32(1)

    p = active_test_ids[p_local]
    q = active_trial_ids[q_local]

    """
     Instantiate the zlocal array for this thread (and thus test-trial pair)
     in close-by shared memory.
     Structure (important for indexing, however arbitrary)


     """
    # zlocal = CUDA.CuStaticSharedArray(T, num_tshapes * num_bshapes)

    tcell = test_elements[p]
    bcell = bsis_elements[q]

    """
    TODO: really think through how this integrand call affects warp divergence.
    """
    igd = Integrand(op, test_shapes, trial_shapes, tcell, bcell)

    o_off = tqp_offsets[p]
    o_len = tqp_lengths[p]
    i_off = bqp_offsets[q]
    i_len = bqp_lengths[q]

    acc = zero(zlocal_type)

    oi = Int32(0)
    while i < num_tshapes
        while j < num_bshapes
            t_off = tad.offsets[p, i]
            t_len = tad.lengths[p, i]
            b_off = bad.offsets[q, j]
            b_len = bad.lengths[q, j]

            # Calculate 1 zlocal entry, n_t * n_b times per thread
            # Why not have 1 entry per thread? zlocal cannot be instantiated in shared memory if it contains more than 32 threads? Well actually it can. but then the next warp needs to know where to read. how?
            while oi < o_len
                womp = tqp_flat[o_off+oi]
                tgeo = womp.point
                tvals = womp.value
                jx = womp.weight

                ii = Int32(0)
                while ii < i_len
                    wimp = bqp_flat[i_off+ii]
                    bgeo = wimp.point
                    bvals = wimp.value
                    jy = wimp.weight

                    z1 = igd(tgeo, bgeo, tvals, bvals)
                    acc += (jx * jy) * z1[i, j]

                    ii += Int32(1)
                end

                oi += Int32(1)
            end

            """if acc != zero(T) I believe this check leads to unnecessary divergence"""

            ti = Int32(0)
            while ti < t_len
                @inbounds (m_global, a_coeff) = tad.flat[t_off+ti]
                @inbounds m_local = test_id_map[m_global]

                if m_local != Int32(0)
                    m_tile = m_local - tile_m_start + Int32(1)
                    if m_tile >= Int32(1) && m_tile <= TILE_SIZE
                        bi = Int32(0)
                        while bi < b_len
                            @inbounds (n_global, b_coeff) = bad.flat[b_off+bi]
                            @inbounds n_local = trial_id_map[n_global]

                            if n_local != Int32(0)
                                n_tile = n_local - tile_n_start + Int32(1)
                                if n_tile >= Int32(1) && n_tile <= TILE_SIZE
                                    contribution = a_coeff * acc * b_coeff
                                    tile[m_tile, n_tile] += contribution
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
end
