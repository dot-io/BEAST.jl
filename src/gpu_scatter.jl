using CUDA

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
Flattened AssemblyData struct. Contains:
- flat: device memory mapping an index to data (need to be more specific)
- offsets : A 2D matrix on devmem containing the offsets for each index of assembly data

N.B.: Assembly data type is parametrized because it may be complex or require different precision. TODO: i should check if this is actually the case.
In any case Julia's polymorphism should mean no performance overhead except at compile-time.
"""

struct FlattenedAssemblyData{T}
    flat::CuVector{Tuple{Int32,T}}
    offsets::CuMatrix{Int32}
    lengths::CuMatrix{Int32}
    num_elements::Int32
    num_local_shapes::Int32
end
"""
FlattenedAssemblyData constructor
Iterating over num_elements, num_local_shapes allows for heterogeneous entry lengths which lead to uneven offsets.
"""
function FlattenedAssemblyData(
    assembly_data,
    num_elements::Int,
    num_local_shapes::Int,
    ::Type{T},
) where {T}
    entries = Tuple{Int32,T}[]
    offsets = zeros(Int32, num_elements, num_local_shapes)
    lengths = zeros(Int32, num_elements, num_local_shapes)

    for p = 1:num_elements
        for i = 1:num_local_shapes
            offsets[p, i] = length(entries) + 1
            count = Int32(0)
            for (m, a) in assembly_data[p, i]
                push!(entries, (Int32(m), T(a)))
                count += 1
            end
            lengths[p, i] = count
        end
    end

    return FlattenedAssemblyData{T}(
        CUDA.cu(entries),
        CUDA.cu(offsets),
        CUDA.cu(lengths),
        Int32(num_elements),
        Int32(num_local_shapes),
    )
end

# Length overload to get number of elements easily
Base.length(fad::FlattenedAssemblyData) = fad.num_elements

#Idexing overload as a convenience function
function Base.getindex(
    fad::FlattenedAssemblyData{T},
    p::Int32,
    i::Int32,
) where {T}
    off = fad.offsets[p, i]
    len = fad.lengths[p, i]
    return (@view fad.flat[off:(off+len-1)])
end

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

"""
    create_id_maps(test_ids, trial_ids)

Create global → local-in-block index mappings for test and trial functions.

Returns `(test_id_map, trial_id_map)` as CuVectors where:
- `test_id_map[m] = i` if global basis function `m` is the `i`-th function in the block
- `test_id_map[m] = 0` if `m` is not in this block
"""
function create_id_maps(
    test_ids::AbstractVector{Int},
    trial_ids::AbstractVector{Int},
)
    max_test = isempty(test_ids) ? 0 : maximum(test_ids)
    max_trial = isempty(trial_ids) ? 0 : maximum(trial_ids)

    test_id_map = zeros(Int32, max_test)
    trial_id_map = zeros(Int32, max_trial)

    for (i, m) in enumerate(test_ids)
        test_id_map[m] = i
    end
    for (i, n) in enumerate(trial_ids)
        trial_id_map[n] = i
    end

    return CUDA.cu(test_id_map), CUDA.cu(trial_id_map)
end

export FlattenedAssemblyData, gpu_scatter!, create_id_maps