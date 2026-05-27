#helper typedefs for readability
const RTShapeValue{T} = NamedTuple{(:value, :divergence),Tuple{SVector{3,T},T}}
const QuadPoint{T} = MeshPointNM{T,Simplex{3,2,1,3,T},2,3}
const WeightedQuadPoint{N,T} = NamedTuple{(:weight, :point, :value),Tuple{T,QuadPoint{T},SVector{N,RTShapeValue{T}}}}

# helper constants for scatter kernel
const TILE_SIZE = Int32(16)
const TILE_SIZE_SQ = TILE_SIZE^2

abstract type DeviceStore end

struct CuMatrixStore{T} <: DeviceStore
    data::CuMatrix{T}
end

function (s::CuMatrixStore{T})(v, m::CuVector{Int32}, n::CuVector{Int32}) where T
    s.data[m, n] .+= v
end

#convenience overload if store called with scalar indices
function (s::CuMatrixStore{T})(v, m::Int, n::Int) where T
    s(v, CUDA.cu(Int32[m]), CUDA.cu(Int32[n]))
end
"""
Flattened AssemblyData struct. Contains:
- flat: device memory mapping an index to data (need to be more specific)
- offsets : A 2D matrix on devmem containing the offsets for each index of assembly data

N.B.: Assembly data type is parametrized because it may be complex or require different precision.
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
InvAssemblyData struct. Contains:
- flat:  device memory vector containing tuples (cellid, refid, coeff)
"""
struct InvAssemblyData{T}        # one entry per (cell, refid) supporting dof m
    flat::CuVector{Tuple{Int32,Int32,T}}    # (cellid, refid, coeff)
    offsets::CuVector{Int32}                  # one per dof
    lengths::CuVector{Int32}
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
        for i = 1:num_local_shapes # iterating over (element, local shape function) described as p, i here but is valid for q, j also in the case of trial assembly data
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


# TODO fix for correctness: I of course cannot index simply on the Degree of Freedom
function InvAssemblyData(
    assembly_data,
    num_elements::Int,
    num_local_shapes::Int,
    num_dofs::Int,
    ::Type{T},
) where {T}
    inv_map = [Tuple{Int32,Int32,T}[] for _ in 1:num_dofs]
    offsets = zeros(Int32, num_dofs)
    lengths = zeros(Int32, num_dofs)
    entries = Tuple{Int32,Int32,T}[]

    for p = 1:num_elements
        for i = 1:num_local_shapes # iterating over degrees of freedom (test or trial depending on which assembly data is given as argument)
            dofs_coeffs = assembly_data[p, i]

            for (m, coeff) in dofs_coeffs
                push!(inv_map[m], (Int32(p), Int32(i), T(coeff))) # push the tuple (cellid, refid, coeff) to the list of entries for dof m
            end
        end
    end

    for m = 1:num_dofs
        count = Int32(0)
        offsets[m] = length(entries) + 1
        for entry in inv_map[m]
            push!(entries, entry) # flatten the list of entries for dof m into the flat array
            count += 1
        end
        lengths[m] = count
    end

    return InvAssemblyData{T}(
        CUDA.cu(entries),
        CUDA.cu(offsets),
        CUDA.cu(lengths),
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
    create_id_maps(test_ids, trial_ids, num_tfs, num_bfs)

Create global → local-in-block index mappings for test and trial functions.
"""
function create_id_maps(
    test_ids::AbstractVector{Int},
    trial_ids::AbstractVector{Int},
    num_tfs::Int,
    num_bfs::Int,
)
    test_id_map = zeros(Int32, num_tfs)
    trial_id_map = zeros(Int32, num_bfs)

    for (i, m) in enumerate(test_ids)
        test_id_map[m] = i
    end
    for (i, n) in enumerate(trial_ids)
        trial_id_map[n] = i
    end

    return CUDA.cu(test_id_map), CUDA.cu(trial_id_map)
end
u

#legacy TODO remove refs
function create_id_maps(
    test_ids::AbstractVector{Int},
    trial_ids::AbstractVector{Int},
)
    max_test = isempty(test_ids) ? 0 : maximum(test_ids)
    max_trial = isempty(trial_ids) ? 0 : maximum(trial_ids)
    return create_id_maps(test_ids, trial_ids, max_test, max_trial)
end


function flatten_quaddata_gpu(quadrature_data, n_test, n_trial)
    # Flatten quadrature data for GPU.
    # For DoubleNumWiltonSauterQStrat, tpoints/bpoints are 2×N matrices
    # where row 1 = far-field quad points, row 2 = near-field quad points.
    # The GPU kernel currently only handles the far-field (DoubleQuadRule) path,
    # so we extract only row 1 and build per-element offsets.
    T = typeof(quadrature_data.tpoints[1][1].weight)
    N = length(quadrature_data.tpoints[1][1].value) # number of shape functions per element

    # Extract far-field quad points (row 1) for each element
    test_far = [quadrature_data.tpoints[1, p] for p in 1:n_test]
    trial_far = [quadrature_data.bpoints[1, q] for q in 1:n_trial]

    # Flatten into contiguous arrays with per-element offsets
    tqp_flat = reduce(vcat, test_far)
    bqp_flat = reduce(vcat, trial_far)
    tqp_lengths = Int32.(length.(test_far))
    bqp_lengths = Int32.(length.(trial_far))
    tqp_offsets = Int32.(cumsum(vcat(Int32(1), tqp_lengths[1:end-1])))
    bqp_offsets = Int32.(cumsum(vcat(Int32(1), bqp_lengths[1:end-1])))
    return (
        tqp_flat=CUDA.cu(tqp_flat),
        tqp_offsets=CUDA.cu(tqp_offsets),
        tqp_lengths=CUDA.cu(tqp_lengths),
        bqp_flat=CUDA.cu(bqp_flat),
        bqp_offsets=CUDA.cu(bqp_offsets),
        bqp_lengths=CUDA.cu(bqp_lengths),
    )
end

@inline function compute_pair_entry(
    ::Type{T},
    operator, test_shapes, trial_shapes,
    test_element, trial_element,
    i::Int32, j::Int32,
    test_qp, t_off::Int32, t_len::Int32,
    trial_qp, b_off::Int32, b_len::Int32,
) where {T}
    igd = Integrand(operator, test_shapes, trial_shapes, test_element, trial_element)
    acc = zero(T)

    oi = Int32(0)
    while oi < t_len
        @inbounds womp = test_qp[t_off+oi]
        tgeo = womp.point
        tvals = womp.value
        jx = womp.weight

        ii = Int32(0)
        while ii < b_len
            @inbounds wimp = trial_qp[b_off+ii]
            bgeo = wimp.point
            bvals = wimp.value
            jy = wimp.weight

            z1 = igd(tgeo, bgeo, tvals, bvals)   # M × N matrix of integrand values
            acc += jx * jy * z1[i, j]            # extract only the (i, j) entry
            # here some expensive recomputation happens
            ii += Int32(1)
        end
        oi += Int32(1)
    end

    return acc
end


"""
    accumulate_zlocal_ntuple(::Type{T}, op, test_shapes, trial_shapes, tcell, bcell,
        tqp_flat, t_off, t_len, bqp_flat, b_off, b_len,
        ::Val{NS}, ::Val{MS}) -> NTuple{NS*MS, T}
"""
@inline function accumulate_zlocal_ntuple(::Type{T}, op,
    test_shapes, trial_shapes, tcell, bcell,
    tqp_flat, t_off::Int32, t_len::Int32,
    bqp_flat, b_off::Int32, b_len::Int32,
    ::Val{NS}, ::Val{MS},
) where {T,NS,MS}
    igd = Integrand(op, test_shapes, trial_shapes, tcell, bcell)
    z = ntuple(_ -> zero(T), Val(NS * MS))

    oi = Int32(0)
    @inbounds while oi < t_len
        womp = tqp_flat[t_off+oi]
        tgeo = womp.point
        tvals = womp.value
        jx = womp.weight

        ii = Int32(0)
        while ii < b_len
            wimp = bqp_flat[b_off+ii]
            z1 = igd(tgeo, wimp.point, tvals, wimp.value)
            jxjy = jx * wimp.weight

            for j_acc in Int32(1):Int32(MS)
                for i_acc in Int32(1):Int32(NS)
                    flat_idx = (j_acc - Int32(1)) * Int32(NS) + i_acc
                    z = Base.setindex(z, z[flat_idx] + jxjy * z1[i_acc, j_acc], flat_idx)
                end
            end

            ii += Int32(1)
        end
        oi += Int32(1)
    end

    return z
end


struct FlattenedAssemblyDataWithK{T}
    flat::CuVector{Tuple{Int32,Int32,T}}   # (dof_global, contributor_index_k, coefficient)
    offsets::CuMatrix{Int32}               # indexed by (element, local_shape)
    lengths::CuMatrix{Int32}
    num_elements::Int32
    num_local_shapes::Int32
end

"""
    FlattenedAssemblyDataWithK constructor
"""
function FlattenedAssemblyDataWithK(
    assembly_data,
    num_elements::Int,
    num_local_shapes::Int,
    num_dofs::Int,
    ::Type{T},
) where {T}
    # First, build the inverse map to determine contributor indices
    inv_map = [Tuple{Int32,Int32,T}[] for _ in 1:num_dofs]
    for p = 1:num_elements
        for i = 1:num_local_shapes
            for (m, coeff) in assembly_data[p, i]
                push!(inv_map[m], (Int32(p), Int32(i), T(coeff)))
            end
        end
    end

    # Build forward map with k: for each (p, i) → (m, a), determine k
    # by looking up the position of (p, i, a) in inv_map[m]
    fwd_entries = Tuple{Int32,Int32,T}[]
    offsets = zeros(Int32, num_elements, num_local_shapes)
    lengths = zeros(Int32, num_elements, num_local_shapes)

    for p = 1:num_elements
        for i = 1:num_local_shapes
            offsets[p, i] = length(fwd_entries) + 1
            count = Int32(0)
            for (m, a) in assembly_data[p, i]
                # Find k: the position of (p, i, a) in inv_map[m]
                k = Int32(0)
                for (idx, entry) in enumerate(inv_map[m])
                    if entry[1] == Int32(p) && entry[2] == Int32(i)
                        k = Int32(idx)
                        break
                    end
                end
                @assert k > 0 "Could not find (p=$p, i=$i) in inv_map[m=$m]"
                push!(fwd_entries, (Int32(m), k, T(a)))
                count += 1
            end
            lengths[p, i] = count
        end
    end

    return FlattenedAssemblyDataWithK{T}(
        CUDA.cu(fwd_entries),
        CUDA.cu(offsets),
        CUDA.cu(lengths),
        Int32(num_elements),
        Int32(num_local_shapes),
    )
end

# Length overload to get number of elements easily
Base.length(fad::FlattenedAssemblyDataWithK) = fad.num_elements

# Indexing overload as a convenience function
function Base.getindex(
    fad::FlattenedAssemblyDataWithK{T},
    p::Int32,
    i::Int32,
) where {T}
    off = fad.offsets[p, i]
    len = fad.lengths[p, i]
    return (@view fad.flat[off:(off+len-1)])
end


"""
    HybridAssemblyData{T}

Combined forward + inverse assembly data for the hybrid (biphasic) GPU kernels.
"""
struct HybridAssemblyData{T}
    # Forward maps with contributor index k
    fwd_tad::FlattenedAssemblyDataWithK{T}
    fwd_bad::FlattenedAssemblyDataWithK{T}

    # Inverse maps (same as InvAssemblyData)
    inv_tad::InvAssemblyData{T}
    inv_bad::InvAssemblyData{T}

    # Zero-padded coefficient arrays for branchless Phase 2
    coeff_test_padded::CuMatrix{T}    # [m, k] — coefficient a for the k-th contributor to test dof m
    coeff_trial_padded::CuMatrix{T}   # [n, l] — coefficient b for the l-th contributor to trial dof n

    # Maximum support sizes
    K_max_test::Int32
    K_max_trial::Int32
end

"""
    HybridAssemblyData constructor

"""
function HybridAssemblyData(
    tad_cpu,
    bad_cpu,
    num_test_elements::Int,
    num_trial_elements::Int,
    num_tshapes::Int,
    num_bshapes::Int,
    num_tfs::Int,
    num_bfs::Int,
    ::Type{T},
) where {T}
    # Build forward maps with contributor index k
    fwd_tad = FlattenedAssemblyDataWithK(tad_cpu, num_test_elements, num_tshapes, num_tfs, T)
    fwd_bad = FlattenedAssemblyDataWithK(bad_cpu, num_trial_elements, num_bshapes, num_bfs, T)

    # Build inverse maps
    inv_tad = InvAssemblyData(tad_cpu, num_test_elements, num_tshapes, num_tfs, T)
    inv_bad = InvAssemblyData(bad_cpu, num_trial_elements, num_bshapes, num_bfs, T)

    # Compute K_max values
    K_max_test = isempty(inv_tad.lengths) ? Int32(1) : maximum(inv_tad.lengths)
    K_max_trial = isempty(inv_bad.lengths) ? Int32(1) : maximum(inv_bad.lengths)

    # Build padded coefficient arrays from CPU-side inverse maps
    # (avoid scalar indexing on CuArrays)
    # coeff_test_padded[m, k] = coefficient of the k-th contributor to test dof m
    # coeff_trial_padded[n, l] = coefficient of the l-th contributor to trial dof n
    coeff_test_padded = zeros(T, num_tfs, Int(K_max_test))
    coeff_trial_padded = zeros(T, num_bfs, Int(K_max_trial))

    # Rebuild CPU-side inverse maps for coefficient extraction
    inv_map_test = [Tuple{Int32,Int32,T}[] for _ in 1:num_tfs]
    inv_map_trial = [Tuple{Int32,Int32,T}[] for _ in 1:num_bfs]
    for p in 1:num_test_elements, i in 1:num_tshapes
        for (m, coeff) in tad_cpu[p, i]
            push!(inv_map_test[m], (Int32(p), Int32(i), T(coeff)))
        end
    end
    for q in 1:num_trial_elements, j in 1:num_bshapes
        for (n, coeff) in bad_cpu[q, j]
            push!(inv_map_trial[n], (Int32(q), Int32(j), T(coeff)))
        end
    end
    for m in 1:num_tfs
        for k in 1:length(inv_map_test[m])
            coeff_test_padded[m, k] = inv_map_test[m][k][3]
        end
    end
    for n in 1:num_bfs
        for l in 1:length(inv_map_trial[n])
            coeff_trial_padded[n, l] = inv_map_trial[n][l][3]
        end
    end

    return HybridAssemblyData{T}(
        fwd_tad, fwd_bad,
        inv_tad, inv_bad,
        CUDA.cu(coeff_test_padded), CUDA.cu(coeff_trial_padded),
        K_max_test, K_max_trial,
    )
end


# extract active element ids and upload to device
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


# per pair integrand eval (used by scatter/gather)
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

"""
    build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)
    -> (pair_flat, pair_off)

store all pairs that have contributions to a 'tile'.
"""
function build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)
    n_tiles_m = cld(length(test_ids), M_tile)
    n_tiles_n = cld(length(trial_ids), N_tile)
    pair_off = Int32[1]
    pair_flat = Tuple{Int32,Int32}[]

    # Linear tile index is tm + (tn - 1) * n_tiles_m
    for tn in 1:n_tiles_n, tm in 1:n_tiles_m # inter-tile loop
        # collect unique test elements supporting any dof in this tile's rows
        pset = Set{Int32}() # where p stands for a test element index
        for k in 1:M_tile # intra-tile loop
            m_local = (tm - 1) * M_tile + k # local dof index within the tile
            m_local > length(test_ids) && break
            for sh in tfs.fns[test_ids[m_local]]
                push!(pset, Int32(sh.cellid))
            end
        end
        # idem for trial
        qset = Set{Int32}()
        for k in 1:N_tile
            n_local = (tn - 1) * N_tile + k
            n_local > length(trial_ids) && break
            for sh in bfs.fns[trial_ids[n_local]]
                push!(qset, Int32(sh.cellid))
            end
        end
        for p in pset
            for q in qset
                push!(pair_flat, (p, q))
            end
        end
        push!(pair_off, Int32(length(pair_flat) + 1))   # offset till next tile
    end
    return CUDA.cu(pair_flat), CUDA.cu(pair_off)
end
