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
"""

     assembleblock_primer_gpu(biop::IntegralOperator, tfs::Space, bfs::Space) -> ctx::NamedTuple

     Computes block assembly data, element geometry and quadrature data on the
     CPU and transfers this to the device.
     """
function assembleblock_primer_gpu(biop, tfs, bfs; loop_order=:gather)
    qs = BEAST.defaultquadstrat(biop, tfs, bfs)
    test_elements, tad, trial_elements, bad, qd, _ =
        BEAST.assembleblock_primer(biop, tfs, bfs; quadstrat=qs) #Still computes CPU side elements and assemblydata, THEN copies to GPU

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

    test_elements_dev = CUDA.cu(test_elements)
    trial_elements_dev = CUDA.cu(trial_elements)
    if loop_order == :gather
        tad_gpu = InvAssemblyData(tad, length(test_elements), num_tshapes, num_tfs, ZT)
        bad_gpu = InvAssemblyData(bad, length(trial_elements), num_bshapes, num_bfs, ZT)
    elseif loop_order == :scatter
        tad_gpu = FlattenedAssemblyData(tad, length(test_elements), num_tshapes, ZT)
        bad_gpu = FlattenedAssemblyData(bad, length(trial_elements), num_bshapes, ZT)
    end

    # Flatten quadrature data into device arrays. The GPU module is expected
    # to provide `flatten_quaddata_gpu`; if it doesn't yet, this line is the
    # signal that the helper still has to land alongside `gpu_2.jl`.
    quaddata_gpu = flatten_quaddata_gpu(qd, length(test_elements), length(trial_elements))

    zlocals = CUDA.zeros(ZT, num_tshapes, num_bshapes)

    return (;
        qs, ZT,
        test_elements, trial_elements,
        tad, bad, qd,
        test_elements_dev, trial_elements_dev,
        tad_gpu, bad_gpu, quaddata_gpu,
        num_tshapes, num_bshapes, zlocals,
        test_shapes, trial_shapes,
    )
end
