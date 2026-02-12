# Define overload for BEAST's structs that need to be run on GPU
# function Adapt.adapt_structure(to, test_elements::Vector{CompScienceMeshes.Simplex{3,2,1,3,Float64}})
#     return [Adapt.adapt(to, e) for e in test_elements]
# end

# function Adapt.adapt_structure(to, cell::CompScienceMeshes.Simplex{3, 2, 1, 3, Float64})
#     vertices = Adapt.adapt(to, cell.vertices)
#     tangents = Adapt.adapt(to, cell.tangents)
#     normals  = Adapt.adapt(to, cell.normals)
#     volume  = Float32(cell.volume) # Float32 is the most efficient GPU numeric datatype
#     return CompScienceMeshes.Simplex{3, 2, 1, 3, Float32}(vertices, tangents, normals, volume)
# end

# function Adapt.adapt_structure(to, vertices::SVector{3, SVector{3, Float64}})
#     new_vertices = SVector{3, SVector{3, Float32}}(
#         Adapt.adapt(to, vertices[1]),
#         Adapt.adapt(to, vertices[2]),
#         Adapt.adapt(to, vertices[3])
#     )
#     return new_vertices
# end

# GPU-adapted LagrangeBasis definition
# Immutable version for GPU
struct LagrangeBasisGPU
    geo_vertices::CuArray{SVector{3,Float64}}
    geo_faces::CuArray{SVector{3,Int32}}
    shapes_cellid::CuArray{Int32}
    shapes_refid::CuArray{Int32}
    shapes_coeff::CuArray{Float64}
    pos::CuArray{SVector{3,Float64}}
    offsets::CuArray{Int32}
end

function LagrangeBasisGPU(basis::LagrangeBasis)
    shapes = reduce(vcat, basis.fns)

    vertices = [SVector{3, Float64}(v) for v in basis.geo.vertices]
    faces = [SVector{3, Int32}(f) for f in basis.geo.faces]
    cell_ids = Int32[s.cellid for s in shapes]
    ref_ids = Int32[s.refid for s in shapes]
    coeffs = Float64[s.coeff for s in shapes]
    positions = [SVector{3, Float64}(p) for p in basis.pos]
    offsets = Int32.(cumsum([0; length.(basis.fns)]))

    cu_vertices = CUDA.cu(vertices)
    cu_faces = CUDA.cu(faces)
    cu_cell_ids = CUDA.cu(cell_ids)
    cu_ref_ids = CUDA.cu(ref_ids)
    cu_coeffs = CUDA.cu(coeffs)
    cu_positions = CUDA.cu(positions)
    cu_offsets = CUDA.cu(offsets)

    return LagrangeBasisGPU(
        cu_vertices,
        cu_faces,
        cu_cell_ids,
        cu_ref_ids,
        cu_coeffs,
        cu_positions,
        cu_offsets
    )
end

to_gpu(basis::LagrangeBasis) = LagrangeBasisGPU(basis)

function Adapt.adapt_structure(to, basis::BEAST.LagrangeBasis{D,C,M,T,NF,P} where {D,C,M,T,NF,P})
    geo = Adapt.adapt(to, basis.geo)
    fns = Adapt.adapt(to, basis.fns)
    pos = Adapt.adapt(to, basis.pos)
    return BEAST.LagrangeBasis{D,C,M,T,NF,P}(geo, fns, pos) where {D,C,M,T,NF,P}
end

function Adapt.adapt_structure(to, fns::Vector{Vector{BEAST.Shape{T}}}) where T
    return [Adapt.adapt(to, fn) for fn in fns]
end

function Adapt.adapt_structure(to, geo::CompScienceMeshes.Mesh{U,D1,T}) where {U, D1, T}
    vertices = Adapt.adapt(to, geo.vertices)
    faces = Adapt.adapt(to, geo.faces)
    return CompScienceMeshes.Mesh(vertices, faces)
end

function assembleblock_primer_gpu(biop, tfs, bfs;
        quadstrat=defaultquadstrat(biop, tfs, bfs), gpu=false)
    print("type of tfs: ", typeof(tfs), "\n")
    print("type of bfs: ", typeof(bfs), "\n")
    test_elements, tad, tcells = assemblydata(tfs; onlyactives=false)
    bsis_elements, bad, bcells = assemblydata(bfs; onlyactives=false)

    tgeo = geometry(tfs)
    bgeo = geometry(bfs)

    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))

    tshapes = refspace(tfs); num_tshapes = numfunctions(tshapes, tdom)
    bshapes = refspace(bfs); num_bshapes = numfunctions(bshapes, bdom)
    qd = quaddata(biop, tshapes, bshapes, test_elements, bsis_elements, quadstrat)

    zlocals_dev = CUDA.zeros(scalartype(biop, tfs, bfs), num_tshapes, num_bshapes)
    tad_dev = CUDA.cu(tad.data)
    bad_dev = CUDA.cu(bad.data)
    return test_elements, tad_dev, bsis_elements, bad_dev, qd, zlocals_dev, tcells, bcells
end 

function momintegrals_kernel!(out, op, outer_quad_points, inner_quad_points)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    num_t = size(out, 1)
    num_b = size(out, 2)
    total = num_t * num_b

    if idx <= total
        i = ((idx - 1) % num_t) + 1
        j = (div((idx - 1), num_t)) + 1

        acc = zero(eltype(out))
        for womp in outer_quad_points
            tgeo = womp.point
            tvals = womp.value
            jx = womp.weight
            for wimp in inner_quad_points
                bgeo = wimp.point
                bvals = wimp.value
                jy = wimp.weight
                kervals = kernelvals(op, tgeo, bgeo)
                acc += jx * jy * integrand(op, kervals, tvals[i], tgeo, bvals[j], bgeo)
            end
        end
        out[i, j] += acc
    end

    return nothing
end

function momintegrals!(biop, tshs, bshs, tcell, bcell, z::CuArray, strat::DoubleQuadRule)
    outer_quad_points = CUDA.cu(strat.outer_quad_points)
    inner_quad_points = CUDA.cu(strat.inner_quad_points)

    threads = 256
    total = size(z, 1) * size(z, 2)
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks momintegrals_kernel!(z, biop, outer_quad_points, inner_quad_points)
    return z
end

function assembleblock_body_gpu!(
        test_assembly_data, trial_assembly_data,
        active_test_el_ids, active_trial_el_ids,
        test_id_in_blk, trial_id_in_blk,
        zlocal, store)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_test = length(active_test_el_ids)
    n_trial = length(active_trial_el_ids)
    total_pairs = n_test * n_trial

    if idx <= total_pairs
        p_idx = (div((idx - 1), n_trial)) + 1
        q_idx = ((idx - 1) % n_trial) + 1

        p = Int(active_test_el_ids[p_idx])
        q = Int(active_trial_el_ids[q_idx])

        for j in 1:size(zlocal, 2)
            for i in 1:size(zlocal, 1)
                zval = zlocal[i, j]
                zval == zero(zval) && continue
                for k in 1:size(trial_assembly_data, 1)
                    n, b = trial_assembly_data[k, j, q]
                    n == 0 && break
                    npos = Int(trial_id_in_blk[n])
                    npos == 0 && continue
                    for l in 1:size(test_assembly_data, 1)
                        m, a = test_assembly_data[l, i, p]
                        m == 0 && break
                        mpos = Int(test_id_in_blk[m])
                        mpos == 0 && continue
                        CUDA.@atomic store[mpos, npos] += a * zval * b
                    end
                end
            end
        end
    end

    return nothing
end
