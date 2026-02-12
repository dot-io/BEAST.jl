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

function blockassembler_gpu(biop::IntegralOperator, tfs::Space, bfs::Space; quadstrat=defaultquadstrat(biop, tfs, bfs), gpu=false)

    tgeo = geometry(tfs)
    bgeo = geometry(bfs)
    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))
    test_shapes = refspace(tfs)
    trial_shapes = refspace(bfs)
    num_tshapes = numfunctions(test_shapes, tdom)
    num_bshapes = numfunctions(trial_shapes, bdom)

    qs = if CompScienceMeshes.refines(tgeo, bgeo)
        TestRefinesTrialQStrat(quadstrat)
    elseif CompScienceMeshes.refines(bgeo, tgeo)
        TrialRefinesTestQStrat(quadstrat)
    else
        quadstrat
    end

    if CUDA.functional() && gpu
        @info "CUDA is available, using GPU assembly"
        # convert lagrangebasis to device memory
        tfs_dev = to_gpu(tfs)
        bfs_dev = to_gpu(bfs)

        gpu_momintegrals_ok = Ref(true)
        gpu_momintegrals_checked = Ref(false)

        test_elements, test_assembly_dev, trial_elements, trial_assembly_dev, quadrature_data, zlocals_dev, test_cell_ptrs, trial_cell_ptrs = assembleblock_primer_gpu(biop, tfs, bfs; quadstrat=qs, gpu=gpu)
        # convert the test and trial index struct to a suitable CUDA bitstype naively
        return (test_ids, trial_ids, store) -> begin
            active_test_el_ids = Int32[]
            active_trial_el_ids = Int32[]
            for m in test_ids, sh in tfs.fns[m]
                push!(active_test_el_ids, Int32(sh.cellid))
            end
            for m in trial_ids, sh in bfs.fns[m]
                push!(active_trial_el_ids, Int32(sh.cellid))
            end
            active_test_el_ids = unique!(sort!(active_test_el_ids))
            active_trial_el_ids = unique!(sort!(active_trial_el_ids))

            test_id_in_blk = fill(Int32(0), numfunctions(tfs))
            trial_id_in_blk = fill(Int32(0), numfunctions(bfs))
            for (i, m) in enumerate(test_ids)
                test_id_in_blk[m] = Int32(i)
            end
            for (i, m) in enumerate(trial_ids)
                trial_id_in_blk[m] = Int32(i)
            end

            test_id_in_blk_dev = CUDA.cu(test_id_in_blk)
            trial_id_in_blk_dev = CUDA.cu(trial_id_in_blk)

            total_pairs = length(active_test_el_ids) * length(active_trial_el_ids)
            store_dev = CUDA.zeros(eltype(zlocals_dev), length(test_ids), length(trial_ids))
            kernel_time = 0.0
            total_time = @elapsed begin
                if total_pairs > 0
                    zlocal_host = zeros(scalartype(biop, tfs, bfs), num_tshapes, num_bshapes)
                    zlocal_dev = zlocals_dev
                    pair_p = Int32[]
                    pair_q = Int32[]
                    outer_offsets = Int32[]
                    outer_lengths = Int32[]
                    inner_offsets = Int32[]
                    inner_lengths = Int32[]
                    outer_points = nothing
                    inner_points = nothing
                    outer_offset = 0
                    inner_offset = 0
                    all_compatible = true

                    for p in active_test_el_ids
                        tcell = test_elements[p]
                        tptr = test_cell_ptrs[p]
                        for q in active_trial_el_ids
                            bcell = trial_elements[q]
                            bptr = trial_cell_ptrs[q]
                            qrule = quadrule(biop, test_shapes, trial_shapes, p, tcell, q, bcell, quadrature_data, qs)

                            if !(can_use_gpu_momintegrals(qrule) && gpu_momintegrals_ok[])
                                all_compatible = false
                                break
                            end

                            if !gpu_momintegrals_checked[]
                                try
                                    fill!(zlocal_dev, 0)
                                    momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal_dev, qrule)
                                    fill!(zlocal_host, 0)
                                    momintegrals!(zlocal_host, biop, tfs, tptr, tcell, bfs, bptr, bcell, qrule)
                                    zlocal_gpu_host = Array(zlocal_dev)
                                    diff = maximum(abs.(zlocal_gpu_host - zlocal_host))
                                    if diff > 1e-8 * max(1.0, maximum(abs.(zlocal_host)))
                                        gpu_momintegrals_ok[] = false
                                        all_compatible = false
                                        @warn "GPU momintegrals mismatch; falling back to CPU" diff=diff
                                        break
                                    else
                                        gpu_momintegrals_checked[] = true
                                    end
                                catch err
                                    gpu_momintegrals_ok[] = false
                                    all_compatible = false
                                    @warn "GPU momintegrals disabled; falling back to CPU" err
                                    break
                                end
                            end

                            if outer_points === nothing
                                outer_points = Vector{eltype(qrule.outer_quad_points)}()
                                inner_points = Vector{eltype(qrule.inner_quad_points)}()
                            elseif eltype(qrule.outer_quad_points) != eltype(outer_points) ||
                                   eltype(qrule.inner_quad_points) != eltype(inner_points)
                                all_compatible = false
                                break
                            end

                            push!(pair_p, Int32(p))
                            push!(pair_q, Int32(q))
                            push!(outer_offsets, Int32(outer_offset + 1))
                            olen = length(qrule.outer_quad_points)
                            push!(outer_lengths, Int32(olen))
                            append!(outer_points, qrule.outer_quad_points)
                            outer_offset += olen
                            push!(inner_offsets, Int32(inner_offset + 1))
                            ilen = length(qrule.inner_quad_points)
                            push!(inner_lengths, Int32(ilen))
                            append!(inner_points, qrule.inner_quad_points)
                            inner_offset += ilen
                        end
                        all_compatible || break
                    end

                    if all_compatible && !isempty(pair_p)
                        pair_p_dev = CUDA.cu(pair_p)
                        pair_q_dev = CUDA.cu(pair_q)
                        outer_offsets_dev = CUDA.cu(outer_offsets)
                        outer_lengths_dev = CUDA.cu(outer_lengths)
                        inner_offsets_dev = CUDA.cu(inner_offsets)
                        inner_lengths_dev = CUDA.cu(inner_lengths)
                        outer_points_dev = CUDA.cu(outer_points)
                        inner_points_dev = CUDA.cu(inner_points)

                        total_work = length(pair_p) * num_tshapes * num_bshapes
                        threads = 256
                        blocks = cld(total_work, threads)
                        kernel_time += @elapsed begin
                            @cuda threads=threads blocks=blocks momintegrals_assemble_pairs_kernel!(
                                store_dev, biop,
                                pair_p_dev, pair_q_dev,
                                outer_points_dev, outer_offsets_dev, outer_lengths_dev,
                                inner_points_dev, inner_offsets_dev, inner_lengths_dev,
                                test_assembly_dev, trial_assembly_dev,
                                test_id_in_blk_dev, trial_id_in_blk_dev,
                                Int32(num_tshapes), Int32(num_bshapes))
                            CUDA.synchronize()
                        end
                    else
                        for p in active_test_el_ids
                            tcell = test_elements[p]
                            tptr = test_cell_ptrs[p]
                            for q in active_trial_el_ids
                                bcell = trial_elements[q]
                                bptr = trial_cell_ptrs[q]
                                qrule = quadrule(biop, test_shapes, trial_shapes, p, tcell, q, bcell, quadrature_data, qs)

                            if gpu_momintegrals_ok[]
                                    try
                                        fill!(zlocal_dev, 0)
                                        momintegrals!(biop, test_shapes, trial_shapes, tcell, bcell, zlocal_dev, qrule)
                                        if !gpu_momintegrals_checked[]
                                            fill!(zlocal_host, 0)
                                            momintegrals!(zlocal_host, biop, tfs, tptr, tcell, bfs, bptr, bcell, qrule)
                                            zlocal_gpu_host = Array(zlocal_dev)
                                            diff = maximum(abs.(zlocal_gpu_host - zlocal_host))
                                            if diff > 1e-8 * max(1.0, maximum(abs.(zlocal_host)))
                                                gpu_momintegrals_ok[] = false
                                                @warn "GPU momintegrals mismatch; falling back to CPU" diff=diff
                                                copyto!(zlocal_dev, zlocal_host)
                                            else
                                                gpu_momintegrals_checked[] = true
                                            end
                                        end
                                    catch err
                                        gpu_momintegrals_ok[] = false
                                        @warn "GPU momintegrals disabled; falling back to CPU" err
                                        fill!(zlocal_host, 0)
                                        momintegrals!(zlocal_host, biop, tfs, tptr, tcell, bfs, bptr, bcell, qrule)
                                        copyto!(zlocal_dev, zlocal_host)
                                    end
                                else
                                    fill!(zlocal_host, 0)
                                    momintegrals!(zlocal_host, biop, tfs, tptr, tcell, bfs, bptr, bcell, qrule)
                                    copyto!(zlocal_dev, zlocal_host)
                                end

                                active_test_el_ids_dev = CUDA.cu(Int32[p])
                                active_trial_el_ids_dev = CUDA.cu(Int32[q])
                                kernel_time += @elapsed begin
                                    @cuda threads=256 blocks=1 assembleblock_body_gpu!(
                                        test_assembly_dev, trial_assembly_dev,
                                        active_test_el_ids_dev, active_trial_el_ids_dev,
                                        test_id_in_blk_dev, trial_id_in_blk_dev,
                                        zlocal_dev, store_dev)
                                    CUDA.synchronize()
                                end
                            end
                        end
                    end

                    store_host = Array(store_dev)
                    for j in 1:size(store_host, 2)
                        for i in 1:size(store_host, 1)
                            v = store_host[i, j]
                            v == zero(v) && continue
                            store(v, i, j)
                        end
                    end
                end
            end
            @info "GPU assembleblock timing" kernel_time=kernel_time total_time=total_time total_pairs=total_pairs
        end
    else
        @warn "CUDA not available, falling back to CPU assembly"
        test_elements, test_assembly_data, trial_elements, trial_assembly_data, quadrature_data, zlocals = assembleblock_primer(biop, tfs, bfs; quadstrat=qs, gpu=gpu)
        return (test_ids, trial_ids, store) -> begin
            cpu_time = @elapsed assembleblock_body!(biop,
                tfs, test_ids,   test_elements,  test_assembly_data,
                bfs, trial_ids, trial_elements, trial_assembly_data,
                quadrature_data, zlocals, store; quadstrat=qs)
            @info "CPU assembleblock timing" time=cpu_time
        end
    end

    print("primer finished.\n")

    # if CompScienceMeshes.refines(tgeo, bgeo)
    #     return (test_ids, trial_ids, store) -> begin
    #         assembleblock_body_test_refines_trial!(biop,
    #             tfs, test_ids,   test_elements,  test_assembly_data,
    #             bfs, trial_ids, trial_elements, trial_assembly_data,
    #             quadrature_data, zlocals, store; quadstrat)
    #     end
    # elseif CompScienceMeshes.refines(bgeo, tgeo)
    #     return (test_ids, trial_ids, store) -> begin
    #         assembleblock_body_trial_refines_test!(biop,
    #             tfs, test_ids,   test_elements,  test_assembly_data,
    #             bfs, trial_ids, trial_elements, trial_assembly_data,
    #             quadrature_data, zlocals, store; quadstrat)
    #     end
    # else
    #     return (test_ids, trial_ids, store) -> begin
    #         assembleblock_body!(biop,
    #             tfs, test_ids,   test_elements,  test_assembly_data,
    #             bfs, trial_ids, trial_elements, trial_assembly_data,
    #             quadrature_data, zlocals, store; quadstrat)
    #     end
    # end
end

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
