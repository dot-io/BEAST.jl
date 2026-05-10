# This file implements a first,n naive version of GPU assembly of integral operators in BEAST.


#============================================================#
# AssemblyData
#============================================================#

# Transform assembly data into a format centered around the Dof and transfer it to the GPU.
function load_assemblydata_gpu(X)
    el, ad, cls = BEAST.assemblydata(X)

    num_shapes = numfunctions(refspace(X), domain(el[1]))

    l2g_map = l2g_maps!(ad, numfunctions(X))

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    idx = 1
    ax = axes(ad.data)
    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                dof = ad.data[i, j, k][1]
                if dof > 0
                    push!(rows, dof)
                    push!(cols, num_shapes * (k - 1) + j)
                    push!(vals, ad.data[i, j, k][2])
                    idx += 1
                end
            end
        end
    end

    dof_ad = sparse(rows, cols, vals, length(l2g_map), num_shapes * length(el))

    ad_sparse_d = CuSparseMatrixCSC(dof_ad)
    el_d = CuArray(el)
    return el_d, ad_sparse_d, l2g_map
end



#============================================================#
# Singularity detection
#============================================================#

#Flag singularities based on overlap of test and trial element.
function gpu_singularityflag!(singularity_d, testel_d::CuDeviceVector{S}, trialel_d::CuDeviceVector{S}) where S
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    cols = length(trialel_d)
    rows = length(testel_d)

    if glb_idx <= rows * cols

        T = coordtype(S)

        tol = 1e3 * eps(T)

        i = mod(glb_idx - 1, rows) + 1
        j = div(glb_idx - 1, rows) + 1

        hits = 1
        for t in vertices(testel_d[i])
            for b in vertices(trialel_d[j])
                d = norm(t - b)
                hits += (d < tol)
            end
        end
        @inbounds singularity_d[glb_idx, hits] = true
    end
    return nothing
end

# Asign quadrature strategy based on singularity flags.
function gpu_quadstrat!(quadstrat_d, singularity_map_d, map_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sing_idx = threadIdx().y

    N = size(singularity_map_d, 1)

    if glb_idx <= N && sing_idx <= 4
        if singularity_map_d[glb_idx, sing_idx] == true
            @inbounds quadstrat_d[map_d[glb_idx, sing_idx], sing_idx] = glb_idx
        end
    end

    return nothing
end

# Assign strat to each element pair
function singularitydetection!(quadstrat_d, numpairs, test_el_d, trial_el_d)


    singularity_map_d = CUDA.fill(false, length(test_el_d) * length(trial_el_d), 4)

    launch_gpu_kernel!(gpu_singularityflag!, singularity_map_d, test_el_d, trial_el_d; gpu_blocksize=(256), problem_size=(length(test_el_d) * length(trial_el_d)))


    tally_strats_d = accumulate(+, singularity_map_d, dims=1)

    numpairs .= Array(tally_strats_d[end, :])

    launch_gpu_kernel!(gpu_quadstrat!, quadstrat_d, singularity_map_d, tally_strats_d; gpu_blocksize=(256, 4), problem_size=size(singularity_map_d))

    return
end


#============================================================#
# Shape function evaluation
#============================================================#

function gpu_shapefunction_eval!(shapefunction, el_d, refspace, quadrule)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    qp_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    N = length(el_d)

    if glb_idx <= N && qp_idx <= length(quadrule)

        el = el_d[glb_idx]

        px = quadrule[qp_idx][1]
        #wx = quadrule[qp_idx][2]
        mp = neighborhood(el, px)
        #jx = jacobian(mp)
        val = refspace(mp)
        shapefunction[glb_idx, qp_idx] = val #(val,mp,jx*wx)
    end

    return nothing
end

#============================================================#
# Double Num Quadrature
#============================================================#

function gpu_momintegral_doublenum!(zlocal, biop, npairs, elpairs, test_els, trial_els, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x



    if glb_idx <= npairs

        rows = length(test_els)

        test_npts = length(test_qrule_d)
        trial_npts = length(trial_qrule_d)

        i = mod(elpairs[glb_idx] - 1, rows) + 1
        j = div(elpairs[glb_idx] - 1, rows) + 1

        el_test = test_els[i]
        el_trial = trial_els[j]
        shape_test = view(test_shapes, i, :)
        shape_trial = view(trial_shapes, j, :)
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
        numshapes_test = numfunctions(test_refspace, test_domain)
        numshapes_trial = numfunctions(trial_refspace, trial_domain)

        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)

        z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,ComplexF64})
        for l in 1:test_npts
            px = test_qrule_d[l][1]
            #wx = qrule_d[l][2]
            x = neighborhood(el_test, px)
            wx = test_qrule_d[l][2] * jacobian(x)
            for m in 1:trial_npts
                py = trial_qrule_d[m][1]
                #wy = qrule_d[m][2]
                #@cuprintln("Test shapes for element $i: $(BEAST.getvalue(shape_test[l])[1][1]), $(BEAST.getvalue(shape_test[l])[1][2]), $(BEAST.getvalue(shape_test[l])[1][3]) )")
                y = neighborhood(el_trial, py)
                wy = trial_qrule_d[m][2] * jacobian(y)
                #@cuprintln("Trial shapes for element $j: $(BEAST.getvalue(shape_trial[m])[1][1]), $(BEAST.getvalue(shape_trial[m])[1][2]), $(BEAST.getvalue(shape_trial[m])[1][3]) )")
                #@inbounds z += shape_test[l][3]*shape_trial[m][3]*igd(shape_test[l][2],shape_trial[m][2],shape_test[l][1],shape_trial[m][1])

                @inbounds z += wx * wy * igd(x, y, shape_test[l], shape_trial[m])
            end
        end
        view(zlocal, numshapes_test*(i-1)+1:numshapes_test*i, numshapes_trial*(j-1)+1:numshapes_trial*j) .= z
    end

    return nothing
end

#============================================================#
# SauterSchwab Quadrature
#============================================================#


#TODO Move the reorder functions to SauterSchwabQuadrature.jl

#Calculat set difference A B assuming A contains all element of B
function gpu_setdifference(A::SVector{N1,T}, B::SVector{N2,T}) where {N1,N2,T}

    out = zeros(StaticArrays.MVector{N1 - N2,Int64})
    l = 1
    for i in eachindex(A)
        found = false
        for j in eachindex(B)
            if A[i] == B[j]
                found = true
            end
        end
        if !found
            @inbounds out[l] = A[i]
            l += 1
        end
    end

    return out
end

function gpu_sauterschwab_reorder(t, s, strat::CommonVertex)
    T = eltype(t[1])
    tol = 1e4 * eps(T)

    # Find the permutation P of t and s that make
    # Pt = [P, A1, A2]
    # Ps = [P, B1, B2]
    I1 = 0
    J1 = 0
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                I1 = i
                J1 = j
                e += 1
                break
            end
        end
        e == 2 && break
    end

    A = SVector{3,Int64}(1, 2, 3)
    a = gpu_setdifference(A, SVector{1,Int64}(I1))
    b = gpu_setdifference(A, SVector{1,Int64}(J1))

    I = SVector{3,Int64}(I1, a...)
    J = SVector{3,Int64}(J1, b...)

    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t, s, strat::CommonEdge)
    T = eltype(t[1])
    tol = 1e3 * eps(T)


    I1, I2 = 0, 0
    J1, J2 = 0, 0
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                if e == 1
                    I1 = i
                    J1 = j
                    e += 1
                elseif e == 2
                    I2 = i
                    J2 = j
                    e += 1
                end
                break
            end
        end
    end

    #cricshift of -1
    I = SVector{3,Int64}(I2, 6 - I2 - I1, I1)
    J = SVector{3,Int64}(J2, 6 - J2 - J1, J1)


    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t, s, strat::CommonFace)
    T = eltype(t[1])
    tol = 1e3 * eps(T)
    # tol = 1e5 * eps(T)
    # tol = sqrt(eps(T))


    I = SVector{3,Int64}(1, 2, 3)
    J = zeros(MVector{3,Int64})
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                J[i] = j
                e += 1
            end
        end
    end

    J = SVector{3,Int64}(J...)

    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_momintegral_sauterschwab!(zlocal, biop, npairs, elpairs, test_els, trial_els, test_refspace, trial_refspace, strat)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x


    if glb_idx <= npairs

        rows = length(test_els)

        i = mod(elpairs[glb_idx] - 1, rows) + 1
        j = div(elpairs[glb_idx] - 1, rows) + 1

        el_test = test_els[i]
        el_trial = trial_els[j]
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
        numshapes_test = numfunctions(test_refspace, test_domain)
        numshapes_trial = numfunctions(trial_refspace, trial_domain)

        #I = StaticArrays.SVector{3,Int64}(1,2,3)
        #J = StaticArrays.SVector{3,Int64}(1,2,3)
        I, J, _, _ = gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
        igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

        z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,ComplexF64})

        qps = strat.qps
        for (η1, w1) in qps
            for (η2, w2) in qps
                for (η3, w3) in qps
                    for (ξ, w4) in qps
                        z += w1 * w2 * w3 * w4 * strat(igdp, η1, η2, η3, ξ)
                    end
                end
            end
        end
        view(zlocal, numshapes_test*(i-1)+1:numshapes_test*i, numshapes_trial*(j-1)+1:numshapes_trial*j) .= z
    end

    return nothing
end


#============================================================#
# Assemble matrix
#============================================================#


function build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d)

    ZAT = CUDA.fill(zero(promote_type(eltype(zlocal_d), eltype(trial_ad_d))), size(trial_ad_d, 1), size(zlocal_d, 1))
    CUSPARSE.mm!('N', 'T', 1.0, trial_ad_d, zlocal_d, 0.0, ZAT, 'O')
    CUSPARSE.mm!('N', 'T', 1.0, test_ad_d, ZAT, 0.0, matrix_d, 'O')
    return nothing
end


#=======================================#
# BEAST assemble function for GPU threading
#=======================================#

function assemble!(operator::Operator, test_functions::Space, trial_functions::Space,
    store, threading::Type{Threading{:gpu}};
    quadstrat=defaultquadstrat, tilingstrat=TilingStrategy(EqualTiling(1), EqualTiling(1)), kwargs...)

    quadstrat = quadstrat(operator, test_functions, trial_functions)

    #test_splits = [round(Int,s) for s in range(0, stop=numfunctions(test_functions), length=gpu_tiling[1]+1)]
    #trial_splits = [round(Int,s) for s in range(0, stop=numfunctions(trial_functions), length=gpu_tiling[2]+1)]
    println("GPU assemble called.")

    test_geo = geometry(test_functions)
    trial_geo = geometry(trial_functions)

    test_splits = split(numcells(test_geo), tilingstrat[1])
    trial_splits = split(numcells(trial_geo), tilingstrat[2])


    # test_splits = split(numfunctions(test_functions), tilingstrat[1])
    # trial_splits = split(numfunctions(test_functions), tilingstrat[2])

    println("Gpu tiling: $tilingstrat")
    @show first.(test_splits)
    @show first.(trial_splits)

    test_l2g = Vector{Vector{Int}}(undef, length(test_splits))
    test_ad_qd = Vector{Tuple{Tuple{CuArray,CuSparseMatrixCSC},Tuple{CuArray,CuArray}}}(undef, length(test_splits))

    # In the case of index tiling, the *_splits vectors are 1-D. this is a quick workaround for now
    if ndims(test_splits) == 1
        test_splits = [test_splits]
    end
    if ndims(trial_splits) == 1
        trial_splits = [trial_splits]
    end
    for i in eachindex(test_splits)

        # test_subgeo = CompScienceMeshes.SubMesh(test_geo, test_splits[i])
        # test_functions_p = restrict(test_functions, test_subgeo)

        test_functions_p = subset(test_functions, test_splits[i])

        test_l2g[i], test_ad_qd[i] = assemble_primer_gpu(operator, test_functions_p, quadstrat.outer_rule)
    end

    trial_l2g = Vector{Vector{Int}}(undef, length(trial_splits))
    trial_ad_qd = Vector{Tuple{Tuple{CuArray,CuSparseMatrixCSC},Tuple{CuArray,CuArray}}}(undef, length(trial_splits))

    for i in eachindex(trial_splits)
        # trial_subgeo = CompScienceMeshes.SubMesh(trial_geo, trial_splits[i])
        # trial_functions_p = restrict(trial_functions, trial_subgeo)

        trial_functions_p = subset(trial_functions, trial_splits[i])

        trial_l2g[i], trial_ad_qd[i] = assemble_primer_gpu(operator, trial_functions_p, quadstrat.inner_rule)
    end

    #TODO Convert all rules for CommonVertex, CommonEdge, and CommonFace.
    cv_rule = CompScienceMeshes.legendre(quadstrat.sauter_schwab_common_vert, 0.0, 1.0)
    # @show cv_rule
    q = Array{Tuple{Float64,Float64}}(undef, length(cv_rule[2]))
    for (i, a) in enumerate(zip(cv_rule[1], cv_rule[2]))
        q[i] = a
    end
    # @show q
    cvrule_d = CuArray(q)

    for i in eachindex(test_splits)

        (test_el_d, test_ad_d), test_qd = test_ad_qd[i]
        #test_functions_p = subset(test_functions, test_splits[i])

        for j in eachindex(trial_splits)

            #trial_functions_p = subset(trial_functions, trial_splits[j])
            (trial_el_d, trial_ad_d), trial_qd = trial_ad_qd[j]


            qd_d = (test_qd, trial_qd, cvrule_d)
            store1(v, m, n) = store(v, test_l2g[i][m], trial_l2g[j][n]) #) BEAST._OffsetStore(store, lo_test-1, lo_trial-1)

            assemblechunk_body_gpu!(operator,
                refspace(test_functions), test_el_d, test_ad_d,
                refspace(trial_functions), trial_el_d, trial_ad_d,
                qd_d, store1)
            #assemblechunk_gpu!(operator, test_functions_p, trial_functions_p, store1, quadstrat=quadstrat)
        end
    end

    #TODO: tiling of operator assembly to fit GPU memory constraints.

    #assemblechunk_gpu!(operator, test_functions, trial_functions, store; quadstrat =quadstrat)

end

function assemble_primer_gpu(operator::Operator, functions::Space, quadrule)

    space = refspace(functions)

    qrule = CompScienceMeshes.trgauss(quadrule)
    # @show qrule
    q = Array{Tuple{SVector{2,Float64},Float64}}(undef, length(qrule[2]))
    for (i, a) in enumerate(zip(eachcol(qrule[1]), qrule[2]))
        q[i] = a
    end
    #@show q
    quadrule_d = CuArray(q)

    el_d, ad_d, l2g_map = load_assemblydata_gpu(functions)

    chart = CUDA.@allowscalar domain(el_d[1])
    numshapes = numfunctions(space, chart)

    #@show eltype(test_el_d)
    shapefunction_type = shapetype(space)
    shapes_d = CuArray{SVector{numshapes,shapefunction_type}}(undef, length(el_d), length(quadrule_d))

    launch_gpu_kernel!(gpu_shapefunction_eval!, shapes_d, el_d, space, quadrule_d;
        gpu_blocksize=(128, 4), problem_size=(length(el_d), length(quadrule_d)))


    return l2g_map, ((el_d, ad_d), (quadrule_d, shapes_d))
end



function assemblechunk_body_gpu!(operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_d::CuSparseMatrixCSC,
    qd_d, store)


    (quadrule_test_d, test_shapes_d), (quadrule_trial_d, trial_shapes_d), cvrule_d = qd_d


    test_domain = CUDA.@allowscalar domain(test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(trial_el_d[1])

    numshapes_test = numfunctions(test_space, test_domain)
    numshapes_trial = numfunctions(trial_space, trial_domain)


    quadstrat_d = CUDA.fill(0, length(test_el_d) * length(trial_el_d), 4)
    numpairs = zeros(Int, 4)
    #println("Singularity detection")
    singularitydetection!(quadstrat_d, numpairs, test_el_d, trial_el_d)

    @show numpairs

    zlocal_d = CUDA.fill(zero(ComplexF64), numshapes_test * length(test_el_d), length(trial_el_d) * numshapes_trial)

    #println("Double num integrals")
    launch_gpu_kernel!(gpu_momintegral_doublenum!, zlocal_d, operator, numpairs[1], quadstrat_d[:, 1],
        test_el_d, trial_el_d, test_shapes_d, trial_shapes_d, test_space, trial_space, quadrule_test_d, quadrule_trial_d;
        gpu_blocksize=(256), problem_size=(numpairs[1]))

    #println("Common vertex integrals")
    strategy = CommonVertex(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[2], quadstrat_d[:, 2], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[2]))

    #println("Common edge integrals")
    strategy = CommonEdge(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[3], quadstrat_d[:, 3], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[3]))

    #println("Common face integrals")
    strategy = CommonFace(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[4], quadstrat_d[:, 4], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[4]))

    #@show CUDA.@allowscalar zlocal_d[end]

    matrix_d = CUDA.fill(0.0 + 0.0im, size(test_ad_d, 1), size(trial_ad_d, 1))
    # println("Building global matrix")


    build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d)
    #launch_gpu_kernel!(gpu_build_matrix!,matrix_d, zlocal_d, test_ad_d, trial_ad_d, numshapes_test, numshapes_trial;
    #                   gpu_blocksize=(512,1), problem_size=(numfunctions(test_functions), numfunctions(trial_functions)))

    matrix = Array(matrix_d)

    # println(matrix[end,end])
    # println("Store called")
    for j in 1:size(matrix, 2)
        for i in 1:size(matrix, 1)
            store(matrix[i, j], i, j)
        end
    end
end

function assemble_block_gpu(operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_cpu::SparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_cpu::SparseMatrixCSC,
    qd_d, testids, trialids;
    numshapes_test::Union{Nothing,Int}=nothing,
    numshapes_trial::Union{Nothing,Int}=nothing,
    keep_on_gpu::Bool=false)

    (quadrule_test_d, test_shapes_d), (quadrule_trial_d, trial_shapes_d), cvrule_d = qd_d

    testids_d = CuArray(collect(Int, testids))
    trialids_d = CuArray(collect(Int, trialids))
    test_el_sub = test_el_d[testids_d]
    trial_el_sub = trial_el_d[trialids_d]
    test_shapes_sub = test_shapes_d[testids_d, :]
    trial_shapes_sub = trial_shapes_d[trialids_d, :]

    if numshapes_test === nothing
        test_domain = CUDA.@allowscalar domain(test_el_sub[1])
        numshapes_test = numfunctions(test_space, test_domain)
    end
    if numshapes_trial === nothing
        trial_domain = CUDA.@allowscalar domain(trial_el_sub[1])
        numshapes_trial = numfunctions(trial_space, trial_domain)
    end

    n_test = length(testids)
    n_trial = length(trialids)

    quadstrat_d = CUDA.fill(0, n_test * n_trial, 4)
    numpairs = zeros(Int, 4)
    singularitydetection!(quadstrat_d, numpairs, test_el_sub, trial_el_sub)

    zlocal_d = CUDA.fill(zero(ComplexF64), numshapes_test * n_test, n_trial * numshapes_trial)

    launch_gpu_kernel!(gpu_momintegral_doublenum!, zlocal_d, operator, numpairs[1], quadstrat_d[:, 1],
        test_el_sub, trial_el_sub, test_shapes_sub, trial_shapes_sub, test_space, trial_space, quadrule_test_d, quadrule_trial_d;
        gpu_blocksize=(256), problem_size=(numpairs[1]))

    strategy = CommonVertex(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[2], quadstrat_d[:, 2], test_el_sub, trial_el_sub,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[2]))

    strategy = CommonEdge(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[3], quadstrat_d[:, 3], test_el_sub, trial_el_sub,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[3]))

    strategy = CommonFace(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[4], quadstrat_d[:, 4], test_el_sub, trial_el_sub,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[4]))

    test_col_idcs = vec([numshapes_test * (testids[k] - 1) + s for s in 1:numshapes_test, k in 1:n_test])
    trial_col_idcs = vec([numshapes_trial * (trialids[k] - 1) + s for s in 1:numshapes_trial, k in 1:n_trial])

    test_ad_sub_cpu = test_ad_cpu[:, test_col_idcs]
    trial_ad_sub_cpu = trial_ad_cpu[:, trial_col_idcs]
    test_active_rows = unique!(sort!(copy(rowvals(test_ad_sub_cpu))))
    trial_active_rows = unique!(sort!(copy(rowvals(trial_ad_sub_cpu))))
    test_ad_sub = CuSparseMatrixCSC(test_ad_sub_cpu[test_active_rows, :])
    trial_ad_sub = CuSparseMatrixCSC(trial_ad_sub_cpu[trial_active_rows, :])

    matrix_d = CUDA.fill(0.0 + 0.0im, length(test_active_rows), length(trial_active_rows))
    build_matrix!(matrix_d, zlocal_d, test_ad_sub, trial_ad_sub)

    if keep_on_gpu
        return matrix_d, test_active_rows, trial_active_rows
    else
        matrix = Array(matrix_d)
        return matrix, test_active_rows, trial_active_rows
    end
end

function assembleblock_body_gpu!(operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_cpu::SparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_cpu::SparseMatrixCSC,
    qd_d, testids, trialids, store)

    matrix, test_active_rows, trial_active_rows = assemble_block_gpu(
        operator,
        test_space, test_el_d, test_ad_cpu,
        trial_space, trial_el_d, trial_ad_cpu,
        qd_d, testids, trialids,
    )

    for j in 1:size(matrix, 2)
        for i in 1:size(matrix, 1)
            store(matrix[i, j], test_active_rows[i], trial_active_rows[j])
        end
    end
end

function assemblechunk_gpu!(operator::IntegralOperator, test_functions::Space, trial_functions::Space, store;
    quadstrat=BEAST.defaultquadstrat, gpu_blocksize=(0, 0))

    println("GPU assemble called.")
    test_space = refspace(test_functions)
    trial_space = refspace(trial_functions)



    qrule = CompScienceMeshes.trgauss(quadstrat(operator, test_space, trial_space).outer_rule)
    # @show qrule
    q = Array{Tuple{SVector{2,Float64},Float64}}(undef, length(qrule[2]))
    for (i, a) in enumerate(zip(eachcol(qrule[1]), qrule[2]))
        q[i] = a
    end
    #@show q
    quadrule_d = CuArray(q)


    cv_rule = CompScienceMeshes.legendre(quadstrat(operator, test_space, trial_space).sauter_schwab_common_vert, 0.0, 1.0)
    # @show cv_rule
    q = Array{Tuple{Float64,Float64}}(undef, length(cv_rule[2]))
    for (i, a) in enumerate(zip(cv_rule[1], cv_rule[2]))
        q[i] = a
    end
    # @show q
    cvrule_d = CuArray(q)

    test_el_d, test_ad_d = load_assemblydata_gpu(test_functions)
    trial_el_d, trial_ad_d = load_assemblydata_gpu(trial_functions)

    test_domain = CUDA.@allowscalar domain(test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(trial_el_d[1])

    numshapes_test = numfunctions(test_space, test_domain)
    numshapes_trial = numfunctions(trial_space, trial_domain)

    #@show eltype(test_el_d)
    shapefunction_type = shapetype(test_space)
    test_shapes_d = CuArray{SVector{numshapes_test,shapefunction_type}}(undef, length(test_el_d), length(quadrule_d))
    shapefunction_type = shapetype(trial_space)
    trial_shapes_d = CuArray{SVector{numshapes_trial,shapefunction_type}}(undef, length(trial_el_d), length(quadrule_d))

    #meshpoint_type = CompScienceMeshes.MeshPointNM{Float64,eltype(test_el_d),2,3}
    #test_shapes_d = CuArray{Tuple{SVector{numfunctions(test_space,test_domain),shapefunction_type},meshpoint_type,Float64}}(undef, length(test_el_d), length(quadrule_d))
    #test_shapes_d = CUDA.fill((value=zeros(SVector{3,Float64}),divergence=0.0), length(test_el_d), length(quadrule_d))
    #trial_shapes_d = CuArray{Tuple{SVector{numfunctions(trial_space,trial_domain),shapefunction_type},meshpoint_type,Float64}}(undef, length(trial_el_d), length(quadrule_d))
    #trial_shapes_d = CUDA.fill((value=zeros(SVector{3,Float64}),divergence=0.0),  length(trial_el_d), length(quadrule_d))
    # @show CUDA.@allowscalar test_shapes_d[1,1][1]
    #  @show CUDA.@allowscalar test_shapes_d[1,1][2]
    #   @show CUDA.@allowscalar test_shapes_d[1,1][3]

    launch_gpu_kernel!(gpu_shapefunction_eval!, test_shapes_d, test_el_d, test_space, quadrule_d;
        gpu_blocksize=(128, 4), problem_size=(length(test_el_d), length(quadrule_d)))
    launch_gpu_kernel!(gpu_shapefunction_eval!, trial_shapes_d, trial_el_d, trial_space, quadrule_d;
        gpu_blocksize=(128, 4), problem_size=(length(trial_el_d), length(quadrule_d)))

    #@show CUDA.@allowscalar test_shapes_d[1,:]
    #@show CUDA.@allowscalar trial_shapes_d[1,:]
    #@show size(trial_el_d)

    quadstrat_d = CUDA.fill(0, length(test_el_d) * length(trial_el_d), 4)
    numpairs = zeros(Int, 4)
    #println("Singularity detection")
    singularitydetection!(quadstrat_d, numpairs, test_el_d, trial_el_d)

    @show numpairs

    zlocal_d = CUDA.fill(zero(ComplexF64), numshapes_test * length(test_el_d), length(trial_el_d) * numshapes_trial)

    #println("Double num integrals")
    launch_gpu_kernel!(gpu_momintegral_doublenum!, zlocal_d, operator, numpairs[1], quadstrat_d[:, 1],
        test_el_d, trial_el_d, test_shapes_d, trial_shapes_d, test_space, trial_space, quadrule_d;
        gpu_blocksize=(256), problem_size=(numpairs[1]))

    #println("Common vertex integrals")
    strategy = CommonVertex(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[2], quadstrat_d[:, 2], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[2]))

    #println("Common edge integrals")
    strategy = CommonEdge(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[3], quadstrat_d[:, 3], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[3]))

    #println("Common face integrals")
    strategy = CommonFace(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[4], quadstrat_d[:, 4], test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[4]))

    #@show CUDA.@allowscalar zlocal_d[end]

    matrix_d = CUDA.fill(0.0 + 0.0im, numfunctions(test_functions), numfunctions(trial_functions))
    # println("Building global matrix")


    build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d)
    #launch_gpu_kernel!(gpu_build_matrix!,matrix_d, zlocal_d, test_ad_d, trial_ad_d, numshapes_test, numshapes_trial;
    #                   gpu_blocksize=(512,1), problem_size=(numfunctions(test_functions), numfunctions(trial_functions)))

    matrix = Array(matrix_d)

    # println(matrix[end,end])
    # println("Store called")
    for j in 1:size(matrix, 2)
        for i in 1:size(matrix, 1)
            store(matrix[i, j], i, j)
        end
    end
end




# function numhits(test_el::CompScienceMeshes.Simplex,trial_el::CompScienceMeshes.Simplex)

#         hits = 0
#         dtol = 1e-6
#         for t in test_el.vertices
#             for b in trial_el.vertices
#                 d2 = sum((t-b).^2)
#                 d = sqrt(d2)
#                 hits += (d < dtol)
#             end
#         end
# end






# function singularitydetection_config!(singularity_d,testel_d,trialel_d)
#     kernel = @cuda launch=false gpu_singularitydetection!(singularity_d,testel_d,trialel_d)
#     config = launch_configuration(kernel.fun)


#     println(config)
#     threads = config.threads
#     blocks  = config.blocks

#     CUDA.@sync begin
#          kernel(singularity_d,testel_d,trialel_d; threads, blocks)
#     end

#     return
# end


# function quadstrat_config!(quadstart_d,singularity_d,map_d)
#     kernel = @cuda launch=false gpu_quadstrat!(quadstart_d,singularity_d,map_d)
#     config = launch_configuration(kernel.fun)

#     println(config)
#     threads = config.threads
#     blocks  = config.blocks

#     CUDA.@sync begin
#          kernel(quadstart_d,singularity_d,map_d; threads, blocks)
#     end

#     return
# end







# function gpu_momintegral_cv!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     glb_stride = gridDim().x * blockDim().x

#     K = npairs

#     if glb_idx > K
#         return nothing
#     end

#     N = length(test_els)

#     strat =  SauterSchwabQuadrature.CommonVertex(qrule_d)
#     for k in glb_idx:glb_stride:K

#         i =  mod(elpairs[k]-1,N) + 1
#         j =  div(elpairs[k]-1,N) + 1

#         el_test = test_els[i]
#         el_trial = trial_els[j]
#         test_domain = CompScienceMeshes.domain(el_test)
#         trial_domain = CompScienceMeshes.domain(el_trial)

#         #I = StaticArrays.SVector{3,Int64}(1,2,3)
#         #J = StaticArrays.SVector{3,Int64}(1,2,3)
#         I,J,_,_ =  gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

#         igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
#         igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

#         z = zeros(StaticArrays.SMatrix{ numfunctions(test_refspace,test_domain), numfunctions(trial_refspace,trial_domain),ComplexF64})
#         #@inline gpu_sauterschawb_parametrized!(zlocal[k],igdp,strat)
#         qps = strat.qps
#         for (η1,w1) in qps
#             for (η2,w2) in qps
#                 for (η3,w3) in qps
#                     for (ξ,w4) in qps
#                         z += w1 * w2 * w3 * w4 *  strat(igdp, η1, η2, η3, ξ)
#                     end
#                 end
#             end
#         end
#         zlocal[elpairs[k]] = z
#     end

#     return nothing
# end


# function gpu_momintegral_ce!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     glb_stride = gridDim().x * blockDim().x

#     K = npairs

#     if glb_idx > K
#         return nothing
#     end

#     N = length(test_els)

#     strat =  SauterSchwabQuadrature.CommonEdge(qrule_d)
#     for k in glb_idx:glb_stride:K

#         i =  mod(elpairs[k]-1,N) + 1
#         j =  div(elpairs[k]-1,N) + 1

#         el_test = test_els[i]
#         el_trial = trial_els[j]
#         test_domain = CompScienceMeshes.domain(el_test)
#         trial_domain = CompScienceMeshes.domain(el_trial)

#         #I = StaticArrays.SVector{3,Int64}(1,2,3)
#         #J = StaticArrays.SVector{3,Int64}(1,2,3)
#         @inline I,J,_,_ =  gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

#         igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
#         igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

#         z = zeros(StaticArrays.SMatrix{ numfunctions(test_refspace,test_domain), numfunctions(trial_refspace,trial_domain),ComplexF64})
#         #@inline gpu_sauterschawb_parametrized!(zlocal[k],igdp,strat)
#         qps = strat.qps
#         for (η1,w1) in qps
#             for (η2,w2) in qps
#                 for (η3,w3) in qps
#                     for (ξ,w4) in qps
#                         z += w1 * w2 * w3 * w4 *  strat(igdp, η1, η2, η3, ξ)
#                     end
#                 end
#             end
#         end
#         zlocal[elpairs[k]] = z
#     end

#     return nothing
# end



# function gpu_momintegral_cf!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     glb_stride = gridDim().x * blockDim().x

#     K = npairs

#     if glb_idx > K
#         return nothing
#     end

#     N = length(test_els)

#     strat =  SauterSchwabQuadrature.CommonFace(qrule_d)
#     for k in glb_idx:glb_stride:K

#         i =  mod(elpairs[k]-1,N) + 1
#         j =  div(elpairs[k]-1,N) + 1

#         el_test = test_els[i]
#         el_trial = trial_els[j]
#         test_domain = CompScienceMeshes.domain(el_test)
#         trial_domain = CompScienceMeshes.domain(el_trial)

#         #I = StaticArrays.SVector{3,Int64}(1,2,3)
#         #J = StaticArrays.SVector{3,Int64}(1,2,3)
#         I,J,_,_ =  gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

#         igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
#         igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

#         z = zeros(StaticArrays.SMatrix{ numfunctions(test_refspace,test_domain), numfunctions(trial_refspace,trial_domain),ComplexF64})
#         #@inline gpu_sauterschawb_parametrized!(zlocal[k],igdp,strat)
#         qps = strat.qps
#         for (η1,w1) in qps
#             for (η2,w2) in qps
#                 for (η3,w3) in qps
#                     for (ξ,w4) in qps
#                         z += w1 * w2 * w3 * w4 *  strat(igdp, η1, η2, η3, ξ)
#                     end
#                 end
#             end
#         end
#         zlocal[elpairs[k]] = z
#     end

#     return nothing
# end


# function momintegral_config!(zlocal_d,gpu_kernel,biop,npairs,elpairs_d,test_els_d,trial_els_d,test_refspace,trial_refspace,qrule_d)
#     kernel = @cuda launch=false gpu_kernel(zlocal_d,biop,npairs,elpairs_d,test_els_d,trial_els_d,test_refspace,trial_refspace,qrule_d)
#     config = launch_configuration(kernel.fun)

#     println(config)
#     threads = config.threads
#     blocks  = config.blocks

#     CUDA.@sync begin
#          kernel(zlocal_d,biop,npairs,elpairs_d,test_els_d,trial_els_d,test_refspace,trial_refspace,qrule_d; threads, blocks)
#     end

#     return
# end


# function gpu_momintegral_cv!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     strat =  CommonVertex(qrule_d)
#     gpu_momintegral_sauterschwab!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,strat)

#     return nothing
# end

# function gpu_momintegral_ce!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     strat =  CommonEdge(qrule_d)
#     gpu_momintegral_sauterschwab!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,strat)

#     return nothing
# end

# function gpu_momintegral_cf!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
#     strat =  CommonFace(qrule_d)
#     gpu_momintegral_sauterschwab!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,strat)

#     return nothing
# end
