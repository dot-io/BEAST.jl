# This file implements a first, naive version of GPU assembly of integral operators in BEAST.

#============================================================#
# AssemblyData
#============================================================#

# Transform assembly data into a format centered around the Dof and transfer it to the GPU.
function load_assemblydata_gpu(X)
    el, ad, cls = BEAST.assemblydata(X)

    res = zeros(Int,numfunctions(X))
    ax = axes(ad.data)
    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                idx = ad.data[i,j,k][1]
                if idx > 0
                    res[ad.data[i,j,k][1]] += 1
                end
            end
        end
    end

    max_els = maximum(res)

    dof_ad = Vector{Vector{Tuple{Int,Int,Float64}}}(undef, numfunctions(X))

    for i in 1:numfunctions(X)
        dof_ad[i] = Vector{Tuple{Int,Int,Float64}}(undef, max_els)
        fill!(dof_ad[i], (0,0,0.0))
    end

    idx_count = ones(Int, numfunctions(X))
    ax = axes(ad.data)
    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                idx = ad.data[i,j,k][1]
                if idx > 0
                    dof_ad[idx][idx_count[idx]] = (k,j,ad.data[i,j,k][2])
                    idx_count[idx] += 1
                end
            end
        end
    end

    dof_ad2 = Vector{SVector{max_els,Tuple{Int,Int,Float64}}}(undef, numfunctions(X))
    for i in 1:numfunctions(X)
        dof_ad2[i] = SVector{max_els,Tuple{Int,Int,Float64}}(dof_ad[i]...)
    end


    el_d = CuArray(el)
    dof_ad_d = CuArray(dof_ad2)
    cells_d = CuArray(cls)

    return el_d,dof_ad_d,cells_d
end

#============================================================#
# Singularity detection
#============================================================#

#Flag singularities based on overlap of test and trial element.
function gpu_singularityflag!(singularity_d,testel_d,trialel_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    glb_stride = gridDim().x * blockDim().x

    N = length(trialel_d)
    M = length(testel_d)

    if glb_idx > N*M
        return nothing
    end
    
    T = eltype(testel_d[1].vertices[1])
    tol = 1e3 * eps(T)
    for k in glb_idx:glb_stride:M*N
        i =  mod(k-1,M) + 1
        j =  div(k-1,M) + 1

        #@cuprintln("$glb_idx, $k, test: $i, trial: $j")
        hits = 1
        for t in testel_d[i].vertices
            for b in trialel_d[j].vertices
                d = norm(t - b) 
                hits += (d < tol)
            end
        end
        @inbounds singularity_d[k,hits] = true
    end
    return nothing
end

# Asign quadrature strategy based on singularity flags.
function gpu_quadstrat!(quadstrat_d, singularity_map_d,map_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    glb_stride = gridDim().x * blockDim().x

    N = size(singularity_map_d,1)

    if glb_idx > N
        return nothing
    end

    for k in glb_idx:glb_stride:N
        if singularity_map_d[k,1] == true
            @inbounds quadstrat_d[map_d[k,1],1] = k
        end
        if singularity_map_d[k,2] == true
            @inbounds quadstrat_d[map_d[k,2],2] = k
        end
        if singularity_map_d[k,3] == true
            @inbounds quadstrat_d[map_d[k,3],3] = k
        end
        if singularity_map_d[k,4] == true
            @inbounds quadstrat_d[map_d[k,4],4] = k
        end
    end

    return nothing
end

# Assign strat to each element pair 
function gpu_singularitydetection!(quadstrat_d,numpairs,test_el_d,trial_el_d)


    singularity_map_d = CUDA.fill(false,length(test_el_d)*length(trial_el_d),4)

    kernel_config!(gpu_singularityflag!,singularity_map_d,test_el_d,trial_el_d)


    tally_strats_d = accumulate(+,singularity_map_d, dims=1)

    numpairs .= Array(tally_strats_d[end,:])

    kernel_config!(gpu_quadstrat!,quadstrat_d,singularity_map_d,tally_strats_d)

    return
end

#============================================================#
# Double Num Quadrature
#============================================================#

function gpu_momintegral_doublenum!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,qrule_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    glb_stride = gridDim().x * blockDim().x

    K = npairs[1]

    if glb_idx > K
        return nothing
    end

    N = length(test_els)
    M = length(trial_els)

    npts = length(qrule_d)

    for k in glb_idx:glb_stride:K
   
        i =  mod(elpairs[k]-1,N) + 1
        j =  div(elpairs[k]-1,N) + 1

        el_test = test_els[i]
        el_trial = trial_els[j]
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)

        z = zeros(StaticArrays.SMatrix{ numfunctions(test_refspace,test_domain), numfunctions(trial_refspace,trial_domain),ComplexF64})
        for l in 1:npts
            px = qrule_d[l][1]
            wx = qrule_d[l][2]

            for m in 1:npts
                py = qrule_d[m][1]
                wy = qrule_d[m][2]

                z += wy*wx* igd(px,py)
            end
        end
        zlocal[elpairs[k]] = z
    end

    return nothing
end

#============================================================#
# SauterSchwab Quadrature
#============================================================#


#TODO Move the reorder functions to SauterSchwabQuadrature.jl

#Calculat set difference A B assuming A contains all element of B
function gpu_setdifference(A::SVector{N1,T},B::SVector{N2,T}) where {N1,N2,T}
    
    out =  zeros(StaticArrays.MVector{N1-N2,Int64})
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
            l+=1
        end
    end

    return out
end

function gpu_sauterschwab_reorder(t,s,strat::CommonVertex)
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

    A = SVector{3,Int64}(1,2,3)
    a = gpu_setdifference(A, SVector{1,Int64}(I1))
    b = gpu_setdifference(A, SVector{1,Int64}(J1))

    I = SVector{3,Int64}(I1,a...)
    J = SVector{3,Int64}(J1,b...)

    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t,s,strat::CommonEdge)
    T = eltype(t[1])
    tol = 1e3 * eps(T)


    I1,I2 = 0,0
    J1,J2 = 0,0
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                if e==1
                    I1 = i
                    J1 = j
                    e += 1
                elseif e==2
                    I2 = i
                    J2 = j
                    e += 1
                end
                break
            end
        end
    end

    #cricshift of -1
    I = SVector{3,Int64}(I2,6-I2-I1,I1)
    J = SVector{3,Int64}(J2,6-J2-J1,J1)


    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t,s,strat::CommonFace)
    T = eltype(t[1])
    tol = 1e3 * eps(T)
    # tol = 1e5 * eps(T)
    # tol = sqrt(eps(T))


    I = SVector{3,Int64}(1,2,3)
    J = zeros(MVector{3,Int64})
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if  norm(w - v) < tol
                J[i] = j
                e +=1
            end
        end
    end

    J = SVector{3,Int64}(J...)
   
    #I = StaticArrays.SVector{3,Int64}(1,2,3)
    #J = StaticArrays.SVector{3,Int64}(1,2,3)
    return I, J, nothing, nothing
end


function gpu_momintegral_sauterschwab!(zlocal,biop,npairs,elpairs,test_els,trial_els,test_refspace,trial_refspace,strat)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    glb_stride = gridDim().x * blockDim().x

    K = npairs[1]

    if glb_idx > K
        return nothing
    end

    N = length(test_els)
  
    for k in glb_idx:glb_stride:K
   
        i =  mod(elpairs[k]-1,N) + 1
        j =  div(elpairs[k]-1,N) + 1

        el_test = test_els[i]
        el_trial = trial_els[j]
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
       
        #I = StaticArrays.SVector{3,Int64}(1,2,3)
        #J = StaticArrays.SVector{3,Int64}(1,2,3)
        I,J,_,_ =  gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)
        
        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
        igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

        z = zeros(StaticArrays.SMatrix{ numfunctions(test_refspace,test_domain), numfunctions(trial_refspace,trial_domain),ComplexF64})
        
        qps = strat.qps
        for (η1,w1) in qps
            for (η2,w2) in qps
                for (η3,w3) in qps
                    for (ξ,w4) in qps
                        z += w1 * w2 * w3 * w4 *  strat(igdp, η1, η2, η3, ξ)  
                    end
                end
            end
        end
        zlocal[elpairs[k]] = z
    end

    return nothing
end


#============================================================#
# Assemble matrix
#============================================================#

function gpu_build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d,test_el_d,trial_el_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    glb_stride = gridDim().x * blockDim().x

    K = size(test_ad_d,1)*size(trial_ad_d,1)

    if glb_idx > K
        return nothing
    end

    # TODO: Save guard against irregular assembly data where some dofs have more associated elements than others.
    for k in glb_idx:glb_stride:K
        m =  mod(k-1,size(test_ad_d,1)) + 1
        l =  div(k-1,size(test_ad_d,1)) + 1
        for (test_idx,j,test_coeff) in test_ad_d[m]
            for (trial_idx,i,trial_coeff) in trial_ad_d[l]
                if trial_idx == 0 || test_idx == 0
                    continue
                end
                elpair_idx = (trial_idx-1)*length(test_el_d) + test_idx
           
                @inbounds matrix_d[m,l] += test_coeff * trial_coeff * zlocal_d[elpair_idx][j,i]
            end
        end
    end

    return nothing
end


#=======================================#
# BEAST assemble function for GPU threading
#=======================================#

function assemble!(operator::Operator, test_functions::Space, trial_functions::Space,
    store, threading::Type{Threading{:gpu}};
    quadstrat=defaultquadstrat, gpu_tiling=(1,1),kwargs...)

    quadstrat = quadstrat(operator, test_functions, trial_functions)

    test_splits = [round(Int,s) for s in range(0, stop=numfunctions(test_functions), length=gpu_tiling[1]+1)]
    trial_splits = [round(Int,s) for s in range(0, stop=numfunctions(trial_functions), length=gpu_tiling[2]+1)]

    println("Gpu tiling: $gpu_tiling")
    @show test_splits
    @show trial_splits

    for i in 1:length(test_splits)-1

        lo_test, hi_test = test_splits[i]+1, test_splits[i+1]
        test_functions_p = subset(test_functions, lo_test:hi_test)

        for j in 1:length(trial_splits)-1

            lo_trial, hi_trial = trial_splits[j]+1, trial_splits[j+1]        
            trial_functions_p = subset(trial_functions, lo_trial:hi_trial)
       
            store1 = BEAST._OffsetStore(store, lo_test-1, lo_trial-1)
            assemblechunk_gpu!(operator, test_functions_p, trial_functions_p, store1, quadstrat=quadstrat)
        end
    end
   
    #TODO: tiling of operator assembly to fit GPU memory constraints.

    #assemblechunk_gpu!(operator, test_functions, trial_functions, store; quadstrat =quadstrat)
  
end

function assemblechunk_gpu!(operator::IntegralOperator, test_functions::Space, trial_functions::Space,store;
    quadstrat=BEAST.defaultquadstrat)

    println("GPU assemble called.")
    test_space = refspace(test_functions)
    trial_space = refspace(trial_functions)

   

    qrule = CompScienceMeshes.trgauss(quadstrat(operator,test_space,trial_space).outer_rule)
    # @show qrule
    q = Array{Tuple{SVector{2,Float64},Float64}}(undef,length(qrule[2]))
    for (i,a) in enumerate(zip(eachcol(qrule[1]), qrule[2]))
        q[i]=a
    end
    #@show q
    quadrule_d = CuArray(q)


    cv_rule = CompScienceMeshes.legendre(quadstrat(operator,test_space,trial_space).sauter_schwab_common_vert, 0.0, 1.0)
    # @show cv_rule
    q = Array{Tuple{Float64,Float64}}(undef,length(cv_rule[2]))
    for (i,a) in enumerate(zip(cv_rule[1], cv_rule[2]))
        q[i]=a
    end
    # @show q
    cvrule_d = CuArray(q)

    test_el_d,test_ad_d,_ = load_assemblydata_gpu(test_functions)
    trial_el_d,trial_ad_d,_ = load_assemblydata_gpu(trial_functions)

    test_domain = CUDA.@allowscalar domain(test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(trial_el_d[1])

    #@show size(test_el_d)
    #@show size(trial_el_d)

    quadstrat_d = CUDA.fill(0,length(test_el_d)*length(trial_el_d), 4)
    numpairs = zeros(Int,4)
    #println("Singularity detection")
    gpu_singularitydetection!(quadstrat_d,numpairs,test_el_d,trial_el_d)

    @show numpairs
    
    zlocal_d = CUDA.fill(zeros(SMatrix{ numfunctions(test_space,test_domain), numfunctions(trial_space,trial_domain),ComplexF64}),length(test_el_d)*length(trial_el_d))
  
    #println("Double num integrals")
    kernel_config!(gpu_momintegral_doublenum!,zlocal_d,operator,numpairs[1],quadstrat_d[:,1],test_el_d,trial_el_d,
                        test_space,trial_space,quadrule_d)

    #println("Common vertex integrals")
    strategy = CommonVertex(cvrule_d)
    kernel_config!(gpu_momintegral_sauterschwab!,zlocal_d,operator,numpairs[2],quadstrat_d[:,2],test_el_d,trial_el_d,
                        test_space,trial_space,strategy)

    #println("Common edge integrals")
    strategy = CommonEdge(cvrule_d)
    kernel_config!(gpu_momintegral_sauterschwab!,zlocal_d,operator,numpairs[3],quadstrat_d[:,3],test_el_d,trial_el_d,
                        test_space,trial_space,strategy)

    #println("Common face integrals")
    strategy = CommonFace(cvrule_d)
    kernel_config!(gpu_momintegral_sauterschwab!,zlocal_d,operator,numpairs[4],quadstrat_d[:,4],test_el_d,trial_el_d,
                        test_space,trial_space,strategy)

    #@show CUDA.@allowscalar zlocal_d[end]
    
    matrix_d = CUDA.fill(0.0 + 0.0im, numfunctions(test_functions), numfunctions(trial_functions))
    println("Building global matrix")
    kernel_config!(gpu_build_matrix!,matrix_d, zlocal_d, test_ad_d, trial_ad_d,test_el_d,trial_el_d)
  
    matrix = Array(matrix_d)
   
    println("Store called")
    for j in 1:size(matrix,2)
        for i in 1:size(matrix,1)
            store(matrix[i,j],i,j)   
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
