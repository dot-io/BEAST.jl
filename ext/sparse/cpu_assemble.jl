


function l2g_maps!(ad,nfunctions)
    glb_dofs = zeros(Int,nfunctions)

    ax = axes(ad.data)
    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                dof = ad.data[i,j,k][1]
                if dof > 0
                    glb_dofs[dof] = 1
                end
            end
        end
    end

    tally = accumulate(+,glb_dofs)

    l2g_map = zeros(Int,tally[end])
    for (i,flag) in enumerate(glb_dofs)
        if flag > 0
            l2g_map[tally[i]] = i
        end
    end
    
    localmap = Dict{Int,Int}()
    for (i,dof) in enumerate(l2g_map)
        localmap[dof] = i
    end

    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                dof = ad.data[i,j,k][1]
                if haskey(localmap, dof)
                    ad.data[i,j,k] = (localmap[dof], ad.data[i,j,k][2])
                end
            end
        end
    end

    return l2g_map
end


function assemblydata_quaddata_worker(buffer::Channel,operator,functions::Space, cellids, quadrule)

    geo = geometry(functions)
    subgeo = CompScienceMeshes.SubMesh(geo, cellids)
    subfuncs = restrict(functions, subgeo)

    el, ad, cells = BEAST.assemblydata(subfuncs)
    l2g_map = l2g_maps!(ad,numfunctions(subfuncs))

    assemblydata = (subfuncs, (el, ad, cells, l2g_map))
    quaddata = quadpoints(refspace(subfuncs),  el,  (quadrule,))

    put!(buffer,(assemblydata,quaddata))
end


function assemble_primer(operator::Operator, functions::Space, quadrule,tiling::Tiling)
    geo = geometry(functions)
    splits = split(numcells(geo), tiling)


    chart_t = typeof(chart(geo, first(geo)))
    assemblydata_t = Tuple{Space, Tuple{Vector{chart_t}, BEAST.AssemblyData{Float64}, Vector{Int}, Vector{Int}}}
    shape_t = shapetype(refspace(functions))
    quaddata_t = Matrix{Vector{@NamedTuple{weight::Float64, point::CompScienceMeshes.MeshPointNM{Float64, chart_t, 2, 3}, value::SVector{3, shape_t}}}}
   

    k=0
    buffer = Channel{Tuple{assemblydata_t, quaddata_t}}(2)
    for i in eachindex(splits)
          errormonitor(Threads.@spawn assemblydata_quaddata_worker(buffer,operator,functions,splits[i], quadrule))
      
         k+=1
    end
    ad_qd = Vector{Tuple{assemblydata_t, quaddata_t}}()
    while k>0
        push!(ad_qd, take!(buffer))
        k -= 1
    end

    return ad_qd
end


function assemblechunk_body!(biop,
        test_space, test_elements, test_assembly_data, test_cells,
        trial_space, trial_elements, trial_assembly_data, trial_cells,
        qd, zlocal, store; quadstrat)

    test_shapes = refspace(test_space)
    trial_shapes = refspace(trial_space)

    n = length(test_elements)
    m = length(trial_elements)
    for p in 1:n
        for q in 1:m

            fill!(zlocal, 0)
            qrule = quadrule(biop, test_shapes, trial_shapes, p, test_elements[p], q, trial_elements[q], qd, quadstrat)
            BEAST.momintegrals!(zlocal, biop,
                test_space,  test_cells[p], test_elements[p],
                trial_space, trial_cells[q], trial_elements[q], qrule)
            I = length(test_assembly_data[p])
            J = length(trial_assembly_data[q])
            for j in 1 : J
                QQ = @view trial_assembly_data.data[:,j,q]
                for i in 1 : I
                    zij = zlocal[i,j]
                    PP = @view test_assembly_data.data[:,i,p]
                    for (n,b) in QQ
                        n < 1 && break
                        zb = zij*b
                        for (m,a) in PP
                            m < 1 && break
                            store(a*zb, m, n)
                        end
                    end
                end
            end 
        end
    end
end

function assemble_body_worker(buffer::Channel,operator, test_assebmeldata,trial_assemblydata,qd, quadstrat)

    (test_functions,(test_els, test_ad, test_cells, test_l2g_map)) = test_assebmeldata
    (trial_functions,(trial_els, trial_ad, trial_cells, trial_l2g_map)) = trial_assemblydata
    test_shapes = refspace(test_functions)
    trial_shapes = refspace(trial_functions)

    test_dom = domain(first(test_els))
    trial_dom = domain(first(trial_els))

    num_testshapes = numfunctions(test_shapes, test_dom)
    num_trialshapes = numfunctions(trial_shapes, trial_dom)


    N = length(test_l2g_map)
    M = length(trial_l2g_map)

    Z = zeros(scalartype(operator,test_functions,trial_functions), N, M)
    zlocal = zero(MMatrix{ num_testshapes, num_trialshapes,eltype(Z)})
    localstore(v,m,n) = (Z[m,n] += v)

    assemblechunk_body!(operator,
                test_functions, test_els, test_ad, test_cells,
                trial_functions, trial_els, trial_ad, trial_cells,
                qd,zlocal, localstore; quadstrat=quadstrat)

    put!(buffer,(Z,test_l2g_map,trial_l2g_map) )
end



function assemble!(operator::Operator, test_functions::Space, trial_functions::Space,
    store, threading::Type{Threading{:cellsplitting}}; 
    quadstrat=defaultquadstrat, tilingstrat=TilingStrategy(WorksizeTiling(128),WorksizeTiling(128)),kwargs...)

    quadstrat = quadstrat(operator, test_functions, trial_functions)

  

    test_ad_qd = assemble_primer(operator, test_functions, quadstrat.outer_rule, tilingstrat[1])

    trial_ad_qd = assemble_primer(operator, trial_functions, quadstrat.inner_rule, tilingstrat[2])



    T = scalartype(operator,test_functions,trial_functions)
    
    leg = (
      convert.(NTuple{2,T},BEAST._legendre(quadstrat.sauter_schwab_common_vert,0,1)),
      convert.(NTuple{2,T},BEAST._legendre(quadstrat.sauter_schwab_common_edge,0,1)),
      convert.(NTuple{2,T},BEAST._legendre(quadstrat.sauter_schwab_common_face,0,1)),
      convert.(NTuple{2,T},BEAST._legendre(quadstrat.sauter_schwab_common_tetr,0,1)),
      )
    


    Zbuffer = Channel{Tuple{Matrix{T},Vector{Int},Vector{Int}}}(2)

    k = 0
    for (test_ad,test_qd) in test_ad_qd
        for (trial_ad,trial_qd) in trial_ad_qd
            qd = (tpoints=test_qd, bpoints=trial_qd, gausslegendre=leg)
            errormonitor(Threads.@spawn assemble_body_worker(Zbuffer,operator,test_ad,trial_ad,qd,quadstrat))
            k += 1
        end
    end 
    pbar = BEAST.progressbar(k, true)
    maxk = k
    while k>0
        (Z,test_l2g_map,trial_l2g_map) = take!(Zbuffer)

        for i in eachindex(test_l2g_map)
            m = test_l2g_map[i]
            for j in eachindex(trial_l2g_map)
                n = trial_l2g_map[j]
                store(Z[i,j],m,n) 
            end
        end

        k -= 1
        update!(pbar, maxk-k)
    end
    finish!(pbar)
end






