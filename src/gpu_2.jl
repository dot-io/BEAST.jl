using BEAST
using CUDA

function filter_and_copy_dev(test_ids, trial_ids)

    for (i, m) in enumerate(test_ids)
        test_id_in_blk[m] = i
    end
    for (i, m) in enumerate(trial_ids)
        trial_id_in_blk[m] = i
    end

    for m in test_ids, sh in tfs.fns[m]
        push!(active_test_el_ids, Int32(sh.cellid));
    end
    for m in trial_ids, sh in bfs.fns[m]
        push!(active_trial_el_ids, Int32(sh.cellid));
    end

    active_test_el_ids = unique!(sort!(active_test_el_ids))
    active_trial_el_ids = unique!(sort!(active_trial_el_ids))

    (isempty(active_test_el_ids) || isempty(active_trial_el_ids)) && return


    # If I remove these assertions I don't have to pass the CPU arrays anymore. This doesnt cost anything however
    @assert maximum(active_test_el_ids) <= length(test_elements)
    @assert maximum(active_trial_el_ids) <= length(bsis_elements)

    # Transfer active element id lists to GPU
    active_test_ids_dev = CUDA.cu(active_test_el_ids)
    active_trial_ids_dev = CUDA.cu(active_trial_el_ids)

    return active_test_ids_dev, active_trial_ids_dev
end

function assembleblock_body_gpu!(
    biop::IntegralOperator,
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
    store;
    quadstrat,
)

    test_id_in_blk = Dict{Int,Int}()
    trial_id_in_blk = Dict{Int,Int}()

    for (i, m) in enumerate(test_ids)
        test_id_in_blk[m] = i
    end
    for (i, m) in enumerate(trial_ids)
        trial_id_in_blk[m] = i
    end

    test_id_dev, trial_id_dev = filter_and_copy_dev(test_ids, trial_ids)

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

    kernel = @cuda launch=false momintegrals_kernel!(
        # zlocals_all_dev,
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
        ZT,
        Int32(num_tshapes * num_bshapes),
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
        M,
        N,
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
    zlocals_all_host = Array(zlocals_all_dev)

    for k = 1:num_pairs
        # Decode the pair index back to element ids
        p_local = div(k - 1, num_trial) + 1
        q_local = mod(k - 1, num_trial) + 1
        p = active_test_el_ids[p_local]
        q = active_trial_el_ids[q_local]

        for j = 1:num_bshapes
            for i = 1:num_tshapes
                zval = zlocals_all_host[i, j, k]
                zval == zero(zval) && continue

                for (n, b) in trial_assembly_data[q, j]
                    n′ = get(trial_id_in_blk, n, 0)
                    n′ == 0 && continue
                    for (m, a) in test_assembly_data[p, i]
                        m′ = get(test_id_in_blk, m, 0)
                        m′ == 0 && continue
                        store(a * zval * b, m′, n′)
                    end
                end
            end
        end
    end
end

function momintegrals!(
    op,
    output::CuMatrix{T},
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
