# Gathering-based implementation 3: shared memory reduction of integrand matrix

"""
    assembleblock_gpu!(biop, tfs, bfs, store; test_ids, trial_ids)

Call a primer routine, followed by the block assembly body.
See also [`assembleblock_primer_gpu`](@ref) and
[`assembleblock_body_gpu!`](@ref).
"""
function assembleblock_gpu!(
    biop, tfs, bfs, store;
    test_ids=collect(keys(tfs.fns)),
    trial_ids=collect(keys(bfs.fns)),
)

    ctx::NamedTuple = assembleblock_primer_gpu(biop, tfs, bfs, loop_order=:gather)

    assembleblock_body_gpu!(
        biop,
        tfs, test_ids, ctx.test_elements_dev, ctx.tad_gpu,
        bfs, trial_ids, ctx.trial_elements_dev, ctx.bad_gpu,
        ctx.quaddata_gpu, ctx.zlocals,
        ctx.num_tshapes, ctx.num_bshapes,
        store;
        quadstrat=ctx.qs,
        ctx.test_shapes, ctx.trial_shapes,
    )
end


"""
    assembleblock_body_gpu!(biop, tfs, bfs; loop_order=:gather)

Launch a user-chosen CUDA kernel that computes the system matrix for test
basis `tfs` and trial basis `bfs` with bilinear operator `biop`.

See also [`tile_gather_cooperative_kernel!`](@ref) for the kernel implementation.
"""
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
    test_shapes, trial_shapes,
)

    # Local-to-global maps: local_index -> global_dof_id
    # The kernel needs to convert its per-thread local index into the
    # global DOF id used by InvAssemblyData offsets/lengths.
    test_l2g = CUDA.cu(Int32.(test_ids))
    trial_l2g = CUDA.cu(Int32.(trial_ids))

    M_tile = Int32(16)   # tile size for test functions (rows)
    N_tile = Int32(16)   # tile size for trial functions (cols)
    pair_flat::CuArray{Tuple{Int32,Int32}}, pair_off::CuArray{Int32} = build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)

    n_tiles_m = Int32(cld(length(test_ids), M_tile))
    n_tiles_n = Int32(cld(length(trial_ids), N_tile))

    M_total = Int32(length(test_ids))
    N_total = Int32(length(trial_ids))

    @cuda threads = (M_tile, N_tile) blocks = (n_tiles_m, n_tiles_n) tile_gather_cooperative_kernel!(
        store.data,
        biop,
        test_shapes,
        trial_shapes,
        test_elements_dev,
        bsis_elements_dev,
        test_assembly_dev.flat, test_assembly_dev.offsets, test_assembly_dev.lengths,
        trial_assembly_dev.flat, trial_assembly_dev.offsets, trial_assembly_dev.lengths,
        test_l2g,
        trial_l2g,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        pair_flat, pair_off,
        M_total, N_total, n_tiles_m,
        Int32(num_tshapes), Int32(num_bshapes),
        Val(M_tile), Val(N_tile),
    )
    return


end

"""
    build_tile_pairs(test_ids, trial_ids, tfs, bfs, M_tile, N_tile)
    -> (pair_flat, pair_off)

For each test function and each trial function of the 'tile', retrieve elements
that belong to those respective test and trial functions and store all element
pairs that have contributions to a 'tile'.
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

"""
    tile_gather_cooperative_kernel!(...)

Arguments:

"""
function tile_gather_cooperative_kernel!(
    output, op, test_shapes, trial_shapes,
    test_elements, bsis_elements,
    inv_tad_flat::CuDeviceVector{Tuple{Int32,Int32,T},1}, inv_tad_offsets, inv_tad_lengths,
    inv_bad_flat::CuDeviceVector{Tuple{Int32,Int32,T},1}, inv_bad_offsets, inv_bad_lengths,
    test_l2g, trial_l2g,
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    pair_flat, pair_off,
    M_block::Int32, N_block::Int32,
    n_tiles_m::Int32,
    num_tshapes::Int32, num_bshapes::Int32,
    ::Val{M_max}, ::Val{N_max},          # static for shared-memory sizing
) where {T,M_max,N_max}

    # split integrand matrix into real and imaginary parts so that
    # atomic_add!(...) can be used properly
    F = real(T)
    Z_re = @cuStaticSharedMem(F, (M_max, N_max))
    Z_im = @cuStaticSharedMem(F, (M_max, N_max))

    tx = threadIdx().x
    ty = threadIdx().y
    tid_linear = (ty - Int32(1)) * blockDim().x + tx #TODO check if req
    nthreads = blockDim().x * blockDim().y

    tile_idx = blockIdx().x + (blockIdx().y - Int32(1)) * n_tiles_m
    pair_lo = pair_off[tile_idx] # first element pair of the 'tile'
    # last element pair of the 'tile'
    pair_hi = pair_off[tile_idx+Int32(1)] - Int32(1)
    `                                               `
    # the test DoF to be used by the thread
    m_local = (blockIdx().x - Int32(1)) * blockDim().x + tx
    # the trial DoF to be used by the thread
    n_local = (blockIdx().y - Int32(1)) * blockDim().y + ty

    # indicates whether the test and trial DoF are within the block
    # We cannot simply tell these threads to return, as they compute meaningful
    # parts of the integrands
    in_bounds = (m_local <= M_block) & (n_local <= N_block)

    # get global DoF indices
    @inbounds m_global = in_bounds ? test_l2g[m_local] : Int32(0)
    @inbounds n_global = in_bounds ? trial_l2g[n_local] : Int32(0)

    # get offsets and lengths to index the flattened (m) -> (p, i, a) map
    @inbounds t_off = m_global == 0 ? Int32(0) : inv_tad_offsets[m_global]
    @inbounds t_len = m_global == 0 ? Int32(0) : inv_tad_lengths[m_global]
    @inbounds b_off = n_global == 0 ? Int32(0) : inv_bad_offsets[n_global]
    @inbounds b_len = n_global == 0 ? Int32(0) : inv_bad_lengths[n_global]

    # initialize 2 accumulators per thread (one for real one for imaginary part)
    acc_re = zero(F)
    acc_im = zero(F)

    current_pair_index = pair_lo
    while current_pair_index <= pair_hi
        @inbounds (p, q) = pair_flat[current_pair_index]

        # zero the shared integrand matrices (1 element per thread)
        if tx <= Int32(M_max) && ty <= Int32(N_max)
            @inbounds Z_re[tx, ty] = zero(F)
            @inbounds Z_im[tx, ty] = zero(F)
        end
        CUDA.sync_threads() # ensure Z_re, Z_im are set to 0

        # cooperative quadrature, split oi, ii work across all threads in the block
        # this is done in a separate step so that there isn't recomputation
        # of integrands/quadratures per thread
        tcell = test_elements[p]
        bcell = bsis_elements[q]
        # TODO: can this igd be made more efficient?
        igd = Integrand(op, test_shapes, trial_shapes, tcell, bcell)
        oo = tqp_offsets[p]
        olen = tqp_lengths[p]
        ii_off = bqp_offsets[q]
        ilen = bqp_lengths[q]
        total_qp = olen * ilen

        kqp = tid_linear - Int32(1)
        while kqp < total_qp
            oi = div(kqp, ilen)
            ii = mod(kqp, ilen)
            @inbounds womp = tqp_flat[oo+oi]
            @inbounds wimp = bqp_flat[ii_off+ii]
            #TODO: look at igd function
            z1 = igd(womp.point, wimp.point, womp.value, wimp.value)
            jxjy = womp.weight * wimp.weight

            for j_acc in Int32(1):num_bshapes
                for i_acc in Int32(1):num_tshapes
                    val = jxjy * z1[i_acc, j_acc]
                    CUDA.@atomic Z_re[i_acc, j_acc] += real(val)
                    CUDA.@atomic Z_im[i_acc, j_acc] += imag(val)
                end
            end

            kqp += nthreads # in strided fashion for CMA
        end
        sync_threads() # ensure Z_re, Z_im has been computed fully

        # per thread, extract (i, j) for th (p, q) through linear scan of inv_tad / inv_bad
        # Now the threads take on the role of 'gatherers' once again.
        if in_bounds
            ti = Int32(0)
            while ti < t_len
                @inbounds (pp, i, a) = inv_tad_flat[t_off+ti]
                if pp == p
                    bi = Int32(0)
                    while bi < b_len
                        @inbounds (qq, j, b) = inv_bad_flat[b_off+bi]
                        if qq == q
                            @inbounds zr = Z_re[i, j]
                            @inbounds zi = Z_im[i, j]
                            # acc += a * (zr + im*zi) * b  expanded:
                            ab_r = real(a) * real(b) - imag(a) * imag(b)
                            ab_i = real(a) * imag(b) + imag(a) * real(b)
                            acc_re += ab_r * zr - ab_i * zi
                            acc_im += ab_r * zi + ab_i * zr
                        end
                        bi += Int32(1)
                    end
                end
                ti += Int32(1)
            end
        end
        sync_threads()

        current_pair_index += Int32(1)
    end

    if in_bounds
        @inbounds output[m_local, n_local] = complex(acc_re, acc_im)
    end
    return
end
