# ===========================================================================
# Hybrid Global — Biphasic kernel with Z in global memory
# ===========================================================================
#
# Two-phase approach:
#   Phase 1: Compute all integrand values z_{ij}^{(p,q)} and scatter them
#            into a dof-renamed 4D intermediate matrix Z_padded stored in
#            global memory. Layout: Z_padded[m, k, l, n] where m (test dof)
#            varies fastest — chosen so that Phase 2's threads (with m =
#            threadIdx().x) read stride-1 from Z_padded.
#   Phase 2: One thread per output entry (m, n). Gather from Z_padded using
#            zero-padded coefficient arrays. Branchless inner loops.
#            Write output[m, n] directly — no atomics.
#
# Key properties:
#   • Phase 1: one thread per element pair (p, q), grid-stride loop.
#     Uses FlattenedAssemblyDataWithK (forward map with contributor index k)
#     to scatter into Z_padded. No atomics needed because each (m,k,n,l)
#     maps to a unique (p,i,q,j).
#   • Phase 2: one thread per (m, n) reads Z_padded[m, k, l, n] with
#     zero-padded coefficients. Fixed trip counts (K_max_test, K_max_trial
#     as Val parameters) enable full loop unrolling and eliminate warp
#     divergence in the inner loops.
#
# Memory layout note:
#   Z_padded sized (M_block, K_max_t, K_max_b, N_block) gives stride-1
#   coalesced access in Phase 2: thread (m,n) reads Z_padded[m, k, l, n] and
#   threadIdx().x → m varies fastest across a warp, so adjacent threads read
#   consecutive memory addresses. The previous layout [N,K_b,K_t,M] had
#   stride N×K_b×K_t between adjacent threads — devastating for bandwidth.
#
# Trade-offs vs. hybrid_shared.jl:
#   + Simpler: no tile pair lists, no shared memory sizing constraints
#   + No barrier overhead (two separate kernels)
#   + Works for any basis (tile size not limited by shared memory)
#   - Z_padded lives in global memory → extra DRAM traffic
#   - Z_padded peak memory is M*K_t*K_b*N*sizeof(T) — scales as O(n²);
#     for n=10K with K_max=2 that's 6.4 GB. Future improvement: tile the
#     output into M_TILE×N_TILE blocks to bound peak memory.
#   - Two kernel launches (Phase 1 + Phase 2) vs. one fused kernel

using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @inbounds, synchronize


"""
    hybrid_global_phase1_kernel!(Z_padded, op, ...)

One thread per element pair (p, q). Computes the full zlocal NTuple via
`accumulate_zlocal_ntuple` and scatters each entry into Z_padded[m, k, l, n]
using the forward assembly data with contributor indices.

No atomics: each (m, k, n, l) entry maps to a unique (p, i, q, j) so exactly
one thread writes each Z_padded slot.
"""
function hybrid_global_phase1_kernel!(
    Z_padded::CuDeviceArray{T,4},     # [M_block, K_max_t, K_max_b, N_block]
    op, test_shapes, trial_shapes,
    test_elements, trial_elements,
    active_test_ids::CuDeviceVector{Int32},
    active_trial_ids::CuDeviceVector{Int32},
    fwd_tad_flat, fwd_tad_offsets, fwd_tad_lengths,
    fwd_bad_flat, fwd_bad_offsets, fwd_bad_lengths,
    test_id_map::CuDeviceVector{Int32},
    trial_id_map::CuDeviceVector{Int32},
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    num_test::Int32, num_trial::Int32,
    M_block::Int32, N_block::Int32,
    ::Val{NS}, ::Val{MS},
    ::Val{K_max_t}, ::Val{K_max_b},
) where {T,NS,MS,K_max_t,K_max_b}

    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total_pairs = num_test * num_trial
    stride = blockDim().x * gridDim().x

    linear_pair = idx

    while linear_pair <= total_pairs
        p_local = mod(linear_pair - Int32(1), num_test) + Int32(1)
        q_local = div(linear_pair - Int32(1), num_test) + Int32(1)

        @inbounds p = active_test_ids[p_local]
        @inbounds q = active_trial_ids[q_local]

        @inbounds tcell = test_elements[p]
        @inbounds bcell = trial_elements[q]

        @inbounds t_off = tqp_offsets[p]
        @inbounds t_len = tqp_lengths[p]
        @inbounds b_off = bqp_offsets[q]
        @inbounds b_len = bqp_lengths[q]

        # Full zlocal NTuple via the shared quadrature helper (inlined)
        z = accumulate_zlocal_ntuple(T, op, test_shapes, trial_shapes,
            tcell, bcell,
            tqp_flat, t_off, t_len,
            bqp_flat, b_off, b_len,
            Val(NS), Val(MS))

        # Scatter zlocal into Z_padded[m, k, l, n]
        @inbounds for i in Int32(1):Int32(NS)
            t_ad_off = fwd_tad_offsets[p, i]
            t_ad_len = fwd_tad_lengths[p, i]
            for j in Int32(1):Int32(MS)
                flat_idx = (j - Int32(1)) * Int32(NS) + i
                zval = z[flat_idx]

                b_ad_off = fwd_bad_offsets[q, j]
                b_ad_len = fwd_bad_lengths[q, j]

                ti = Int32(0)
                while ti < t_ad_len
                    (m_global, k, a_coeff) = fwd_tad_flat[t_ad_off+ti]
                    m_local = test_id_map[m_global]

                    if m_local != Int32(0)
                        bi = Int32(0)
                        while bi < b_ad_len
                            (n_global, l, b_coeff) = fwd_bad_flat[b_ad_off+bi]
                            n_local = trial_id_map[n_global]

                            if n_local != Int32(0)
                                # No atomic: unique (m,k,n,l) per (p,i,q,j)
                                Z_padded[m_local, k, l, n_local] = zval
                            end

                            bi += Int32(1)
                        end
                    end

                    ti += Int32(1)
                end
            end
        end

        linear_pair += stride
    end

    return nothing
end


"""
    hybrid_global_phase2_kernel!(output, Z_padded, coeff_test_padded,
        coeff_trial_padded, M_block, N_block, ::Val{K_max_t}, ::Val{K_max_b})

One thread per output entry (m, n). Reads Z_padded[m, k, l, n] with stride-1
access patterns (m fastest in memory; adjacent threads adjacent m), combines
with zero-padded coefficients in fully unrolled loops, writes output[m, n].
"""
function hybrid_global_phase2_kernel!(
    output::CuDeviceMatrix{T},
    Z_padded::CuDeviceArray{T,4},
    coeff_test_padded::CuDeviceMatrix{CT},   # [num_tfs, K_max_t]  (global DOF index)
    coeff_trial_padded::CuDeviceMatrix{CB},  # [num_bfs, K_max_b]  (global DOF index)
    test_dof_ids::CuDeviceVector{Int32},     # length M_block; local row m → global DOF
    trial_dof_ids::CuDeviceVector{Int32},    # length N_block; local col n → global DOF
    M_block::Int32, N_block::Int32,
    ::Val{K_max_t}, ::Val{K_max_b},
) where {T,CT,CB,K_max_t,K_max_b}

    m = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    n = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    (m > M_block || n > N_block) && return

    # coeff_*_padded are indexed by global DOF; Z_padded is indexed by local
    # (m, n). Look up the corresponding global DOFs for this thread's local
    # output entry so that arbitrary index subsets (e.g. those emitted by
    # ACA pivoting) read the correct coefficients.
    @inbounds m_global = test_dof_ids[m]
    @inbounds n_global = trial_dof_ids[n]

    acc = zero(T)

    @inbounds for k in Int32(1):Int32(K_max_t)
        a = coeff_test_padded[m_global, k]
        for l in Int32(1):Int32(K_max_b)
            b = coeff_trial_padded[n_global, l]
            acc += a * Z_padded[m, k, l, n] * b
        end
    end

    @inbounds output[m, n] = acc
    return nothing
end


"""
    gpu_hybrid_global!(output, biop, test_shapes, trial_shapes, ...)

Launch the two-phase hybrid kernel with Z_padded in global memory.
"""
function gpu_hybrid_global!(
    output::CuMatrix{T},
    biop, test_shapes, trial_shapes,
    test_elements_dev, trial_elements_dev,
    active_test_ids::CuVector{Int32},
    active_trial_ids::CuVector{Int32},
    had::HybridAssemblyData{T},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    quaddata_gpu,
    num_tshapes::Int, num_bshapes::Int,
    test_dof_ids::CuVector{Int32},
    trial_dof_ids::CuVector{Int32},
) where {T}

    M_block = Int32(size(output, 1))
    N_block = Int32(size(output, 2))
    K_max_t = Int(had.K_max_test)
    K_max_b = Int(had.K_max_trial)

    # Layout: (M_block, K_max_t, K_max_b, N_block) — m fastest → stride-1
    # for Phase 2 reads where threadIdx().x = m varies across warps.
    Z_padded = CUDA.zeros(T, Int(M_block), K_max_t, K_max_b, Int(N_block))

    # ── Phase 1 ──
    num_test = Int32(length(active_test_ids))
    num_trial = Int32(length(active_trial_ids))
    total_pairs = Int(num_test) * Int(num_trial)

    kernel1 = @cuda launch = false hybrid_global_phase1_kernel!(
        Z_padded, biop, test_shapes, trial_shapes,
        test_elements_dev, trial_elements_dev,
        active_test_ids, active_trial_ids,
        had.fwd_tad.flat, had.fwd_tad.offsets, had.fwd_tad.lengths,
        had.fwd_bad.flat, had.fwd_bad.offsets, had.fwd_bad.lengths,
        test_id_map, trial_id_map,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        num_test, num_trial,
        M_block, N_block,
        Val(num_tshapes), Val(num_bshapes),
        Val(K_max_t), Val(K_max_b),
    )
    config1 = launch_configuration(kernel1.fun)
    threads1 = min(total_pairs, config1.threads)
    blocks1 = cld(total_pairs, threads1)

    kernel1(
        Z_padded, biop, test_shapes, trial_shapes,
        test_elements_dev, trial_elements_dev,
        active_test_ids, active_trial_ids,
        had.fwd_tad.flat, had.fwd_tad.offsets, had.fwd_tad.lengths,
        had.fwd_bad.flat, had.fwd_bad.offsets, had.fwd_bad.lengths,
        test_id_map, trial_id_map,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        num_test, num_trial,
        M_block, N_block,
        Val(num_tshapes), Val(num_bshapes),
        Val(K_max_t), Val(K_max_b);
        threads=threads1, blocks=blocks1,
    )

    CUDA.synchronize()

    # ── Phase 2 ──
    tile = Int(TILE_SIZE)
    threads2 = (tile, tile)
    blocks2 = (cld(Int(M_block), tile), cld(Int(N_block), tile))

    @cuda threads = threads2 blocks = blocks2 hybrid_global_phase2_kernel!(
        output, Z_padded,
        had.coeff_test_padded, had.coeff_trial_padded,
        test_dof_ids, trial_dof_ids,
        M_block, N_block,
        Val(K_max_t), Val(K_max_b),
    )

    CUDA.synchronize()
    return output
end
