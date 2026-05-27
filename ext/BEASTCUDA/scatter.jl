using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @inbounds, @atomic


"""
    pair_scatter_kernel!(...)

One thread per element pair (p, q). Computes the full zlocal via
`accumulate_zlocal_ntuple` and atomic-scatters each entry into the output.
"""
function pair_scatter_kernel!(
    output_re::CuDeviceMatrix{F},
    output_im::CuDeviceMatrix{F},
    op, test_shapes, trial_shapes,
    test_elements, trial_elements,
    tad_flat, tad_offsets, tad_lengths,
    bad_flat, bad_offsets, bad_lengths,
    active_test_ids::CuDeviceVector{Int32},
    active_trial_ids::CuDeviceVector{Int32},
    test_id_map::CuDeviceVector{Int32},
    trial_id_map::CuDeviceVector{Int32},
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    num_tshapes::Int32,
    num_bshapes::Int32,
    num_test::Int32,
    num_trial::Int32,
    ::Val{NS},
    ::Val{MS},
) where {F<:AbstractFloat,NS,MS}

    T = Complex{F}

    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total_pairs = num_test * num_trial
    idx > total_pairs && return

    p_local = mod(idx - Int32(1), num_test) + Int32(1)
    q_local = div(idx - Int32(1), num_test) + Int32(1)

    @inbounds p = active_test_ids[p_local]
    @inbounds q = active_trial_ids[q_local]

    @inbounds tcell = test_elements[p]
    @inbounds bcell = trial_elements[q]

    @inbounds t_off = tqp_offsets[p]
    @inbounds t_len = tqp_lengths[p]
    @inbounds b_off = bqp_offsets[q]
    @inbounds b_len = bqp_lengths[q]
    #zlocal in regs
    z = accumulate_zlocal_ntuple(T, op, test_shapes, trial_shapes,
        tcell, bcell,
        tqp_flat, t_off, t_len,
        bqp_flat, b_off, b_len,
        Val(NS), Val(MS))

    # Outer loop over i: hoist tad_offsets[p,i] and tad_lengths[p,i] here
    # since they are independent of j.
    @inbounds for i in Int32(1):Int32(NS)
        t_ad_off = tad_offsets[p, i]
        t_ad_len = tad_lengths[p, i]
        for j in Int32(1):Int32(MS)
            flat_idx = (j - Int32(1)) * Int32(NS) + i
            zval = z[flat_idx]

            b_ad_off = bad_offsets[q, j]
            b_ad_len = bad_lengths[q, j]

            ti = Int32(0)
            while ti < t_ad_len
                (m_global, a_coeff) = tad_flat[t_ad_off+ti]
                m_local = test_id_map[m_global]

                if m_local != Int32(0)
                    bi = Int32(0)
                    while bi < b_ad_len
                        (n_global, b_coeff) = bad_flat[b_ad_off+bi]
                        n_local = trial_id_map[n_global]

                        if n_local != Int32(0)
                            contribution = a_coeff * zval * b_coeff
                            CUDA.@atomic output_re[m_local, n_local] += real(contribution)
                            CUDA.@atomic output_im[m_local, n_local] += imag(contribution)
                        end

                        bi += Int32(1)
                    end
                end

                ti += Int32(1)
            end
        end
    end

    return nothing
end


"""
    gpu_pair_scatter!(output, biop, test_shapes, trial_shapes,
        test_elements, trial_elements, tad, bad,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        quaddata_gpu, num_tshapes, num_bshapes)
"""
function gpu_pair_scatter!(
    output::CuMatrix{T},
    biop, test_shapes, trial_shapes,
    test_elements, trial_elements,
    tad::FlattenedAssemblyData{T},
    bad::FlattenedAssemblyData{T},
    active_test_ids::CuVector{Int32},
    active_trial_ids::CuVector{Int32},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    quaddata_gpu,
    num_tshapes::Int,
    num_bshapes::Int,
) where {T}

    F = real(T)
    num_test = length(active_test_ids)
    num_trial = length(active_trial_ids)
    total_pairs = Int(num_test) * Int(num_trial)
    output_rows = size(output, 1)
    output_cols = size(output, 2)

    output_re = CUDA.zeros(F, output_rows, output_cols)
    output_im = CUDA.zeros(F, output_rows, output_cols)

    kernel = @cuda launch = false pair_scatter_kernel!(
        output_re, output_im,
        biop, test_shapes, trial_shapes,
        test_elements, trial_elements,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Val(num_tshapes), Val(num_bshapes),
    )
    config = launch_configuration(kernel.fun)
    threads = min(total_pairs, config.threads)
    blocks = cld(total_pairs, threads)

    kernel(
        output_re, output_im,
        biop, test_shapes, trial_shapes,
        test_elements, trial_elements,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Val(num_tshapes), Val(num_bshapes);
        threads, blocks,
    )

    output .= complex.(output_re, output_im)

    CUDA.synchronize()
    return output
end
