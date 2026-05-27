# Scattering-based implementation — direct variant (v1.0)
#
#   scatter_kernel_direct!   — atomic add directly into global output
#
# Each thread owns one zlocal entry (or strides over several) and expands it
# into contributions to the output matrix via the AssemblyData maps.
# Atomic accumulation happens directly in global memory.
#
# The tiled variant (v1.5) lives in gpu_v1.5.jl.

using CompScienceMeshes: MeshPointNM, Simplex, SVector
using CUDA: CuVector, CuMatrix, CuArray, @cuda, @cuStaticSharedMem, @inbounds,
    @atomic, synchronize


# ===========================================================================
# v1.0 — Direct global-memory scatter
# ===========================================================================
#
# Each thread processes one zlocal[i, j, pair_idx] entry (grid-stride loop)
# and atomically adds its contribution directly to the global output matrix.
#
# Pros:
#   • Simple, easy to verify correct
#   • Each zlocal is read exactly once → minimal memory traffic
#   • For RT basis (fan-out ≈ 1) there is negligible atomic contention
#
# Cons:
#   • Every contribution hits global memory via atomic → higher latency
#     per atomic than shared-memory atomic
#   • If many zlocals map to the same output entry the atomics serialise
#
# For typical BEM problems with RT basis each zlocal contributes to at most
# one output entry, so contention is near-zero and v1.0 is hard to beat.

"""
    scatter_kernel_direct!(output_re, output_im, zlocals_all, tad_flat, tad_offsets, tad_lengths,
                           bad_flat, bad_offsets, bad_lengths,
                           active_test_ids, active_trial_ids,
                           test_id_map, trial_id_map,
                           num_tshapes, num_bshapes,
                           num_test, num_trial,
                           output_rows, output_cols)

Direct scatter kernel: each thread expands one zlocal entry and atomically
adds its contributions into the split real/imag output buffers.
"""
function scatter_kernel_direct!(
    output_re::CuDeviceMatrix{F},
    output_im::CuDeviceMatrix{F},
    zlocals_all::CuDeviceArray{T,3},
    tad_flat, tad_offsets, tad_lengths,
    bad_flat, bad_offsets, bad_lengths,
    active_test_ids::CuDeviceVector{Int32},
    active_trial_ids::CuDeviceVector{Int32},
    test_id_map::CuDeviceVector{Int32},
    trial_id_map::CuDeviceVector{Int32},
    num_tshapes::Int32,
    num_bshapes::Int32,
    num_test::Int32,
    num_trial::Int32,
    output_rows::Int32,
    output_cols::Int32,
) where {T,F<:AbstractFloat}

    total_pairs = num_test * num_trial
    MN = num_tshapes * num_bshapes
    total_zlocals = total_pairs * MN

    # Grid-stride loop: each thread walks zlocals with stride = total threads
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    linear_zlocal = tid

    while linear_zlocal <= total_zlocals
        # Decode linear index → (pair_idx, i, j)
        pair_idx = div(linear_zlocal - Int32(1), MN) + Int32(1)
        rem_ij = mod(linear_zlocal - Int32(1), MN)
        i = mod(rem_ij, num_tshapes) + Int32(1)
        j = div(rem_ij, num_tshapes) + Int32(1)

        # Decode pair_idx → (p_local, q_local) → element ids (p, q)
        # Must match momintegrals! encoding: idx = p_local + (q_local-1)*num_test
        p_local = mod(pair_idx - Int32(1), num_test) + Int32(1)
        q_local = div(pair_idx - Int32(1), num_test) + Int32(1)

        @inbounds p = active_test_ids[p_local]
        @inbounds q = active_trial_ids[q_local]

        @inbounds zval = zlocals_all[i, j, pair_idx]

        if zval != zero(T)
            t_off = tad_offsets[p, i]
            t_len = tad_lengths[p, i]
            b_off = bad_offsets[q, j]
            b_len = bad_lengths[q, j]

            ti = Int32(0)
            while ti < t_len
                @inbounds (m_global, a_coeff) = tad_flat[t_off+ti]
                @inbounds m_local = test_id_map[m_global]

                if m_local != Int32(0)
                    bi = Int32(0)
                    while bi < b_len
                        @inbounds (n_global, b_coeff) = bad_flat[b_off+bi]
                        @inbounds n_local = trial_id_map[n_global]

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

        linear_zlocal += stride
    end

    return nothing
end



# ===========================================================================
# High-level launch wrapper
# ===========================================================================

"""
    gpu_scatter!(output, zlocals_all, tad, bad, active_test_ids, active_trial_ids,
                 test_id_map, trial_id_map, num_tshapes, num_bshapes;
                 variant=:direct)

High-level interface for GPU scatter operation.

# Arguments
- `output`:       output matrix block (CuMatrix{T})
- `zlocals_all`:  local integrals (CuArray{T,3}, size num_tshapes × num_bshapes × num_pairs)
- `tad`, `bad`:   FlattenedAssemblyData{T} for test/trial
- `active_test_ids`, `active_trial_ids`: active element IDs
- `test_id_map`, `trial_id_map`: global → local-in-block index mapping
- `num_tshapes`, `num_bshapes`: number of local shape functions

# Keyword
- `variant::Symbol`: `:direct` (default, v1.0) or `:tiled` (v1.5)

Use `:direct` for low-contention problems (typical for RT basis).
Use `:tiled` only if profiling shows high global-memory atomic contention.
"""
function gpu_scatter!(
    output::CuMatrix{T},
    zlocals_all::CuArray{T,3},
    tad::FlattenedAssemblyData{T},
    bad::FlattenedAssemblyData{T},
    active_test_ids::CuVector{Int32},
    active_trial_ids::CuVector{Int32},
    test_id_map::CuVector{Int32},
    trial_id_map::CuVector{Int32},
    num_tshapes::Int,
    num_bshapes::Int;
    variant::Symbol=:direct,
) where {T}

    F = real(T)
    num_test = length(active_test_ids)
    num_trial = length(active_trial_ids)
    output_rows = size(output, 1)
    output_cols = size(output, 2)

    # Allocate split real/imag output buffers for atomic compatibility
    # (CUDA does not support atomicAdd on ComplexF64)
    output_re = CUDA.zeros(F, output_rows, output_cols)
    output_im = CUDA.zeros(F, output_rows, output_cols)

    if variant === :direct
        _launch_scatter_direct!(
            output_re, output_im, zlocals_all, tad, bad,
            active_test_ids, active_trial_ids,
            test_id_map, trial_id_map,
            num_tshapes, num_bshapes,
            num_test, num_trial,
            output_rows, output_cols,
        )
    elseif variant === :tiled
        _launch_scatter_tiled!(
            output_re, output_im, zlocals_all, tad, bad,
            active_test_ids, active_trial_ids,
            test_id_map, trial_id_map,
            num_tshapes, num_bshapes,
            num_test, num_trial,
            output_rows, output_cols,
        )
    else
        error("gpu_scatter!: unknown variant :$variant — use :direct or :tiled")
    end

    # Combine real and imaginary parts into the complex output
    output .= complex.(output_re, output_im)

    CUDA.synchronize()
    return output
end


function _launch_scatter_direct!(
    output_re, output_im, zlocals_all, tad, bad,
    active_test_ids, active_trial_ids,
    test_id_map, trial_id_map,
    num_tshapes, num_bshapes,
    num_test, num_trial,
    output_rows, output_cols,
)
    total_pairs = Int(num_test) * Int(num_trial)
    MN = Int(num_tshapes) * Int(num_bshapes)
    total_zlocals = total_pairs * MN

    # Use launch_configuration to pick a good thread count
    kernel = @cuda launch = false scatter_kernel_direct!(
        output_re, output_im, zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Int32(output_rows), Int32(output_cols),
    )
    config = launch_configuration(kernel.fun)
    threads = min(total_zlocals, config.threads)
    blocks = cld(total_zlocals, threads)

    kernel(
        output_re, output_im, zlocals_all,
        tad.flat, tad.offsets, tad.lengths,
        bad.flat, bad.offsets, bad.lengths,
        active_test_ids, active_trial_ids,
        test_id_map, trial_id_map,
        Int32(num_tshapes), Int32(num_bshapes),
        Int32(num_test), Int32(num_trial),
        Int32(output_rows), Int32(output_cols);
        threads, blocks,
    )
end

# _launch_scatter_tiled! is defined in gpu_v1.5.jl
