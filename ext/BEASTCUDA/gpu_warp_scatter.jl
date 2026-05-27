# ===========================================================================
# Warp-cooperative scatter kernel
# ===========================================================================
#
# Eliminates the K_max² atomic contention that limits pair_scatter for high
# K_max bases (Lagrange P1, K_max≈6-8 → 36-64 concurrent atomics per
# output entry in pair_scatter → heavy serialisation).
#
# Core idea:
#   Assign ALL K_max_t × K_max_b element pairs that contribute to a single
#   output entry (m, n) to ONE warp (32 lanes). Each lane independently
#   computes one contributing pair's integrand. A warp-level shfl_down_sync
#   reduction combines the 32 partial results.  Lane 0 then does a SINGLE
#   atomic add to output[m, n].
#
#   For K_max²  ≤ 32  (e.g. RT K_max=2 → 4 pairs): zero-contention.
#   For K_max²  ≤ 64  (Lagrange K_max=8 → 64 pairs): 2 warps per entry
#                      → 2 atomics instead of 64. 32× reduction.
#
# Build: `build_warp_scatter_plan` (called in the body, not the primer).
#   Iterates over all (m, n) pairs in the block, enumerates their K_max²
#   contributing element pairs using the CPU-side assembly data from ctx,
#   and packages them into sorted warp-sized work chunks uploaded to the GPU.
#
# Memory access:
#   Work arrays stored in SoA layout (separate CuVectors per field).
#   Within a warp, 32 lanes access consecutive elements of each array
#   → stride-1 coalesced reads.
#
# Overhead vs pair_scatter:
#   • build_warp_scatter_plan runs on CPU: O(M × N × K_max²) time and
#     6×M×N×K_max² + 4×n_warps words of device memory.
#     For an H-matrix 100-DOF block with K_max=8:
#       100 × 100 × 64 = 640 K work items ≈ few MB — fast.
#     For a full 1K-DOF block: ~56 MB, build ≈ 50 ms.
#     Not suitable for full-block large-n; designed for H-matrix blocks.
#   • Compute per item: same as gather_tile — computes the full SMatrix
#     (via `compute_pair_entry`) but uses only one entry per work item.
#     This is unavoidable when each warp lane owns a unique (p,q,i,j) tuple.

using CUDA: CuVector, CuMatrix, @cuda, @inbounds, shfl_down_sync


# ---------------------------------------------------------------------------
# Plan data structure
# ---------------------------------------------------------------------------

"""
    WarpScatterPlan{T}

GPU-resident work list for the warp_scatter kernel. Produced by
`build_warp_scatter_plan`; consumed by `gpu_warp_scatter!`.

Fields are stored in SoA layout: within each warp-aligned chunk of 32 work
items, all 32 lanes read the same field (e.g. `work_p`) at consecutive
addresses → stride-1 coalesced reads across the warp.

`warp_start[w]` is the 1-based index into the work arrays where warp `w`'s
chunk begins.  `warp_len[w] ≤ 32` items are valid; lanes ≥ warp_len
contribute 0 to the reduction.
"""
struct WarpScatterPlan{T}
    work_p::CuVector{Int32}         # test element global id  (1-based)
    work_q::CuVector{Int32}         # trial element global id (1-based)
    work_i::CuVector{Int32}         # test shape index        (1-based)
    work_j::CuVector{Int32}         # trial shape index       (1-based)
    work_a::CuVector{T}             # test assembly coeff (a_coeff)
    work_b::CuVector{T}             # trial assembly coeff (b_coeff)
    warp_m::CuVector{Int32}         # output row  (m_local, block-local 1-based)
    warp_n::CuVector{Int32}         # output col  (n_local, block-local 1-based)
    warp_start::CuVector{Int32}     # 1-based start in work_* arrays
    warp_len::CuVector{Int32}       # valid items in this warp (1..32)
    n_warps::Int32
end

# ---------------------------------------------------------------------------
# Plan builder (CPU-side, called in body)
# ---------------------------------------------------------------------------

"""
    build_warp_scatter_plan(test_ids, trial_ids, tad_cpu, bad_cpu,
                            num_test_el, num_trial_el,
                            num_tshapes, num_bshapes, ::Type{T})
                            -> WarpScatterPlan{T}

Build the warp-scatter work list for the given DOF block.

Iterates over all (m, n) pairs in (test_ids × trial_ids), enumerates their
K_max_t × K_max_b contributing element pairs from the CPU assembly data,
and packs them into warp-sized (≤32) chunks.  Consecutive chunks belonging
to the same (m, n) are guaranteed (the iteration order is (m, n) first),
so each chunk's 32 lanes all write to the same output entry after reduction.

Complexity: O(n_test_el × n_tshapes + n_trial_el × n_bshapes) scan,
            O(|test_ids| × |trial_ids| × K_max²) work items emitted.
"""
function build_warp_scatter_plan(
    test_ids, trial_ids,
    tad_cpu, bad_cpu,
    num_test_el::Int, num_trial_el::Int,
    num_tshapes::Int, num_bshapes::Int,
    ::Type{T},
) where {T}
    n_test  = length(test_ids)
    n_trial = length(trial_ids)

    # ── Build restricted inverse maps ──
    # Scan forward assembly data once, keeping only entries for requested DOFs.
    test_g2l  = Dict(m => k for (k, m) in enumerate(test_ids))
    trial_g2l = Dict(n => k for (k, n) in enumerate(trial_ids))

    inv_test = [Tuple{Int32,Int32,T}[] for _ in 1:n_test]
    inv_trial = [Tuple{Int32,Int32,T}[] for _ in 1:n_trial]

    for p in 1:num_test_el, i in 1:num_tshapes
        for (m, coeff) in tad_cpu[p, i]
            if haskey(test_g2l, m)
                push!(inv_test[test_g2l[m]], (Int32(p), Int32(i), T(coeff)))
            end
        end
    end
    for q in 1:num_trial_el, j in 1:num_bshapes
        for (n, coeff) in bad_cpu[q, j]
            if haskey(trial_g2l, n)
                push!(inv_trial[trial_g2l[n]], (Int32(q), Int32(j), T(coeff)))
            end
        end
    end

    # ── Emit work items grouped by (m_local, n_local) ──
    work_p_h = Int32[]; work_q_h = Int32[]
    work_i_h = Int32[]; work_j_h = Int32[]
    work_a_h = T[];     work_b_h = T[]
    warp_m_h = Int32[]; warp_n_h = Int32[]
    warp_start_h = Int32[]; warp_len_h = Int32[]

    item_count = 0   # running count; warp_start is 1-indexed = item_count+1 at seg start

    for m_local in 1:n_test
        t_list = inv_test[m_local]
        isempty(t_list) && continue
        for n_local in 1:n_trial
            b_list = inv_trial[n_local]
            isempty(b_list) && continue

            seg_start = item_count + 1   # 1-indexed start of this (m,n) segment

            for (p, i, a) in t_list, (q, j, b) in b_list
                push!(work_p_h, p); push!(work_q_h, q)
                push!(work_i_h, i); push!(work_j_h, j)
                push!(work_a_h, a); push!(work_b_h, b)
                item_count += 1
            end

            seg_len = item_count - (seg_start - 1)

            # Split into ≤32-item warp chunks
            chunk = seg_start
            while chunk <= seg_start + seg_len - 1
                clen = min(32, seg_start + seg_len - chunk)
                push!(warp_m_h, Int32(m_local))
                push!(warp_n_h, Int32(n_local))
                push!(warp_start_h, Int32(chunk))
                push!(warp_len_h,   Int32(clen))
                chunk += 32
            end
        end
    end

    isempty(warp_start_h) &&
        error("build_warp_scatter_plan: no work items — check test_ids/trial_ids are non-empty and overlap with the mesh")

    return WarpScatterPlan{T}(
        CUDA.cu(work_p_h), CUDA.cu(work_q_h),
        CUDA.cu(work_i_h), CUDA.cu(work_j_h),
        CUDA.cu(work_a_h), CUDA.cu(work_b_h),
        CUDA.cu(warp_m_h), CUDA.cu(warp_n_h),
        CUDA.cu(warp_start_h), CUDA.cu(warp_len_h),
        Int32(length(warp_start_h)),
    )
end


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------

"""
    warp_scatter_kernel!(output_re, output_im, ...)

One warp (32 lanes) per entry in the `warp_*` arrays.
Each warp:
  1. Each lane reads one work item (p, q, i, j, a, b) — stride-1 coalesced.
  2. Each lane calls `compute_pair_entry` to obtain z[i,j] for its element
     pair and computes `a × z × b`.
  3. Real and imaginary parts are reduced across the 32 lanes via warp shuffles.
  4. Lane 0 atomically adds the reduced contribution to output[m_out, n_out].

For segments with K_max² ≤ 32 (one warp per output entry): each output entry
receives exactly one atomic, which is uncontended.
For K_max² ≤ 64: two warps per entry → two atomics (2-way contention vs 64-way
in pair_scatter).  Idle lanes (lane ≥ warp_len) contribute 0 and do not add
warp divergence since warp_len is uniform within the warp.
"""
function warp_scatter_kernel!(
    output_re::CuDeviceMatrix{F},
    output_im::CuDeviceMatrix{F},
    biop, test_shapes, trial_shapes,
    test_elements, trial_elements,
    work_p::CuDeviceVector{Int32},
    work_q::CuDeviceVector{Int32},
    work_i::CuDeviceVector{Int32},
    work_j::CuDeviceVector{Int32},
    work_a::CuDeviceVector{T},
    work_b::CuDeviceVector{T},
    warp_m::CuDeviceVector{Int32},
    warp_n::CuDeviceVector{Int32},
    warp_start::CuDeviceVector{Int32},
    warp_len::CuDeviceVector{Int32},
    n_warps::Int32,
    tqp_flat, tqp_offsets, tqp_lengths,
    bqp_flat, bqp_offsets, bqp_lengths,
    ::Val{NS}, ::Val{MS},
) where {F<:AbstractFloat,T<:Complex{F},NS,MS}

    warps_per_block = blockDim().x >> Int32(5)
    warp_in_block   = (threadIdx().x - Int32(1)) >> Int32(5)
    warp_id = (blockIdx().x - Int32(1)) * warps_per_block + warp_in_block + Int32(1)
    lane    = (threadIdx().x - Int32(1)) & Int32(31)

    warp_id > n_warps && return

    @inbounds w_m     = warp_m[warp_id]
    @inbounds w_n     = warp_n[warp_id]
    @inbounds w_start = warp_start[warp_id]   # 1-indexed
    @inbounds w_len   = warp_len[warp_id]     # ≤32 valid items

    acc_re = zero(F)
    acc_im = zero(F)

    # Lane computes its assigned work item (if valid)
    if lane < w_len
        item = w_start + lane    # 1-indexed into work_* arrays

        @inbounds p = work_p[item]
        @inbounds q = work_q[item]
        @inbounds i = work_i[item]
        @inbounds j = work_j[item]
        @inbounds a = work_a[item]
        @inbounds b = work_b[item]

        @inbounds tcell = test_elements[p]
        @inbounds bcell = trial_elements[q]
        @inbounds t_off = tqp_offsets[p]
        @inbounds t_len = tqp_lengths[p]
        @inbounds b_off = bqp_offsets[q]
        @inbounds b_len = bqp_lengths[q]

        # compute_pair_entry runs the full quadrature loop and extracts z[i,j].
        # The SMatrix is computed in full inside igd() regardless — using
        # compute_pair_entry avoids accumulating the other NS×MS-1 entries.
        z_ij = compute_pair_entry(T,
            biop, test_shapes, trial_shapes,
            tcell, bcell, i, j,
            tqp_flat, t_off, t_len,
            bqp_flat, b_off, b_len)

        contrib = a * z_ij * b
        acc_re = real(contrib)
        acc_im = imag(contrib)
    end

    # Warp-level reduction (ComplexF64 split into real + imag Float64)
    for offset in (Int32(16), Int32(8), Int32(4), Int32(2), Int32(1))
        acc_re += shfl_down_sync(0xffffffff, acc_re, offset)
        acc_im += shfl_down_sync(0xffffffff, acc_im, offset)
    end

    if lane == Int32(0)
        CUDA.@atomic output_re[w_m, w_n] += acc_re
        CUDA.@atomic output_im[w_m, w_n] += acc_im
    end

    return nothing
end


# ---------------------------------------------------------------------------
# Launch wrapper
# ---------------------------------------------------------------------------

"""
    gpu_warp_scatter!(output, biop, test_shapes, trial_shapes,
                      test_elements, trial_elements, plan, quaddata_gpu,
                      num_tshapes, num_bshapes)

Launch the warp_scatter kernel.  The plan must have been built for the
same (test_ids, trial_ids) block as `output`.
"""
function gpu_warp_scatter!(
    output::CuMatrix{T},
    biop, test_shapes, trial_shapes,
    test_elements, trial_elements,
    plan::WarpScatterPlan{T},
    quaddata_gpu,
    num_tshapes::Int,
    num_bshapes::Int,
) where {T}
    F = real(T)
    output_re = CUDA.zeros(F, size(output, 1), size(output, 2))
    output_im = CUDA.zeros(F, size(output, 1), size(output, 2))

    warps_per_block = 8
    threads = warps_per_block * 32   # 256
    blocks  = cld(Int(plan.n_warps), warps_per_block)

    @cuda threads=threads blocks=blocks warp_scatter_kernel!(
        output_re, output_im,
        biop, test_shapes, trial_shapes,
        test_elements, trial_elements,
        plan.work_p, plan.work_q,
        plan.work_i, plan.work_j,
        plan.work_a, plan.work_b,
        plan.warp_m, plan.warp_n,
        plan.warp_start, plan.warp_len,
        plan.n_warps,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        Val(num_tshapes), Val(num_bshapes),
    )

    output .= complex.(output_re, output_im)
    CUDA.synchronize()
    return output
end
