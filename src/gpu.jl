function blockassembler_gpu(biop::IntegralOperator, tfs::Space, bfs::Space; quadstrat=defaultquadstrat(biop, tfs, bfs))

    tgeo = geometry(tfs) # geometry() Simply extracts the s.geo substruct from the space
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

    if CUDA.functional()
        @info "CUDA is available, using GPU assembly"
        # device = CUDA.devices().first()
        # @info @sprintf("---Device info---\nCompute capability: %d.1f\nWarp size: %d\n \
        # Total memory: %d\nUnified Addressing Support: %d\nMemory Pool support: %d",
        # CUDA.capability(device), CUDA.warpsize(device), CUDA.totalmem(device),
        # CUDA.unified_addressing(device), CUDA.memory_pools_supported(device))

        (test_elements, test_elements_dev, tad, tad_dev,
         bsis_elements, bsis_elements_dev, bad, bad_dev,
         quaddata_gpu, zlocals_dev,
         num_tshapes, num_bshapes) =
            assembleblock_primer_gpu(biop, tfs, bfs; quadstrat=qs)

        return (test_ids, trial_ids, store) -> begin
            assembleblock_body_gpu!(biop,
                tfs, test_ids, test_elements, test_elements_dev, tad, tad_dev,
                bfs, trial_ids, bsis_elements, bsis_elements_dev, bad, bad_dev,
                quaddata_gpu, zlocals_dev, num_tshapes, num_bshapes, store;
                quadstrat=qs)
        end
    else
        @error "CUDA not available."
        return
    end
end


"""
    SpaceGPU{V,F,T}

GPU representation of any `Space` subtype. Flattens the nested `fns`
(Vector{Vector{Shape{T}}}) into contiguous CuArrays indexed by an offset
array, and moves mesh vertices, faces, and positions to device memory.

## Fields

Geometry (from `space.geo`):
- `geo_vertices` – `CuArray{V}` of mesh vertices (e.g. `SVector{3,Float64}`)
- `geo_faces`    – `CuArray{F}` of face index tuples  (e.g. `SVector{3,Int32}`)

Shape functions (flattened from `space.fns`):
- `shapes_cellid` – `CuArray{Int32}`, cell index of each shape
- `shapes_refid`  – `CuArray{Int32}`, local ref-function index of each shape
- `shapes_coeff`  – `CuArray{T}`,     coefficient of each shape
- `fns_offsets`    – `CuArray{Int32}`, length `numfunctions(space)+1`;
                     the shapes for global basis function `m` are at indices
                     `fns_offsets[m]+1 : fns_offsets[m+1]`

Positions:
- `pos` – `CuArray` of position vectors (same eltype as `space.pos`)

Metadata (kept on host for convenience):
- `numfunctions` – number of global basis functions (`length(space.fns)`)
"""
struct SpaceGPU{V,F,T,P}
    # geometry
    geo_vertices::CuArray{V,1}
    geo_faces::CuArray{F,1}
    # flattened shapes
    shapes_cellid::CuArray{Int32,1}
    shapes_refid::CuArray{Int32,1}
    shapes_coeff::CuArray{T,1}
    fns_offsets::CuArray{Int32,1}
    # positions
    pos::CuArray{P,1}
    # host-side metadata
    numfunctions::Int
end

"""
    SpaceGPU(basis::Space)

Construct a `SpaceGPU` from any BEAST `Space` (LagrangeBasis, RTBasis,
NDBasis, BDMBasis, …).  The three fields `basis.geo`, `basis.fns`, and
`basis.pos` are converted to flat GPU arrays.
"""
function SpaceGPU(basis::Space)
    # --- geometry ----------------------------------------------------------
    geo = basis.geo
    V = eltype(geo.vertices)               # e.g. SVector{3,Float64}
    F_idx = eltype(geo.faces)              # e.g. SVector{3,Int}
    # convert face indices to Int32 for GPU efficiency
    F32 = SVector{length(F_idx),Int32}
    verts_host = V.(geo.vertices)
    faces_host = F32.(geo.faces)

    # --- shapes (flatten fns) ----------------------------------------------
    T = scalartype(basis)
    offsets_host = Int32.(cumsum([0; length.(basis.fns)]))
    if isempty(basis.fns) || all(isempty, basis.fns)
        cell_ids = Int32[]
        ref_ids  = Int32[]
        coeffs   = T[]
    else
        shapes = reduce(vcat, basis.fns)
        cell_ids = Int32[s.cellid for s in shapes]
        ref_ids  = Int32[s.refid  for s in shapes]
        coeffs   = T[s.coeff     for s in shapes]
    end

    # --- positions ---------------------------------------------------------
    P = eltype(basis.pos)
    pos_host = P.(basis.pos)

    # --- transfer to device ------------------------------------------------
    return SpaceGPU{V,F32,T,P}(
        CUDA.cu(verts_host),
        CUDA.cu(faces_host),
        CUDA.cu(cell_ids),
        CUDA.cu(ref_ids),
        CUDA.cu(coeffs),
        CUDA.cu(offsets_host),
        CUDA.cu(pos_host),
        length(basis.fns),
    )
end

"""
    to_gpu(basis::Space) -> SpaceGPU

Convert any BEAST `Space` (test or basis functions) to GPU device memory.
Accepts `LagrangeBasis`, `RTBasis`, `NDBasis`, `BDMBasis`, and all other
`Space{T}` subtypes that carry the standard `(geo, fns, pos)` fields.
"""
to_gpu(basis::Space) = SpaceGPU(basis)

function assembleblock_primer_gpu(biop, tfs, bfs;
        quadstrat=defaultquadstrat(biop, tfs, bfs), gpu=false)

#     """
#     charts, admap, act_to_global = assemblydata(basis; onlyactives=true)

# Given a `basis` this function returns a data structure containing the information
# required for matrix assemble, that is, the vector `charts` containing `Simplex` elements,
# a variable `admap` of type `AssemblyData`, and a mapping from indices of actively
# used simplices to global simplices.

# When `onlyactives` is `true`, another layer of indices is introduced to filter out all
# cells of the mesh that are not in the union of the support of the basis functions
# (i.e., when the basis functions are defined only on a part of the mesh).

# `admap` is, in essence, a three-dimensional array of named tuples, which,
# by wrapping it in the struct `AssemblyData`, allows the definition of iterators.
# The tuple consists of the two entries

# ```
# admap[i,r,c].globalindex
# admap[i,r,c].coefficient
# ```

# Here, `c` and `r` are indices in the iterable set of (active) simplices and the
# set of shape functions on each cell/simplex: `r` ranges from 1 to the number of
# shape functions on a cell/simplex, `c` ranges from 1 to the number of active
# simplices, and `i` ranges from 1 to the number of maximal number of basis functions,
# where any of the shape functions contributes to. 

# For example, for continuous piecewise linear lagrange functions (c0d1), each of the
# three shape functions on a triangle are associated with exactly one Lagrange function,
# and therefore `i` is limited to 1.

# *Note*: When `onlyactives=false`, the indices `c` correspond to
# the position of the corresponding cell/simplex whilst iterating over `geometry(basis)`.
# When `onlyactives=true`, then `act_to_global(c)` correspond to the position of the
# corresponding cell/simplex whilst iterating over `geometry(basis)`.

# For a triplet `(i,r,c)`, `globalindex` is the index in the `basis` of the
# `i`th basis function that has a contribution from shape function `r` on
# (active) cell/simplex `c`. `coefficient` is the coefficient of that contribution in the
# linear combination defining that basis function in terms of shape
# function.
# """
# function assemblydata(basis::Space; onlyactives=true)

#     @assert numfunctions(basis) != 0

#     T = scalartype(basis)

#     geo = geometry(basis)
#     num_cells = numcells(geo)

#     num_bfs  = numfunctions(basis)
    
#     ch = chart(geo, first(geo))
#     dom = domain(ch)
#     num_refs = numfunctions(refspace(basis), dom)

#     # Determine the number of functions defined over a given cell
#     # and per local shape function.
#     celltonum = make_celltonum(num_cells, num_refs, num_bfs, basis)

#     # In general, a basis function space might only be defined
#     # over a small portion of the underlying mesh. To avoid
#     # the inefficient iterating of cells, which are not in the support of
#     # any of the basis functions, we filter out only those cells, over 
#     # which at least one basis function is defined.
#     if onlyactives
#         active, index_among_actives, num_active_cells, act_to_global =
#             index_actives(num_cells, celltonum)
#     else
#         active = trues(num_cells)
#         num_active_cells = num_cells
#         index_among_actives = collect(1:num_cells)
#         act_to_global = collect(1:num_cells)
#     end

#     num_active_cells == 0 && return nothing

#     # Generate the a vector of Simplexes associated
#     # with the active cells only
#     elements = instantiate_charts(geo, num_active_cells, active)

#     # Determine the maximal number of functions associated with a
#     # local shape function
#     max_celltonum = maximum(celltonum)
#     fill!(celltonum, 0)
#     data = fill((0,zero(T)), max_celltonum, num_refs, num_active_cells)
#     for b in 1 : num_bfs
#         for shape in basisfunction(basis, b)
#             c = shape.cellid
#             l = index_among_actives[c]
#             @assert 0 < l <= num_active_cells
#             r = shape.refid
#             w = shape.coeff
#             k = (celltonum[c,r] += 1)
#             data[k,r,l] = (b,w)
#         end
#     end

#     return elements, AssemblyData(data), act_to_global
# end

    @info "Starting primer routine (instantiating quadrature data in GPU global memory)..."
    # this iterates over all test and basis functions... consider kernel?
    test_elements, tad, tcells = assemblydata(tfs; onlyactives=false)
    bsis_elements, bad, bcells = assemblydata(bfs; onlyactives=false)
    
    # Simply an access to the .geo substruct
    tgeo = geometry(tfs)
    bgeo = geometry(bfs)

    # Domain creation from the geometry

    # From mesh.jl: 
    
    # chart(mesh::Mesh, cell) = simplex(vertices(mesh, indices(mesh,cell)))
    
    # From mesh.jl:
    
    # function indices(m::Mesh{U,D1}, cell) where {U,D1}
    #    return m.faces[cell]
    # end
    #

    # From mesh.jl
    # @generated function vertices(m::Mesh, I::SVector)
    # N = length(I)
    # xp = :(())
    # for i in 1:N
    #     push!(xp.args, :(m.vertices[I[$i]]))
    # end
    # :(SVector($xp))
    # end

    # Creates the method to construct the static vector of vertices,
    # Each element of the Static Vector contains the vertices for one of the faces
    # Note: N here is small (usually <= 3) hence why it is unrolled at compile-time using @generated
    # This means this loop has little parallelization value, an argument for keeping primer CPU-based


    # From charts.jl

    # simplex(.) generates the simplex from the previous SVector of vertices
    # Contains the vertices themselves, the tangents, the normals and the volume

    #     @generated function simplex(vertices::SVector{D1,P}) where {D1,P}
    #     U = length(P)
    #     D = D1 - 1
    #     C = U-D
    #     T = eltype(P)
    #     xp1 =:(())
    #     for i in 1:D
    #         push!(xp1.args, :(vertices[$i]-vertices[end]))
    #     end
    #     xp2 = :(SVector{$D,P}($xp1))
    #     quote
    #         tangents = $xp2
    #         normals, volume = _normals(tangents, Val{$C})
    #         Simplex(vertices, tangents, normals, $T(volume))
    #     end
    # end

    # Note, the loop over D is again over the dimension of the simplex.
    # For a triangle this is 2. Again no GPU value to be found here.
    # Overhead of launching 2 threads (of a 32-size warp) is much too large.



    # From charts.jl

    # function domain(ch::Simplex{U,D,C,N,T}) where {U,D,C,T,N} ReferenceSimplex{D,T,N}() end
    # function domain(ch::ReferenceSimplex) ch end

    # Idempotent function? domain is the chart if I understant correctly
    # However why is it only generated based on the first cell? Is it because it will interact with
    # every other cell, hence why the domain is a support for every test function?
    
    tdom = domain(chart(tgeo, first(tgeo)))
    bdom = domain(chart(bgeo, first(bgeo)))

    tshapes = refspace(tfs); num_tshapes = numfunctions(tshapes, tdom)
    bshapes = refspace(bfs); num_bshapes = numfunctions(bshapes, bdom)
    qd = quaddata(biop, tshapes, bshapes, test_elements, bsis_elements, quadstrat)
    
    # Allocate the empty zlocals array straight on devmem to save host mem space

    zlocals_dev = CUDA.zeros(scalartype(biop, tfs, bfs), num_tshapes, num_bshapes)
    
    tad_dev = CUDA.cu(tad.data)
    bad_dev = CUDA.cu(bad.data)

    # Transfer element (Simplex) vectors to GPU.
    # Simplex is an isbitstype (SVector fields + scalar), so CuArray stores
    # them directly in device memory.  Indexing on the GPU returns a Simplex
    # with the same .vertices / .tangents / .normals / .volume accessors.
    test_elements_dev  = CUDA.cu(test_elements)
    bsis_elements_dev  = CUDA.cu(bsis_elements)

    # Flattening quaddata for GPU 
    # For a far-field interaction, which is expected if a block is passed to ACA
    # for compression, quadrule() simply returns 
    # DoubleQuadRule(qd.tpoints[1,test_id], qd.bpoints[1,trial_id])
    # where qd.tpoints and qd.bpoints are matrices of Vector{NamedTuple}
    #
    # Flatten these per-element vectors into contiguous arrays so the GPU
    # kernel can look up the quad points for any element by offset+length,
    # bypassing quadrule()
    #
    # tqp_flat / bqp_flat : contiguous array of all quad points
    # tqp_offsets / bqp_offsets : 1-based start index for element i
    # tqp_lengths / bqp_lengths : number of quad points for element i

    # For DoubleNumWiltonSauterQStrat: qd.tpoints[1,:] are the far-rule points
    # For DoubleNumQStrat: qd[1][1,:] are the only points
    tqp_matrix = nothing
    bqp_matrix = nothing
    if qd isa NamedTuple && haskey(qd, :tpoints)
        # DoubleNumWiltonSauterQStrat layout
        tqp_matrix = qd.tpoints
        bqp_matrix = qd.bpoints
    elseif qd isa Tuple
        # DoubleNumQStrat layout: qd = (test_quad_data, trial_quad_data)
        tqp_matrix = qd[1]
        bqp_matrix = qd[2]
    else
        #TODO: generalize for other quadrature rules (?)
        error("Unsupported quadrature data format for GPU assembly")
    end

    num_test_els  = size(tqp_matrix, 2)
    num_trial_els = size(bqp_matrix, 2)

    # Flatten test quad points
    tqp_flat_host   = reduce(vcat, tqp_matrix[1, e] for e in 1:num_test_els)
    tqp_lengths_host = Int32[length(tqp_matrix[1, e]) for e in 1:num_test_els]
    tqp_offsets_host = Int32.(cumsum([1; tqp_lengths_host[1:end-1]]))

    # Flatten trial quad points
    bqp_flat_host   = reduce(vcat, bqp_matrix[1, e] for e in 1:num_trial_els)
    bqp_lengths_host = Int32[length(bqp_matrix[1, e]) for e in 1:num_trial_els]
    bqp_offsets_host = Int32.(cumsum([1; bqp_lengths_host[1:end-1]]))

    # Transfer to GPU
    tqp_flat_dev    = CUDA.cu(tqp_flat_host)
    tqp_offsets_dev = CUDA.cu(tqp_offsets_host)
    tqp_lengths_dev = CUDA.cu(tqp_lengths_host)
    bqp_flat_dev    = CUDA.cu(bqp_flat_host)
    bqp_offsets_dev = CUDA.cu(bqp_offsets_host)
    bqp_lengths_dev = CUDA.cu(bqp_lengths_host)

    # Pack the GPU quad data into a NamedTuple for easy passing
    quaddata_gpu = (
        tqp_flat    = tqp_flat_dev,
        tqp_offsets = tqp_offsets_dev,
        tqp_lengths = tqp_lengths_dev,
        bqp_flat    = bqp_flat_dev,
        bqp_offsets = bqp_offsets_dev,
        bqp_lengths = bqp_lengths_dev,
    )

    @info "...Primer routine finished."

    return (test_elements, test_elements_dev, tad, tad_dev,
            bsis_elements, bsis_elements_dev, bad, bad_dev,
            quaddata_gpu, zlocals_dev, num_tshapes, num_bshapes)
end 



"""
    momintegrals_kernel!(
        zlocals_all, op, test_shapes, trial_shapes,
        test_elements, bsis_elements,
        active_test_ids, active_trial_ids,
        tqp_flat, tqp_offsets, tqp_lengths,
        bqp_flat, bqp_offsets, bqp_lengths,
        num_tshapes, num_bshapes, num_test, num_trial)

Single CUDA kernel that computes local moment integrals for **all** active
element pairs and writes them into `zlocals_all`.

## Thread mapping

Each thread handles one `(pair_idx, i, j)` triple where:
- `pair_idx` ∈ 1:num_test*num_trial — linearised index over all active pairs
- `i` ∈ 1:num_tshapes — local test shape function index
- `j` ∈ 1:num_bshapes — local trial shape function index

The pair index is decoded into element ids:
    p_local = div(pair_idx-1, num_trial) + 1   → index into active_test_ids
    q_local = mod(pair_idx-1, num_trial) + 1   → index into active_trial_ids
    p = active_test_ids[p_local]                → global element id
    q = active_trial_ids[q_local]               → global element id

## Quadrature lookup (replaces quadrule on device)

For DoubleQuadRule (far-field) interactions, the host-side `quadrule`
simply returns `DoubleQuadRule(qd.tpoints[1,p], qd.bpoints[1,q])`.
We bypass this on the GPU by passing pre-flattened quadrature point
arrays indexed by element id:
    outer quad points for element p:
        tqp_flat[tqp_offsets[p] : tqp_offsets[p] + tqp_lengths[p] - 1]
    inner quad points for element q:
        bqp_flat[bqp_offsets[q] : bqp_offsets[q] + bqp_lengths[q] - 1]

## Why this works without __device__

In CUDA.jl every Julia function that operates purely on isbitstype data
is automatically device-callable. The `Integrand` struct and its entire
call chain (operator-specific overloads, `_integrands`, `_krondot`,
`cartesian`, `normal`, `exp`, `norm`, `dot`, `cross`) satisfy this.
Constructing `Integrand` inside the kernel is a register-only operation.
"""
function momintegrals_kernel!(
        zlocals_all, op, test_shapes, trial_shapes,
        test_elements, bsis_elements,
        active_test_ids, active_trial_ids,
        tqp_flat, tqp_offsets, tqp_lengths,
        bqp_flat, bqp_offsets, bqp_lengths,
        num_tshapes::Int32, num_bshapes::Int32,
        num_test::Int32, num_trial::Int32)

    
    # Get global thread index (no lexical advantage in using .y and .z as this loop
    # is 4 dimensional)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    MN         = num_tshapes * num_bshapes
    num_pairs  = num_test * num_trial
    total      = num_pairs * MN
    idx > total && return

    # Convert the global linear index to a 3-tuple index (pair_idx, i, j)
    # pair_idx is the index of a test-trial pair 
    pair_idx = div(idx - Int32(1), MN) + Int32(1)
    rem_ij   = mod(idx - Int32(1), MN)
    i        = mod(rem_ij, num_tshapes) + Int32(1)
    j        = div(rem_ij, num_tshapes) + Int32(1)

    # Convert the pair index to a local and then global row and column index,
    p_local = div(pair_idx - Int32(1), num_trial) + Int32(1)
    q_local = mod(pair_idx - Int32(1), num_trial) + Int32(1)
    p = active_test_ids[p_local]
    q = active_trial_ids[q_local]

    # Fetch elements from global indices and construct the Integrand functor
    # This call is allowed as CUDA.jl does not require __device__ function definitions as CUDA C
    # does. Effectively the programmer has access to other functions defined in the package,
    # as long as the MEMORY is device-based.
    tcell = test_elements[p]
    bcell = bsis_elements[q]
    igd = Integrand(op, test_shapes, trial_shapes, tcell, bcell)

    # Look up quadrature points for elements p and q from the FLATTENED arrays
    o_off = tqp_offsets[p] # the offset from which to index the flattened test quadrature array
    o_len = tqp_lengths[p] # the number of test quadrature points in the array
    i_off = bqp_offsets[q] # idem with trial quadrature points
    i_len = bqp_lengths[q]

    # Accumulate the (i,j) entry of the local moment integral matrix
    acc = zero(eltype(zlocals_all))

    # My reasoning for not tiling along this double for loop: the amount of quadrature points
    # can be seen as fixed: there is rarely a situation where the amount of quadrature points
    # PER CELL scales along with the problem size. A single thread can then be observed to have
    # a constant workload within this kernel.
    for oi in Int32(0):(o_len - Int32(1))
        womp  = tqp_flat[o_off + oi]
        tgeo  = womp.point
        tvals = womp.value
        jx    = womp.weight

        for ii in Int32(0):(i_len - Int32(1))
            wimp  = bqp_flat[i_off + ii]
            bgeo  = wimp.point
            bvals = wimp.value
            jy    = wimp.weight

            # Dispatches to the operator-specific Integrand overload,
            # e.g. (::Integrand{<:HH3DSingleLayerFDBIO})(x,y,f,g).
            # Returns an SMatrix{M,N}.
            z1 = igd(tgeo, bgeo, tvals, bvals)
            acc += (jx * jy) * z1[i, j]
        end
    end

    zlocals_all[i, j, pair_idx] = acc
    return nothing
end

"""
    assembleblock_body_gpu!(biop, tfs, test_ids, ..., store; quadstrat)

GPU-accelerated assembly of a block of the BEM system matrix.

This function is the GPU counterpart of the CPU `assembleblock_body!`.
It performs three phases:

**1. Identify active elements:
  Collect the unique mesh element ids that support the requested test/trial
  basis functions in this block.  Transfer the active id lists to GPU.

**2. Compute local moment integrals:
  Launch `momintegrals_kernel!` with one thread per
  `(pair, i, j)` triple.  Each thread:
  - Decodes its pair into test element `p` and trial element `q`
  - Looks up the pre-flattened far-field quadrature points for `p` and `q`
    (in contrast to what was previousl=y the host/cpu `quadrule()` call)
  - Constructs `Integrand(op, test_shapes, trial_shapes, tcell, bcell)`
  - Loops over all outer x inner quadrature points
  - Writes the result to `zlocals_all[i, j, pair_idx]`

**Phase 3 (CPU)** – Scatter local→global:
  Copy the per-pair local integrals back to host.  Use the CPU-side
  `AssemblyData` iterators to expand local shape indices `(i,j)` on
  element pair `(p,q)` into global basis function indices `(m′,n′)`,
  and call `store(a * zval * b, m′, n′)` for each contribution.
"""
function assembleblock_body_gpu!(
        biop::IntegralOperator,
        tfs, test_ids,
        test_elements,       # CPU Vector{Simplex} for assertions
        test_elements_dev,   # CuArray{Simplex} for the CUDA kernel
        test_assembly_data,  # CPU AssemblyData for local→global scatter
        test_assembly_dev,   # CuArray (tad.data) reserved for future GPU scatter
        bfs, trial_ids,
        bsis_elements,       # CPU Vector{Simplex}
        bsis_elements_dev,   # CuArray{Simplex}
        trial_assembly_data, # CPU AssemblyData
        trial_assembly_dev,  # CuArray (bad.data)
        quaddata_gpu,        # NamedTuple of flattened quad-point CuArrays
        zlocals,             # CuMatrix (M×N scratch, used for eltype/size)
        num_tshapes, num_bshapes,
        store; quadstrat)

    test_shapes  = refspace(tfs)
    trial_shapes = refspace(bfs)

    # Identify active elements (CPU)
    active_test_el_ids  = Vector{Int64}()
    active_trial_el_ids = Vector{Int64}()

    test_id_in_blk  = Dict{Int,Int}()
    trial_id_in_blk = Dict{Int,Int}()

    for (i,m) in enumerate(test_ids);  test_id_in_blk[m]  = i; end
    for (i,m) in enumerate(trial_ids); trial_id_in_blk[m] = i; end

    for m in test_ids,  sh in tfs.fns[m]; push!(active_test_el_ids,  Int32(sh.cellid)); end
    for m in trial_ids, sh in bfs.fns[m]; push!(active_trial_el_ids, Int32(sh.cellid)); end

    active_test_el_ids  = unique!(sort!(active_test_el_ids))
    active_trial_el_ids = unique!(sort!(active_trial_el_ids))

    (isempty(active_test_el_ids) || isempty(active_trial_el_ids)) && return


    # If I remove these assertions I don't have to pass the CPU arrays anymore. This doesnt cost anything however
    @assert maximum(active_test_el_ids)  <= length(test_elements)
    @assert maximum(active_trial_el_ids) <= length(bsis_elements)

    # Transfer active element id lists to GPU
    active_test_ids_dev  = CUDA.cu(active_test_el_ids)
    active_trial_ids_dev = CUDA.cu(active_trial_el_ids)

    num_test  = Int32(length(active_test_el_ids))
    num_trial = Int32(length(active_trial_el_ids))
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
    M  = Int32(num_tshapes)
    N  = Int32(num_bshapes)

    # zlocals instantiated straight on GPU

    zlocals_all_dev = CUDA.zeros(ZT, M, N, num_pairs)

    # total amount of threads to be instantiated.
    total_work = Int(num_pairs) * Int(M) * Int(N)
    threads_per_block = 256 # Increase to 1024?
    nblocks = cld(total_work, threads_per_block)

    # kernel launch

    @cuda threads=threads_per_block blocks=nblocks momintegrals_kernel!(
        zlocals_all_dev, biop, test_shapes, trial_shapes,
        test_elements_dev, bsis_elements_dev,
        active_test_ids_dev, active_trial_ids_dev,
        quaddata_gpu.tqp_flat, quaddata_gpu.tqp_offsets, quaddata_gpu.tqp_lengths,
        quaddata_gpu.bqp_flat, quaddata_gpu.bqp_offsets, quaddata_gpu.bqp_lengths,
        M, N, num_test, num_trial)
    
    #synchronize so that all threads have finished for this iteration
    CUDA.synchronize()

    # scatter from local to global, this is executed on CPU
    # Copy per-pair local integrals back to host, then expand into global
    # matrix positions using the AssemblyData mapping and store() callback.

    # TODO: instantiate global (H-)Matrix and perform scatter-gather on GPU

    # Device to host memory copying
    zlocals_all_host = Array(zlocals_all_dev)

    for k in 1:num_pairs
        # Decode the pair index back to element ids
        p_local = div(k - 1, num_trial) + 1
        q_local = mod(k - 1, num_trial) + 1
        p = active_test_el_ids[p_local]
        q = active_trial_el_ids[q_local]

        for j in 1:num_bshapes
            for i in 1:num_tshapes
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