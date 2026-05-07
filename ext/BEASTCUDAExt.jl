module BEASTCUDAExt

using CUDA
using BEAST: BEAST, Integrand, scalartype, numfunctions, geometry, refspace
using CompScienceMeshes: MeshPointNM, Simplex, domain, chart
using StaticArrays: SVector

# Shared types and helpers (DeviceStore, CuMatrixStore, FlattenedAssemblyData,
# InvAssemblyData, flatten_quaddata_gpu, create_id_maps, filter_and_copy_dev,
# TILE_SIZE, …).
include("utils.jl")

# Kernel implementations — one file per design point. Each file defines only
# its kernel (and any kernel-specific helper such as `build_tile_pairs`).
include("gpu_v1.jl")    # element-stationary scatter        — :scatter
include("gpu_v2.jl")    # entry-stationary gather           — :gather_entry
include("gpu_v3.jl")    # tile-stationary gather Layer 1    — :gather_tile
include("gpu_v4.jl")    # tile-stationary gather Layer 2    — :gather_tile_coop

# Kernel-agnostic launcher: `assembleblock_primer_gpu`,
# `assembleblock_body_gpu!`, `assembleblock_gpu`. Selects the kernel via
# `kernel=:scatter | :gather_entry | :gather_tile | :gather_tile_coop`.
include("assembly.jl")

export assembleblock_gpu, assembleblock_body_gpu!, assembleblock_primer_gpu,
    CuMatrixStore, DeviceStore,
    FlattenedAssemblyData, InvAssemblyData

end
