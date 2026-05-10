module BEASTCUDAExt

using CUDA
using BEAST: BEAST, Integrand, scalartype, numfunctions, geometry, refspace
using CompScienceMeshes: CompScienceMeshes, MeshPointNM, Simplex, domain, chart
using StaticArrays: SVector
using SparseArrays

# Shared types and helpers
include("utils.jl")

# the different implementations one per file
include("gpu_v1.jl")    # element-stationary scatter
include("gpu_v1.5.jl")  # element-stationary scatter with shmem tiling
include("gpu_v2.jl")    # entry-stationary gather (1 GPU block per dof pair)
include("gpu_v3.jl")    # tile-stationary gather v1 (1 CUDA thread per dof pair)
include("gpu_v4.jl")    # tile-stationary gather v2 (adds integrand precomputation)


# Implementation using Sparse matrix multiplication to avoid contention; courtesy of Cedric Münger.

module SparseImpl
using CUDA
using CUDA.Adapt
using CUDA.CUSPARSE
using BEAST
import BEAST: assemble!, Threading, Operator, Space, IntegralOperator
import BEAST: _integrands, _integrands_gen, Integrand, pulledback_integrand
import BEAST: LagrangeRefSpace, RTRefSpace, GWPDivRefSpace
using CompScienceMeshes
using SauterSchwabQuadrature
using StaticArrays
using SparseArrays
using LinearAlgebra
using ProgressMeter
using SparseArrays: SparseMatrixCSC

Adapt.@adapt_structure CommonVertex
Adapt.@adapt_structure CommonEdge
Adapt.@adapt_structure CommonFace
Adapt.@adapt_structure GWPDivRefSpace

include("sparse/tiling.jl")
include("sparse/cpu_assemble.jl")
include("sparse/gpu_utils.jl")
include("sparse/gpu_basis.jl")
include("sparse/gpu_integrals.jl")
include("sparse/gpu_assemble_integralop_v2.jl")
end

# lastly the kernel-agnostic
#  `assembleblock_primer_gpu`,
# `assembleblock_body_gpu!`,
#  `assembleblock_gpu`. Selects the kernel via
# their symbols
include("assembly.jl")

export assembleblock_gpu, assembleblock_body_gpu!, assembleblock_primer_gpu,
    CuMatrixStore, DeviceStore,
    FlattenedAssemblyData, InvAssemblyData,
    gpu_scatter_contention

end
