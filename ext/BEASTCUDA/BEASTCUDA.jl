module BEASTCUDA

using CUDA
using BEAST: BEAST, Integrand, scalartype, numfunctions, geometry, refspace
using CompScienceMeshes: CompScienceMeshes, MeshPointNM, Simplex, domain, chart
using StaticArrays: SVector
using SparseArrays

# Shared types and helpers
include("utils.jl")

# the different implementations one per file
include("gather.jl")
include("scatter.jl")
include("hybrid_global.jl")  # biphasic kernel with Z in global memory
include("hybrid_shared.jl")  # biphasic kernel with Z_tile in shared memory
include("gpu_warp_scatter.jl")


# Implementation using Sparse matrix multiplication to avoid contention. courtesy of Cedric Münger.

module SparseImpl
using CUDA
using Adapt
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
    FlattenedAssemblyData, InvAssemblyData, FlattenedAssemblyDataWithK, HybridAssemblyData
end
