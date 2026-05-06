module BEASTCUDAExt

using CUDA
using BEAST: BEAST, Integrand, scalartype, numfunctions, geometry, refspace
using CompScienceMeshes: MeshPointNM, Simplex, domain, chart
using StaticArrays: SVector


include("utils.jl")

VERSION = 4
include("gpu_v$VERSION.jl")


export assembleblock_gpu!, CuMatrixStore
end
