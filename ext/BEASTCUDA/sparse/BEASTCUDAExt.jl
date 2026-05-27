module BEASTCUDA

using CUDA
using CUDA.Adapt
using CUDA.CUSPARSE

using BEAST
import BEAST: assemble!, Threading, Operator, Space, IntegralOperator
import BEAST: _integrands, _integrands_gen, Integrand, pulledback_integrand
import BEAST: LagrangeRefSpace, RTRefSpace, GWPDivRefSpace
using BEAST.CompScienceMeshes
using BEAST.SauterSchwabQuadrature
using BEAST.StaticArrays
using BEAST.SparseArrays
using BEAST.LinearAlgebra
using BEAST.ProgressMeter
using SparseArrays: SparseMatrixCSC

Adapt.@adapt_structure CommonVertex
Adapt.@adapt_structure CommonEdge
Adapt.@adapt_structure CommonFace

Adapt.@adapt_structure GWPDivRefSpace

include("tiling.jl")

include("cpu_assemble.jl")

include("gpu_utils.jl")
include("gpu_basis.jl")
include("gpu_integrals.jl")
include("gpu_assemble_integralop_v2.jl")

end
