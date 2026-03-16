module BEASTCUDAExt

using CUDA
using CUDA.Adapt
using CUDA.CUSPARSE

using BEAST
import BEAST: assemble!, Threading, Operator, Space, IntegralOperator
import BEAST: _integrands, _integrands_gen, RTRefSpace, Integrand, pulledback_integrand
using BEAST.CompScienceMeshes
using BEAST.SauterSchwabQuadrature
using BEAST.StaticArrays
using BEAST.SparseArrays
using BEAST.LinearAlgebra


Adapt.@adapt_structure CommonVertex
Adapt.@adapt_structure CommonEdge
Adapt.@adapt_structure CommonFace


include("gpu_utils.jl")
# include("gpu_basis.jl")
include("gpu_integrals.jl")
include("gpu_assemble_integralop_v2.jl")

end