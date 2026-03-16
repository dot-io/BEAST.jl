using CompScienceMeshes
using BEAST

using LinearAlgebra
# using SparseArrays
# using Profile

using CUDA


#Γ = meshcuboid(1.0,1.0,1.0,1.0)
Γ = meshsphere(1.0,0.05;generator=:gmsh)

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

@show numcells(Γ)
@show numfunctions(X)

κ, η = 1.0, 1.0
T = Maxwell3D.singlelayer(wavenumber=κ)

qstrat = BEAST.DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6)


CUDA.@time Th_gpu = assemble(T,X,X;threading=:gpu,gpu_tiling=(1,1),quadstrat=qstrat)

@time Th_cpu = assemble(T,X,X;threading=:cellcoloring,quadstrat=qstrat)
@show numfunctions(X)
@show Threads.nthreads()
@show maximum(norm.(Th_gpu-Th_cpu)) eps(real(eltype(Th_cpu)))


# using StaticArrays

# el, ad, cls = BEAST.assemblydata(X)

# el = el[1]

# px = SVector{2}(0.33333,0.3333)
# x = neighborhood(el, px)

# py = SVector{2}(0.2,0.3333)
# y = neighborhood(el, py)

# bx = refspace(X)(x)[1]

# BExt = Base.get_extension(BEAST, :BEASTCUDAExt)

# igd = BEAST.Integrand(T, refspace(X), refspace(X), el, el)

# igd(py,px,1,1)
# igdp(py,px)

# BEAST._integrands(igd,bx,bx) 

# ex = :(ComplexF64)

# abstract type B end
# struct C <: B 
#     y::Int
# end

# struct A{T} 
#     x::T
# end

# function (a::A{<:C})(i) where {C}
#     println("A:", a.x.y)
# end

# function (a::A)(i::Float64)
#     a(ceil(Int, i))
# end

# function load_assemblydata(X)
#     el, ad, cls = BEAST.assemblydata(X)

#     num_shapes = numfunctions(refspace(X), domain(el[1]))
    
#     rows = Int[]
#     cols = Int[]
#     vals = ComplexF64[]
#     idx = 1
#     ax = axes(ad.data)
#     for i in ax[1]
#         for j in ax[2]
#             for k in ax[3]
#                 dof = ad.data[i,j,k][1]
#                 if dof > 0
#                     push!(rows, dof)
#                     push!(cols, num_shapes*(k-1)+j)
#                     push!(vals, ad.data[i,j,k][2])
#                     idx += 1
#                 end
#             end
#         end
#     end

#     dof_ad = sparse(rows, cols, vals)

   
#     return el,dof_ad
# end

# ad = load_assemblydata(X)

# ad_d = CuSparseMatrixCSR(ad)

# A = CuArray(rand(ComplexF64,3*BEAST.numcells(X.geo),3*BEAST.numcells(X.geo)))
# B = CuArray(zeros(ComplexF64,numfunctions(X),3*BEAST.numcells(X.geo)))
# CUDA.@time CUSPARSE.mm!('N','N',1.0,ad_d,A,0.0,B,'O')


# @profview LinearAlgebra.mul!(B, ad, transpose(A),true,false)

# using CUDA
# using StaticArrays
# using BenchmarkTools


# function gpu_outerproduct!(mat, v1, v2)
#     i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y-1) * blockDim().y + threadIdx().y

#     if i <= length(v1) && j <= length(v2)
#         mat[i,j] = v1[i] * v2[j]
#     end
#     return
# end

# function gpu_outerproduct_shared!(mat, v1, v2)
#     i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y-1) * blockDim().y + threadIdx().y

#     a_cache = CuDynamicSharedArray(Float64, blockDim().x)
#     b_cache = CuDynamicSharedArray(Float64, blockDim().y,sizeof(a_cache))

#     if i <= length(v1) && threadIdx().y == 1
#         a_cache[threadIdx().x] = v1[i]
#     end

#     if j <= length(v2) && threadIdx().x == 1
#         b_cache[threadIdx().y] = v2[j]
#     end

#     sync_threads()

   
#     if i <= length(v1) && j <= length(v2)
#         mat[j,i] = b_cache[threadIdx().y]*a_cache[threadIdx().x]
#     end
#     return
# end

# function outerproduct!(mat, v1, v2)
#     for i in 1:length(v1)
#         for j in 1:length(v2)
#             @inbounds mat[i,j] = v1[i] * v2[j]
#         end
#     end
#     return
# end

# n = 20000
# a = rand(n)
# b = transpose(rand(n))
# result = zeros(length(a), length(b))


# outerproduct!(result, a, b)

# a_d = CuArray(a)
# b_d = CuArray(b)
# result_d = CuArray{Float64}(undef, length(a_d), length(b_d))


# xThreads = 32
# yThreads = 32

# xBlocks = ceil(Int, length(a_d) / xThreads)
# yBlocks = ceil(Int, length(b_d) / yThreads)

# CUDA.@time @cuda blocks=(xBlocks,yBlocks) threads=(xThreads,yThreads) gpu_outerproduct!(result_d, a_d, b_d)


# CUDA.@time @cuda blocks=(xBlocks,yBlocks) threads=(xThreads,yThreads)  shmem =
#     ((xThreads+yThreads)*sizeof(Float64)) gpu_outerproduct_shared!(result_d, a_d, b_d)



# @show result[end,end]
# @show CUDA.@allowscalar result_d[end,end]