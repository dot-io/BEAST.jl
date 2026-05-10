
abstract type Tiling end

struct IndexTiling <: Tiling
    indices::Vector{Int}
end

struct EqualTiling <: Tiling
    N::Int
end

struct WorksizeTiling <: Tiling
    workpackagesize::Int
end


function split(workloadsize::Int, tiling::IndexTiling)
    return tiling.indices
end

function split(workloadsize::Int, tiling::EqualTiling)
    len = workloadsize
    N = tiling.N
    splits = [collect(s:min(s+ceil(Int,len/N)-1, len)) for s in 1:ceil(Int,len/N):len]
    return splits
end

function split(workloadsize::Int, tiling::WorksizeTiling)
    len = workloadsize
    N = tiling.workpackagesize
    splits = [collect(s:min(s+N-1, len)) for s in 1:N:len]
    return splits
end



struct TilingStrategy{N}
    tiling::NTuple{N,Tiling}
end

Base.getindex(tstrat::TilingStrategy{N}, i::Int) where N = tstrat.tiling[i]

TilingStrategy(args...) = TilingStrategy{length(args)}(NTuple(args)) 
