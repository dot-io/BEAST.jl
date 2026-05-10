
shapetype(::RTRefSpace{T}) where T =  @NamedTuple{value::SVector{3,T},divergence::T}

shapetype(::LagrangeRefSpace{T,D,3}) where {T,D} =  @NamedTuple{value::T,curl::SVector{3,T}}

shapetype(::LagrangeRefSpace{T,D,2}) where {T,D} =  @NamedTuple{value::T,derivative::T}

shapetype(::LagrangeRefSpace{T,D,4}) where {T,D} =  @NamedTuple{value::T,gradient::SVector{3,T}}

shapetype(::GWPDivRefSpace{T,D}) where {T,D} =  @NamedTuple{value::SVector{3,T},divergence::T}
# function (ϕ::RTRefSpace)(mp,i::Int)

#     u, v = parametric(mp)
#     j = jacobian(mp)

#     tu = tangents(mp,1)
#     tv = tangents(mp,2)

#     inv_j = 1/j
#     d = 2 * inv_j

#     u_tu = u*tu
#     v_tv = v*tv
    
#     rt = SVector((u_tu-tu + v_tv,
#                  u_tu     + v_tv-tv,
#                  u_tu     + v_tv ))

#     return (value=rt[i]*inv_j, divergence=d)
# end