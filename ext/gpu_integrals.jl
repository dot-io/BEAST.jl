
# function (igd::Integrand{T})(u,v,i::Int,j::Int) where {T} 

#         x = neighborhood(igd.test_chart,u)
#         y = neighborhood(igd.trial_chart,v)
        
#         f = igd.local_test_space(x,i)
#         g = igd.local_trial_space(y,j)

#         jacobian(x) * jacobian(y) * igd(x,y,f,g)
# end


# struct GPUPulledBackIntegrand{I,C1,C2}
#     igd::I
#     chart1::C1
#     chart2::C2
#     i::Int
#     j::Int
# end

# function (pulled::GPUPulledBackIntegrand)(u,v)
#     # In general I think a Jacobian determinant needs to be included. For Simplical and
#     # Quadrilateral charts this is not needed because they are 1.

#     #Hack neighborhoods calls, actuall expecting baricentric coordinates.
#     #Works for pullback for for now.
#     x = neighborhood(pulled.igd.test_chart,cartesian(pulled.chart1,u))
#     y = neighborhood(pulled.igd.trial_chart,cartesian(pulled.chart2,v))

#     f = pulled.igd.local_test_space(x,pulled.i)
#     g = pulled.igd.local_trial_space(y,pulled.j)

#     return jacobian(x) * jacobian(y) * pulled.igd(x,y,f,g)
# end

# function pulledback_integrand(igd,
#     I, chart1, i,
#     J, chart2, j)

#     dom1 = domain(chart1)
#     dom2 = domain(chart2)

#     ichart1 = CompScienceMeshes.permute_vertices(dom1, I)
#     ichart2 = CompScienceMeshes.permute_vertices(dom2, J)

#     GPUPulledBackIntegrand(igd, ichart1, ichart2, i, j)
# end 


function _integrands_gen(::Type{U}, ::Type{V}) where {U<:NamedTuple, V<:NamedTuple} 
    ex = :(f(a, b))
    
    return ex
end

@generated function _integrands(f, a::NamedTuple{T}, b::NamedTuple{S}) where {T,S}
     ex = _integrands_gen(a,b)
    return ex
end