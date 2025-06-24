module PhiloxArrays
using Random123: philox
using StaticArrays
export ConjSymRandNArray

"""
    RawPhiloxArray{N} <: AbstractArray{SVector{4,UInt32},N}

`N`-dimensional random array with elements of type `SVector{4,UInt32}`.
"""
struct RawPhiloxArray{N} <: AbstractArray{SVector{4,UInt32},N}
    size::NTuple{N,Int64}
    key::UInt64
    ctr1::UInt64
end

function p(key::UInt64, ctr0::UInt64, ctr1::UInt64)::SVector{4,UInt32}
    philox((key%UInt32,(key>>32)%UInt32), (ctr0%UInt32,(ctr0>>32)%UInt32,ctr1%UInt32,(ctr1>>32)%UInt32), Val(10))
end

Base.size(P::RawPhiloxArray) = P.size
Base.IndexStyle(::Type{<:RawPhiloxArray}) = IndexLinear()
Base.getindex(P::RawPhiloxArray, i::Integer) = p(P.key, i%UInt64, P.ctr1)

"""
Convert UInt32 into a float of type Ftype in [-1,1]
Ported from Random123:
https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/uniform.hpp#L206
"""
function uneg11(Ftype::Type{<:Real}, in::UInt32)::Ftype
    fma(Ftype(in%Int32), Ftype(2)^(-31), Ftype(2)^(-32))
end

"""
Convert UInt64 into a float of type Ftype in [-1,1]
Ported from Random123:
https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/uniform.hpp#L206
"""
function uneg11(Ftype::Type{<:Real}, in::UInt64)::Ftype
    fma(Ftype(in%Int64), Ftype(2)^(-63), Ftype(2)^(-64))
end

"""
Convert UInt32 into a float of type Ftype in (0,1]
Ported from Random123:
https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/uniform.hpp#L175
"""
function u01(Ftype::Type{<:Real}, in::UInt32)::Ftype
    fma(Ftype(in), Ftype(2)^(-32), Ftype(2)^(-33))
end

"""
Convert UInt64 into a float of type Ftype in (0,1]
Ported from Random123:
https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/uniform.hpp#L175
"""
function u01(Ftype::Type{<:Real}, in::UInt64)::Ftype
    fma(Ftype(in), Ftype(2)^(-64), Ftype(2)^(-65))
end

"""
Takes in two uniformly distributed UInt32 and outputs two normally distributed floats of type Ftype.
Ported from Random123:
https://github.com/DEShawResearch/random123/blob/v1.14.0/include/Random123/boxmuller.hpp#L113
"""
function boxmuller(Ftype::Type{<:Real}, u1::T, u2::T)::SVector{2,Ftype} where {T <: Union{UInt32, UInt64}}
    sqrt(-2*log(u01(Ftype,u2))).*sincospi(uneg11(Ftype,u1))
end

"""
Struct for generating N array of 3D vectors whose elements are complex, normally distributed
pseudorandom, complex Float32 numbers. N is a tuple with the dimensions of the array
"""
struct ComplexGaussianVectorField{N} <: AbstractArray{SVector{3,Complex{Float32}},N}
    size::NTuple{N,Int64}
    key::UInt64
    ctr1::UInt64
end

Base.size(G::ComplexGaussianVectorField) = G.size
Base.IndexStyle(::Type{<:ComplexGaussianVectorField}) = IndexLinear()

function Base.getindex(G::ComplexGaussianVectorField, i::Integer)::SVector{3,Complex{Float32}}
    p1 = p(G.key, (i%UInt64)<<1, G.ctr1)
    p2 = p(G.key, (i%UInt64)<<1 | UInt64(1), G.ctr1)
    g1 = boxmuller(Float32,p1[1],p1[2])
    g2 = boxmuller(Float32,p1[3],p1[4])
    g3 = boxmuller(Float32,p2[1],p2[2])
    SA[g1[1]+g1[2]im, g2[1]+g2[2]im, g3[1]+g3[2]im]
end

"""
Struct for generating N array of 3D vectors whose elements are complex, normally distributed
pseudorandom, complex Float32 numbers. N is a tuple with the dimensions of the array
"""
struct ConjSymRandNArray{F, N} <: AbstractArray{SVector{N, Complex{F}}, N}
    size::NTuple{N, Int64}
    key::UInt64
    ctr1::UInt64
end

ConjSymRandNArray{F}(size::NTuple{N, Integer}, key, ctr1) where {N, F} = ConjSymRandNArray{F, N}(size, key, ctr1)

Base.size(G::ConjSymRandNArray) = G.size

function Base.getindex(G::ConjSymRandNArray{Float32, 3}, kvec::Vararg{Int,3})::SVector{3, Complex{Float32}}
    kvecp = mod.(G.size .- Tuple(kvec) .+ 1, G.size) .+ 1
    i = 1 + (kvec[1]-1) + (kvec[2]-1)*G.size[1] + (kvec[3]-1)*G.size[1]*G.size[2]
    ip = 1 + (kvecp[1]-1) + (kvecp[2]-1)*G.size[1] + (kvecp[3]-1)*G.size[1]*G.size[2]
    p1 = p(G.key, (i%UInt64)<<1, G.ctr1)
    p2 = p(G.key, (i%UInt64)<<1 | UInt64(1), G.ctr1)
    g1 = boxmuller(Float32,p1[1],p1[2])
    g2 = boxmuller(Float32,p1[3],p1[4])
    g3 = boxmuller(Float32,p2[1],p2[2])
    p1p = p(G.key, (ip%UInt64)<<1, G.ctr1)
    p2p = p(G.key, (ip%UInt64)<<1 | UInt64(1), G.ctr1)
    g1p = boxmuller(Float32,p1p[1],p1p[2])
    g2p = boxmuller(Float32,p1p[3],p1p[4])
    g3p = boxmuller(Float32,p2p[1],p2p[2])
    inv(sqrt(Float32(2))) * SA[
        g1[1] + g1p[1] + (g1[2] - g1p[2])im,
        g2[1] + g2p[1] + (g2[2] - g2p[2])im,
        g3[1] + g3p[1] + (g3[2] - g3p[2])im,
    ]
end

function Base.getindex(G::ConjSymRandNArray{Float64, 3}, kvec::Vararg{Int,3})::SVector{3, Complex{Float64}}
    kvecp = mod.(G.size .- Tuple(kvec) .+ 1, G.size) .+ 1
    i = 1 + (kvec[1]-1) + (kvec[2]-1)*G.size[1] + (kvec[3]-1)*G.size[1]*G.size[2]
    ip = 1 + (kvecp[1]-1) + (kvecp[2]-1)*G.size[1] + (kvecp[3]-1)*G.size[1]*G.size[2]
    p1 = p(G.key, (i%UInt64)<<2, G.ctr1)
    p2 = p(G.key, (i%UInt64)<<2 | UInt64(1), G.ctr1)
    p3 = p(G.key, (i%UInt64)<<2 | UInt64(2), G.ctr1)
    g1 = boxmuller(Float64,(p1[1]%UInt64)<<32 + p1[2]%UInt64,(p1[3]%UInt64)<<32 + p1[4]%UInt64)
    g2 = boxmuller(Float64,(p2[1]%UInt64)<<32 + p2[2]%UInt64,(p2[3]%UInt64)<<32 + p2[4]%UInt64)
    g3 = boxmuller(Float64,(p3[1]%UInt64)<<32 + p3[2]%UInt64,(p3[3]%UInt64)<<32 + p3[4]%UInt64)
    p1p = p(G.key, (ip%UInt64)<<2, G.ctr1)
    p2p = p(G.key, (ip%UInt64)<<2 | UInt64(1), G.ctr1)
    p3p = p(G.key, (ip%UInt64)<<2 | UInt64(2), G.ctr1)
    g1p = boxmuller(Float64,(p1p[1]%UInt64)<<32 + p1p[2]%UInt64,(p1p[3]%UInt64)<<32 + p1p[4]%UInt64)
    g2p = boxmuller(Float64,(p2p[1]%UInt64)<<32 + p2p[2]%UInt64,(p2p[3]%UInt64)<<32 + p2p[4]%UInt64)
    g3p = boxmuller(Float64,(p3p[1]%UInt64)<<32 + p3p[2]%UInt64,(p3p[3]%UInt64)<<32 + p3p[4]%UInt64)
    inv(sqrt(Float64(2))) * SA[
        g1[1] + g1p[1] + (g1[2] - g1p[2])im,
        g2[1] + g2p[1] + (g2[2] - g2p[2])im,
        g3[1] + g3p[1] + (g3[2] - g3p[2])im,
    ]
end

end
