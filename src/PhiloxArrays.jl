module PhiloxArrays
using Random123: philox
using StaticArrays
export RawPhiloxArray

# Write your package code here.

"""
Struct for generating N array of arrays of 4 UInt32 numbers,
where N is a tuple with the dimensions of the array
"""
struct RawPhiloxArray{N} <: AbstractArray{SVector{4,UInt32},N}
    size::NTuple{N,Int64}
    key::UInt64
    ctr1::UInt64
end

"""
Generate 4 UInt32 numbers
"""
function p(key::UInt64, ctr0::UInt64, ctr1::UInt64)::SVector{4,UInt32}
    philox((key%UInt32,(key>>32)%UInt32), (ctr0%UInt32,(ctr0>>32)%UInt32,ctr1%UInt32,(ctr1>>32)%UInt32), Val(10))
end

"""
Define methods for struct
"""
Base.size(P::RawPhiloxArray) = P.size
Base.IndexStyle(::Type{<:RawPhiloxArray}) = IndexLinear()
Base.getindex(P::RawPhiloxArray, i::Integer) = p(P.key, i%UInt64, P.ctr1)

"""
Convert UInt32 into a float of type Ftype in [-1,1]
"""
function uneg11(Ftype::Type{<:Real},in::UInt32)::Ftype
    fma(Ftype(in%Int32),Ftype(2)^(-31),Ftype(2)^(-32))
end

"""
Convert UInt64 into a float of type Ftype in [-1,1]
"""
function uneg11(Ftype::Type{<:Real},in::UInt64)::Ftype
    fma(Ftype(in%Int32),Ftype(2)^(-63),Ftype(2)^(-64))
end

"""
Convert UInt32 into a float of type Ftype in (0,1]
"""
function u01(Ftype::Type{<:Real},in::UInt32)::Ftype
    fma(in,Ftype(2)^(-32),Ftype(2)^(-33))
end

"""
Convert UInt64 into a float of type Ftype in (0,1]
"""
function u01(Ftype::Type{<:Real},in::UInt64)::Ftype
    fma(in,Ftype(2)^(-64),Ftype(2)^(-65))
end

"""
Takes in two UInt32 philox numbers and outputs two normally distrubuted floats of type Ftype
"""
function boxmuller(Ftype::Type{<:Real},u1::UInt32,u2::UInt32)::SVector{2,Ftype}
    sqrt(-2*log(u01(Ftype,u2))).*sincospi(uneg11(Ftype,u1))
end

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
Takes two normally distributed numbers and converts into a single complex normally distributed number
"""
function complex_normal()
end

"""
Takes two raw philox arrays and outputs a complex, normally distributed vector field
"""
function complex_normal_field_array()
end

end

