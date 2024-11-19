using PhiloxArrays
using Test
using Random123: philox
#using BenchmarkTools

@testset "PhiloxArrays.jl" begin
    # Write your tests here.
    
    @test allunique(reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField((3,),UInt64(1),UInt64(1))))

end
