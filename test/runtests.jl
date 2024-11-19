using PhiloxArrays
using Test
using Random123: philox
#using BenchmarkTools

@testset "PhiloxArrays.jl" begin
    # Write your tests here.
    
    N = (3,3,3)

    #val1 = UInt32(1729374549)
    #val2 = UInt32(3909089009)
    
    #PhiloxArrays.RawPhiloxArray(N,UInt64(1),UInt64(1))

    #println(PhiloxArrays.boxmuller(Float32,val1,val2))

    #println(PhiloxArrays.ComplexGaussianVectorField(N,UInt64(1),UInt64(1)))

    @test allunique(reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField((3,),UInt64(1),UInt64(1))))

end
