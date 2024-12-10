using PhiloxArrays
using Test
using Random123: philox
using Statistics
using LinearAlgebra

@testset "PhiloxArrays.jl" begin
    # Write your tests here.
    
    # Tests that the all the real and imaginary parts of all the components of the vector field at all points are unique random numbers
    @test allunique(reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField((3,),UInt64(1),UInt64(1))))

    #generate collection of sample of the vector field
    nsample = 10000
    key_init = 1
    ctr_init = 1
    N = (2,)
    cgvfield = reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField(N,UInt64(key_init),UInt64(ctr_init)))
    for i in 1:nsample-1
        cgvfield = cat(cgvfield,reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField(N,UInt64(key_init),UInt64(i))),dims=2)
    end
    #test that each part approximately has mean zero
    @test abs.(round.(mean(cgvfield, dims=2))) == zeros(Float32,6*prod(N),1)
    #test that the covariance matrix is approximately the identity matrix
    @test abs.(round.(cov(cgvfield, dims=2))) == Matrix{Float32}(I, 6*prod(N), 6*prod(N))
    
end
