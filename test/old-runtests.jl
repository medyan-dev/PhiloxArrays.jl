using PhiloxArrays
using Test
using Random123: philox
using Statistics
using LinearAlgebra
using CUDA
using StaticArrays
using OffsetArrays
using FFTW

#=
CUDA calculation of IFFT of conjugate symmetric, complex gaussian,
three component, k-space, (N[1]/2+1)xN[2]xN[3]-dimensional vector field
=#
function calc_real_IFFT_cgvfield(key,counter,N)
    Nirfft = (div(N[1],2)+1,N[2],N[3]) 
    cgvfield = CuArray{SVector{3,Complex{Float32}}}(undef, Nirfft)
    p = PhiloxArrays.ComplexGaussianVectorField(N,UInt64(key),UInt64(counter))
    map!(cgvfield,CartesianIndices(Nirfft)) do k
        kp = CartesianIndex(mod.(N .- Tuple(k) .+ 1,N) .+ 1)
        inv(sqrt(Float32(2)))*(p[k] + conj(p[kp]))
    end
    P = plan_irfft(reinterpret(reshape,ComplexF32,cgvfield),N[1],(2,3,4))
    return P*reinterpret(reshape,ComplexF32,cgvfield)
end

#=
Generate collection of samples of the vector field
=#
function sample_field(nsample,key_init,ctr_init,N)
    cgvfield = reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField(N,UInt64(key_init),UInt64(ctr_init)))
    for i in 1:nsample-1
        cgvfield = cat(cgvfield,reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField(N,UInt64(key_init),UInt64(i))),dims=2)
    end
    return cgvfield
end

@testset "PhiloxArrays.jl" begin
    # Write your tests here.
    
    #=
    Tests that the all the real and imaginary parts of all the components of the vector field at all points are unique random numbers
    =#
    @test allunique(reinterpret(Float32, PhiloxArrays.ComplexGaussianVectorField((3,),UInt64(1),UInt64(1))))

    #=
    Test mean and covariance of philox complex vector fields
    =#
    nsample = 10000
    key_init = 1
    ctr_init = 1
    N = (2,)
    cgvfield = sample_field(nsample,key_init,ctr_init,N)
    #test that each part approximately has mean zero
    @test abs.(round.(mean(cgvfield, dims=2))) == zeros(Float32,6*prod(N),1)
    #test that the covariance matrix is approximately the identity matrix
    @test abs.(round.(cov(cgvfield, dims=2))) == Matrix{Float32}(I, 6*prod(N), 6*prod(N))

    #=
    Test GPU accelerated generation of philox vector field has correct symmetry
    =#
    key = 1
    counter = 1
    N = (32,32,32)
    @test isreal(calc_real_IFFT_cgvfield(key,counter,N))
    
    


    
end
