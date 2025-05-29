using Test
using LinearAlgebra
using FFTW
using PhiloxArrays
using Statistics
using HypothesisTests
using Distributions

"""
Function for jittering data to remove ties so KS test can be used
"""
function jitter(a::Array)
    for i in setdiff(collect(eachindex(a)),unique(i -> a[i], eachindex(a)))
        a[i] = nextfloat(a[i])
    end
    return a
end

@testset "ConjSymRandNArray Tests" begin
    @testset "Construction" begin
        # Test basic construction
        A = ConjSymRandNArray{Float32}((3,4,5), UInt64(7), UInt64(11))
        @test size(A) == (3,4,5)
    end
    
    @testset "Symmetry Properties" begin
        # Test conjugate symmetry property: A[i,j] == conj(A[j,i])
        A = ConjSymRandNArray{Float32}((3,4,5), UInt64(7), UInt64(11))
        for kvec in CartesianIndices(A)
            kvecp = CartesianIndex(mod.(A.size .- Tuple(kvec) .+ 1, A.size) .+ 1)
            @test A[kvecp] == conj(A[kvec])
        end
        for i in 1:3
            @test isreal(ifft(map(x->x[i], A)))
        end
    end

    @testset "Special Cases" begin
        # Test empty array
        A = ConjSymRandNArray{Float32}((0,0,0), UInt64(7), UInt64(11))

        # Test 1×1×1 array
        A = ConjSymRandNArray{Float32}((1,1,1), UInt64(7), UInt64(11))
        @test A[1,1,1] == conj(A[1,1,1])
    end

    @testset "Array has the correct distribution" begin
        nsize = (4,5,6)
        normal_samples = Float32[]
        special_samples = Float32[]
        nsamples = 1000
        for i in 1:nsamples
            A = ConjSymRandNArray{Float32}(nsize, UInt64(7), UInt64(i))
            for kvec in CartesianIndices(A)
                kvecp = CartesianIndex(mod.(A.size .- Tuple(kvec) .+ 1, A.size) .+ 1)
                if kvec == kvecp
                    append!(special_samples, vec(reinterpret(reshape, Float32, real.(A[kvec]))))
                elseif kvec > kvecp
                    append!(normal_samples, vec(reinterpret(reshape, Float32, A[kvec])))
                end
            end
        end
        @show length(normal_samples) - length(Set(normal_samples))
        @show length(special_samples) - length(Set(special_samples))
        @show mean(normal_samples)
        @show mean(special_samples)
        @show cov(normal_samples)
        @show cov(special_samples)
        
        #add jittering to remove ties
        @show length(jitter(normal_samples)) - length(Set(jitter(normal_samples)))
        @show length(jitter(special_samples)) - length(Set(jitter(special_samples)))
        d_normal = Normal(0.0,1.0)
        d_special = Normal(0.0,sqrt(2.0))
        @show ExactOneSampleKSTest(jitter(normal_samples), d_normal)
        @show ExactOneSampleKSTest(jitter(special_samples), d_special)

    end


    #Float64 math tests:

    @testset "Construction" begin
        # Test basic construction
        A = ConjSymRandNArray{Float64}((3,4,5), UInt64(7), UInt64(11))
        @test size(A) == (3,4,5)
    end
    
    @testset "Symmetry Properties" begin
        # Test conjugate symmetry property: A[i,j] == conj(A[j,i])
        A = ConjSymRandNArray{Float64}((3,4,5), UInt64(7), UInt64(11))
        for kvec in CartesianIndices(A)
            kvecp = CartesianIndex(mod.(A.size .- Tuple(kvec) .+ 1, A.size) .+ 1)
            @test A[kvecp] == conj(A[kvec])
        end
        for i in 1:3
            @test isreal(ifft(map(x->x[i], A)))
        end
    end

    @testset "Special Cases" begin
        # Test empty array
        A = ConjSymRandNArray{Float64}((0,0,0), UInt64(7), UInt64(11))

        # Test 1×1×1 array
        A = ConjSymRandNArray{Float64}((1,1,1), UInt64(7), UInt64(11))
        @test A[1,1,1] == conj(A[1,1,1])
    end

    @testset "Array has the correct distribution" begin
        nsize = (4,5,6)
        normal_samples = Float64[]
        special_samples = Float64[]
        nsamples = 1000
        for i in 1:nsamples
            A = ConjSymRandNArray{Float64}(nsize, UInt64(7), UInt64(i))
            for kvec in CartesianIndices(A)
                kvecp = CartesianIndex(mod.(A.size .- Tuple(kvec) .+ 1, A.size) .+ 1)
                if kvec == kvecp
                    append!(special_samples, vec(reinterpret(reshape, Float64, real.(A[kvec]))))
                elseif kvec > kvecp
                    append!(normal_samples, vec(reinterpret(reshape, Float64, A[kvec])))
                end
            end
        end
        @show length(normal_samples) - length(Set(normal_samples))
        @show length(special_samples) - length(Set(special_samples))
        @show mean(normal_samples)
        @show mean(special_samples)
        @show cov(normal_samples)
        @show cov(special_samples)
        
        #add jittering to remove ties
        @show length(jitter(normal_samples)) - length(Set(jitter(normal_samples)))
        @show length(jitter(special_samples)) - length(Set(jitter(special_samples)))
        d_normal = Normal(0.0,1.0)
        d_special = Normal(0.0,sqrt(2.0))
        @show ExactOneSampleKSTest(jitter(normal_samples), d_normal)
        @show ExactOneSampleKSTest(jitter(special_samples), d_special)

    end

end
