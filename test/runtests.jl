using PhiloxArrays
using Test
using Random123: philox
using Plots
using BenchmarkTools

@testset "PhiloxArrays.jl" begin
    # Write your tests here.
    function p(key::UInt64,ctr::UInt128)::UInt128
        r = philox(reinterpret(NTuple{2,UInt32},key),reinterpret(NTuple{4,UInt32},ctr),Val(10))
        reinterpret(UInt128,r)
    end
    
    
    #nsamp = 10
    #arr = zeros(UInt64,nsamp)
    #set key
    key = UInt64(1)
    #ctr = UInt128(1)
    #@btime $p($key,$ctr)

    for i in 1:10
        ctr = UInt128(i)
        @btime $p($key,$ctr)
        #r = reinterpret(NTuple{2,UInt64},p(key,ctr))
        #println(r[2])
        #arr[i] = r
    end
    #f = Plots.histogram(arr)
    #savefig(f, "hist_philox.png")
end
