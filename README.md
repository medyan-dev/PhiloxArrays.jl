# PhiloxArrays

[![Build Status](https://github.com/medyan-dev/PhiloxArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/medyan-dev/PhiloxArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package uses code ported from the C++ library random123:

https://github.com/DEShawResearch/random123

This is based on the work by Salmon et al. [1].

PhiloxArrays.jl is for sampling pseudorandom, complex, conjugate symmetric, gaussian distributed random matrices. In other words, one can sample discretized complex random fields in k-space, whose inverse Fourier transforms are real discretized random fields.

1. Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel random numbers. Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis, 1–12. https://doi.org/10.1145/2063384.2063405 
