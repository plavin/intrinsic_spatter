# Intrinsic Spatter

This repo contains some toy Intel intrinsic code that is being used to test gather scatter support on modern CPUs. The code requires AVX512F intrinsic support. 

## Building

First, rename either `Makefile.gcc` or `Makefile.icc` to `Makefile` depending on your platform. Then, compile the code with `make`. This will produce an executable named `a.out`. 

## Running

Run the code with `make run` on Cray compute nodes and `./a.out` otherwise.  
