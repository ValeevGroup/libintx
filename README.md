# LibintX

LibintX is a library for accelerated evaluation of molecular integrals of many-body operators over Gaussian atomic orbitals. The primary purpose of LibintX is to enable efficient evaluation of 2-body operators in Gaussian AO integrals on accelerated architectures like the CUDA-capable graphical processing units (GPU). However, it can also be used on conventional/central processing units (CPUs).

Until version 1.0 this will remain EXPERIMENTAL code development; expect things to break and APIs to change.  https://github.com/ValeevGroup/libintx/ is the public mirror of the private development repo.  Don't make PRs against the public mirror; if you wish to collaborate send us a request.

# Installation

## Prerequisites
- CMake
- A C++ compiler with support for the 2017 C++ standard ([the list of compilers with partial or full support for C++17](https://en.cppreference.com/w/cpp/compiler_support/17))
- CUDA toolkit, version 11 or higher (optional)

Other CMake parameters:
- LIBINTX_MAX_K - maximum primitives
- LIBINTX_MAX_L - maximum angular momentum
- LIBINTX_MAX_X - maximum auxillary angular momentum
- LIBINTX_CUDA_MAX_SHMEM - maximum CUDA shared memory per *threadblock*

## Building
- configure: \
    `cd libintx` \
    `cmake -B ./build` \
    `cd ./build` \
  with cuda: \
    `cmake -DLIBINTX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 .` \
  with libint2: \
    `cmake -DLIBINTX_LIBINT2=ON .`
- build: \
    `cmake --build .`
- 3-center tests and benchmarks: \
    `cmake --build . --target libintx.cuda.md3.test` \
    `./tests/libintx.cuda.md3.test`
    `cmake --build . --target libintx.cuda.md3.benchmarks` \
    `./tests/libintx.cuda.md3.benchmarks`
- 4-center tests and benchmarks: \
    `cmake --build . --target libintx.cuda.md4.test` \
    `./tests/libintx.cuda.md4.test`
    `cmake --build . --target libintx.cuda.md4.benchmarks` \
    `./tests/libintx.cuda.md4.benchmarks`

# Python
- Configure with LIBINTX_PYTHON=TRUE
- Build `libintx-python` target

# Using
Still work in progress.  Read through test programs and/or contact Andrey, asadchev@gmail.com

# Developers
LibintX is developed by the [Valeev Group](http://valeevgroup.github.io/) at [Virginia Tech](http://www.vt.edu).

# License

LibintX is freely available under the terms of the LGPL v3+ licence. See the included LICENSE file for details. If you are interested in using LibintX under different licensing terms, please contact us.

# How to Cite

See the enclosed LICENSE file.

# Acknowledgements

Development of LibintX is made possible by the support provided by the Department of Energy Exascale Computing Project ([NWChemEx subproject](https://github.com/NWChemEx-Project)).
