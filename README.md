# LibintX

LibintX is a library for accelerated evaluation of molecular integrals of many-body operators over Gaussian atomic orbitals. The primary purpose of LibintX is to enable efficient evaluation of 2-body operators in Gaussian AO integrals on accelerated architectures like the CUDA-capable graphical processing units (GPU). However, it can also be used on conventional/central processing units (CPUs).

Until version 1.0 this will remain EXPERIMENTAL code development; expect things to break and APIs to change.  https://github.com/ValeevGroup/libintx/ is the public mirror of the private development repo.  Don't make PRs against the public mirror; if you wish to collaborate send us a request.

# Installation

## Prerequisites
- CMake
- A C++ compiler with support for the 2017 C++ standard ([the list of compilers with partial or full support for C++17](https://en.cppreference.com/w/cpp/compiler_support/17))
- CUDA toolkit, version 11 or higher (optional)

## Building
- configure: `cmake -S /path/to/libintx/top/source/dir -B /path/to/build/dir [-DLIBINTX_ENABLE_CUDA=ON]`
- build: `cmake --build /path/to/build/dir`
- test: `cd /path/to/build/dir && ctest`

# Using

TBC

# Developers
LibintX is developed by the [Valeev Group](http://valeevgroup.github.io/) at [Virginia Tech](http://www.vt.edu).

# License

LibintX is freely available under the terms of the LGPL v3+ licence. See the included LICENSE file for details. If you are interested in using LibintX under different licensing terms, please contact us.

# How to Cite

See the enclosed LICENSE file.

# Acknowledgements

Development of LibintX is made possible by the support provided by the Department of Energy Exascale Computing Project ([NWChemEx subproject](https://github.com/NWChemEx-Project)).
