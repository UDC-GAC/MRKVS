# MRKVS - Mega-Random Kernel Vector SMT

Automatic generation of intrinsics-like C code for any random packing
combination given a subset of instructions.

The idea of this tools is to prune the large exploration space created by the combination of instructions for generating new formulas. The SMT logic of the system is meant for correctness purposes and to check the feasibility of a set of instructions and the values of the masks (if any).

This tool is presented in the paper ["Custom High-Performance Vector Code Generation for
Data-Specific Sparse Computations" in Proceedings of the 31st International Conference on Parallel Architectures and Compilation Techniques (PACT), Chicago, IL, 2022](https://gac.udc.es/~gabriel/files/pact22-final139.pdf). This tool was built to help building optimal vector packing recipes for [MAVETH (Multi-dimensional Array C-compiler for VEctorizing Tensors in HPC)](https://github.com/UDC-GAC/MACVETH).

## Installation and requirements

This code is written using Python 3.
[Z3 SMT framework](https://github.com/Z3Prover/z3) is required. You can find
there the instructions for installing it and more documentation. For the Python
binding, you can install it (Debian-like systems) as follows:

```bash
pip install z3-solver
```

Also, it depends on [zwegner/x86-sat](https://github.com/zwegner/x86-sat/), which is included as a submodule, so you can clone this repo as:

```bash
git clone --recurse-submodules -j8 git://github.com/UDC-GAC/MRKVS.git
```

## Getting started

For executing:

```bash
python src/main.py [<start> <end>] 
```

Where <start> is the lowest number of elements to pack, and <end> the highest, e.g., `2 4` would generate the vector packings for 2, 3, and 4 elements.

## Limitations

Currently only considering floats and doubles using vector intrinsics for x86 (up to AVX2, not considering AVX-512).

## Tests

You can find some tests under `tests/` directory.
