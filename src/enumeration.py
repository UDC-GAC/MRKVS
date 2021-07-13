# Copyright 2021 Marcos Horro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from intrinsics import Mem, Intrinsic, IntrinsicsList, MemList

loads = [
    Intrinsic(
        "_mm_load_ss",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [Mem("p", "-1"), Mem("p", "-1"), Mem("p", "-1"), Mem("p", "N+0")],
    ),
    Intrinsic(
        "_mm_loadu_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
    ),
    Intrinsic(
        "_mm_load_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        [
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
        True,
    ),
    Intrinsic(
        "_mm256_loadu_ps",
        [Mem("p", "N")],
        "__m256",
        "float",
        256,
        [
            Mem("p", "N+7"),
            Mem("p", "N+6"),
            Mem("p", "N+5"),
            Mem("p", "N+4"),
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
    ),
    Intrinsic(
        "_mm256_load_ps",
        [Mem("p", "N")],
        "__m256",
        "float",
        256,
        [
            Mem("p", "N+7"),
            Mem("p", "N+6"),
            Mem("p", "N+5"),
            Mem("p", "N+4"),
            Mem("p", "N+3"),
            Mem("p", "N+2"),
            Mem("p", "N+1"),
            Mem("p", "N+0"),
        ],
        "AVX2",
        True,
    ),
]


class Vector:
    def __eq__(self, other):
        if len(self.elems) != len(other.elems):
            return False
        for i in range(len(self.elems)):
            if self.elems[i] != other.elems[i]:
                return False
        return True

    def __str__(self):
        return f"V({self.elems})"

    def __repr__(self) -> str:
        return f"{self.__str__()}"

    def __hash__(self):
        return hash(tuple(self.elems))

    def __init__(self, elems: list):
        self.elems = elems


class Program:
    # def compute_output(self):
    #     for ins in self.instructions:

    def __init__(self, instructions: dict):
        self.instructions = instructions
        self.output = self.compute_output()


# NOTE: I think this is permutation with repetition, i.e. if input vector are
# size n and output vector is size n, total number is (n+n)**n
# TODO: this should be generated for any size n
size = 4
A = [f"a{i}" for i in range(size)]
B = [f"b{i}" for i in range(size)]
C = A + B
C.reverse()
import itertools as it

outputs = [Vector(list(comb)) for comb in it.product(C, repeat=size)]


def shuffle(a, b, nelems=4, bits=32) -> dict:
    d = {}
    for i in range(256):
        new_key = f"shuffle_{nelems*bits}_{i}"
        output = []
        for j in range(nelems):
            mask = (i >> 2 * j) & 0x3
            mask += int(j / 4) * 4
            output.append(b[mask] if (j % 4) > 1 else a[mask])
        output.reverse()
        d.update({Vector(output): new_key})
    return d


def _mm_shuffle_ps(a, b) -> dict:
    return shuffle(a, b, 4, 32)


def _mm256_shuffle_ps(a, b) -> dict:
    return shuffle(a, b, 8, 32)


def blend(a, b, elems=4, bits=32) -> dict:
    d = {}
    for i in range(2 ** elems):
        new_key = f"blend_{bits*elems}_{i}"
        output = []
        for j in range(elems):
            mask = (i >> j) & 0x1
            output.append(b[j] if mask else a[j])
        output.reverse()
        d.update({Vector(output): new_key})
    return d


def _mm_blend_ps(a, b) -> dict:
    return blend(a, b, 4)


def _mm256_blend_ps(a, b) -> dict:
    return blend(a, b, 8)


def permutevar(a, elems=4, bits=32) -> dict:
    d = {}
    for i in range(4 ** elems):
        new_key = f"permutevar_{elems*bits}_{i}"
        output = []
        for j in range(elems):
            mask = (i >> 2 * j) & 0x3
            offset = int(j / 4)
            output.append(a[mask + 4 * offset])
        output.reverse()
        d.update({Vector(output): new_key})
    return d


def _mm_permutevar_ps(a) -> dict:
    return permutevar(a, 4, 32)


def _mm256_permutevar_ps(a: list) -> dict:
    return permutevar(a, 8, 32)


def get_permutevar8x32_index(n: int) -> str:
    mask = "_mm256_set_epi32("
    for i in range(7, -1, -1):
        val = (n >> 3 * i) & 0x7
        mask += f"{val},"
    return mask[:-1] + ")"


def _mm256_permutevar8x32_ps(a: list) -> dict:
    d = {}
    for i in range(8 ** 8):
        new_key = f"permutevar8x32_256_{i}"
        output = []
        for j in range(elems):
            mask = (i >> 3 * j) & 0x7
            output.append(a[mask])
        output.reverse()
        d.update({Vector(output): new_key})
    return d


size = 8
A = [f"a{i}" for i in range(size)]
B = [f"b{i}" for i in range(size)]
# FIXME: this should and must be generated once and have a full map
instructions = {}
# instructions.update(_mm_shuffle_ps(A, B))
# instructions.update(_mm_blend_ps(A, B))
# instructions.update(_mm_permutevar_ps(A))
# instructions.update(_mm256_blend_ps(A, _mm256_permutevar_ps(B)))

# n = set()
for p in _mm256_permutevar_ps(B):
    count = int(_mm256_permutevar_ps(B)[p].split("_")[-1])
    if count >= 4096:
        break
    print(count)
    with open(f"brute_force/kernel_avx_1perm_1blend_{count}.c", "w") as f:
        kernel = ""
        mask_perm = get_permutevar8x32_index(count)
        # for mask_blend in range(len(_mm256_blend_ps(A, p.elems))):
        kernel += f"    tmp256 = _mm256_blend_ps(tmp1, _mm256_permutevar8x32_ps(tmp0, {mask_perm}), 0x0);\n"
        kernel += "    DO_NOT_TOUCH(tmp0);\n"
        kernel += "    DO_NOT_TOUCH(tmp1);\n"
        f.write(kernel)

# print(n)
# target_swizzling = Vector(["b3", "b3", "a3", "a3"])

# for ins in instructions:
#     if instructions[ins] == target_swizzling:
#         print(f"Found -> {ins} for {target_swizzling}")

# target_swizzling = Vector(["a3", "a3", "a3", "a3"])

# for ins in instructions:
#     if instructions[ins] == target_swizzling:
#         print(f"Found -> {ins} for {target_swizzling}")

# target_swizzling = Vector(["a3", "a2", "a1", "a0"])

# for ins in instructions:
#     if instructions[ins] == target_swizzling:
#         print(f"Found -> {ins} for {target_swizzling}")
