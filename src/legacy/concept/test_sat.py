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

from sat.x86_sat.parse import *

regex = "|".join(
    [
        r"_mm(512|256|)_set_(ps|epi32|epi8)",
        r"_mm(256|)_shuffle_(ps|epi32)",
        r"_mm(256|)_blend_(ps|epi32)",
        r"_mm(256|)_permutevar_ps",
        r"_mm(256|512|)_permute(var|xvar|2f128|var8x32|)_(ps|epi32|epi8)",
    ]
)


def val_to_array(val, elems=8):
    offset_32b = 0xFFFFFFFF
    arr = []
    if type(val) == str:
        val = eval(val)
    for i in range(elems):
        new_val = val >> 32 * i
        arr.append(new_val & offset_32b)
    arr.reverse()
    return arr


mask = Var("mask", "int")
i = Var("i", "int")
index = Var("index", "__m256i")
index128 = Var("index", "__m128i")
intrinsics = parse_whitelist("sat/data-latest.xml", regex=regex)
globals().update(intrinsics)

# values = [19, 18, 17, 16, 3, 2, 1, 0]
# values2 = [19, 18, 16, 17, 3, 2, 0, 1]
# af = _mm256_set_ps(*values)
# af2 = _mm256_set_ps(*values2)
# print(check(af == _mm256_shuffle_ps(af2, af2, i)))

# values = [19, 18, 17, 16, 3, 2, 1, 0]
# values = [16, 16, 16, 16, 0, 0, 0, 0]
# values2 = [18, 19, 17, 16, 2, 3, 0, 1]
# af = _mm256_set_ps(*values)
# af2 = _mm256_set_ps(*values2)
# check(af == _mm256_shuffle_ps(af2, af2, i))
# check(af == _mm256_permutevar_ps(af2, index))

# # nesting things...
# check(af == _mm256_permutevar_ps(_mm256_shuffle_ps(af2, af2, i), index))


def get_index_128(mask):
    index = "_mm_set_epi32("
    array = val_to_array(mask, 4)
    for idx in array[:-1]:
        index += f"{idx},"
    index += f"{array[-1]})"
    return index


def write_kernel(indices: list, val0: list, val1: list, n: int, suffix="") -> None:
    kernel = ""
    if len(val0) == 0:
        val0 = [3, 2, 1, 0]
    if len(val1) == 0:
        val1 = [7, 6, 5, 4]
    af0 = _mm_set_ps(*val0)
    af1 = _mm_set_ps(*val1)

    index_p0 = Var("indexp0", "__m128i")
    index_p1 = Var("indexp1", "__m128i")
    mask_blend = Var("mask_blend", "int")
    res = _mm_set_ps(*indices)

    p0 = _mm_permutevar_ps(af0, index_p0)
    p1 = _mm_permutevar_ps(af1, index_p1)
    b = _mm_blend_ps(p0, p1, mask_blend)
    _, d = check(res == b)
    keys_indices = [f"IDX{i}" for i in range(len(indices))]
    keys_indices.reverse()
    keys_indices2 = [f"IDX{i}_2" for i in range(len(indices))]
    keys_indices2.reverse()
    kernel += (
        f"// {dict(zip(keys_indices,indices))} {dict(zip(keys_indices2,indices))}\n"
    )
    if eval(d["mask_blend"]) == 0:
        # emit only permute 1
        kernel += (
            f"    tmp128 = _mm_permutevar_ps(tmp0, {get_index_128(d['indexp0'])});\n"
        )
        kernel += (
            f"    tmp0128 = _mm_permutevar_ps(tmp00, {get_index_128(d['indexp0'])});\n"
        )
        kernel += "    DO_NOT_TOUCH(tmp0);\n"
        kernel += "    DO_NOT_TOUCH(tmp00);\n"
    elif eval(d["mask_blend"]) == 0xF:
        # emit only permute 2
        kernel += (
            f"    tmp128 = _mm_permutevar_ps(tmp1, {get_index_128(d['indexp1'])});\n"
        )
        kernel += (
            f"    tmp0128 = _mm_permutevar_ps(tmp01, {get_index_128(d['indexp1'])});\n"
        )
        kernel += "    DO_NOT_TOUCH(tmp1);\n"
        kernel += "    DO_NOT_TOUCH(tmp01);\n"
    else:
        # emit everything
        kernel += (
            f"    tmp0 = _mm_permutevar_ps(tmp0, {get_index_128(d['indexp0'])});\n"
        )
        kernel += (
            f"    tmp00 = _mm_permutevar_ps(tmp00, {get_index_128(d['indexp0'])});\n"
        )
        kernel += (
            f"    tmp1 = _mm_permutevar_ps(tmp1, {get_index_128(d['indexp1'])});\n"
        )
        kernel += (
            f"    tmp01 = _mm_permutevar_ps(tmp01, {get_index_128(d['indexp1'])});\n"
        )
        kernel += f"    tmp128 = _mm_blend_ps(tmp0, tmp1, {d['mask_blend']});\n"
        kernel += f"    tmp0128 = _mm_blend_ps(tmp00, tmp01, {d['mask_blend']});\n"
        kernel += "    DO_NOT_TOUCH(tmp0);\n"
        kernel += "    DO_NOT_TOUCH(tmp00);\n"
        kernel += "    DO_NOT_TOUCH(tmp1);\n"
        kernel += "    DO_NOT_TOUCH(tmp01);\n"

    with open(f"brute_force/kernel_2reg_{n}{suffix}.c", "w") as f:
        f.write(kernel)


import itertools as it

n = 0
val0 = [3, 2, 1, 0]
val1 = [11, 10, 9, 8]
res = val1 + val0
res.reverse()

import multiprocessing as mp

for comb in it.product(res, repeat=4):
    # write_kernel(list(comb), val0, val1, n, "_2cl")
    write_kernel(list(comb), val0, val1, n)
    n += 1
