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

from definitions import Intrinsic, Mem, Reg, Imm, TmpVal


def op_blend(vec: Mem, mem: Mem, imm: Imm, pos: int = 0):
    return mem if ((imm >> pos) & 0x1) else vec


def op_insert(vec: Mem, mem: Mem, imm: Imm, pos: int = 0):
    return mem if pos == imm else vec


instructions = [
    Intrinsic(
        "_mm_load_ss",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        Reg([Mem("p", "-1"), Mem("p", "-1"), Mem("p", "-1"), Mem("p", "N+0")]),
    ),
    Intrinsic(
        "_mm_loadu_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        Reg([Mem("p", f"N+{pos}") for pos in range(3, -1, -1)]),
        "AVX2",
    ),
    Intrinsic(
        "_mm_load_ps",
        [Mem("p", "N")],
        "__m128",
        "float",
        128,
        Reg([Mem("p", f"N+{pos}") for pos in range(3, -1, -1)]),
        "AVX2",
        True,
    ),
    Intrinsic(
        "_mm256_loadu_ps",
        [Mem("p", "N")],
        "__m256",
        "float",
        256,
        Reg([Mem("p", f"N+{pos}") for pos in range(7, -1, -1)]),
        "AVX2",
    ),
    Intrinsic(
        "_mm256_load_ps",
        [Mem("p", "N")],
        "__m256",
        "float",
        256,
        Reg([Mem("p", f"N+{pos}") for pos in range(7, -1, -1)]),
        "AVX2",
        True,
    ),
    Intrinsic(
        "__mv256_blend_ps",
        [Reg("vec", 8), Mem("p", "N"), Imm(0b00000000)],
        "__m256",
        "float",
        256,
        Reg(
            [
                [TmpVal(0, abs(pos - 7)), Mem("p", f"N+{pos}"), TmpVal(1, 0), pos]
                for pos in range(7, -1, -1)
            ],
            op_blend,
            True,
        ),
        "AVX2",
        True,
    ),
    Intrinsic(
        "__mv128_blend_ps",
        [Reg("vec", 4), Mem("p", "N"), Imm(0b0000)],
        "__m128",
        "float",
        128,
        Reg(
            [
                [TmpVal(0, abs(pos - 3)), Mem("p", f"N+{pos}"), TmpVal(1, 0), pos]
                for pos in range(3, -1, -1)
            ],
            op_blend,
            True,
        ),
        "AVX2",
        True,
    ),
]
