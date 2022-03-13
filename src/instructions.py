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

from z3 import *
from sat.x86_sat.parse import *
from sat.x86_sat.evaluate import ArgsType, InsType, ArgsTypeException

# Custom defined instructions for MACVETH
_mm_load_ss = ParseOperation(
    "_mm_load_ss",
    [("a", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := 0; dst[95:64] := 0; dst[63:32] := 0; dst[31:0] := a;",
)
_mm_load_ss.instype = InsType.LOAD
_mm_loadu_ps = ParseOperation(
    "_mm_loadu_ps",
    [("a", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := a+3; dst[95:64] := a+2; dst[63:32] := a+1; dst[31:0] := a;",
)
_mm_loadu_ps.instype = InsType.LOAD
_mm_loadl_pi = ParseOperation(
    "_mm_loadl_pi",
    [("a", "__m128", ""), ("mem", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := a[127:96]; dst[95:64] := a[95:64]; dst[63:32] := mem+1; dst[31:0] := mem;",
)
_mm_loadl_pi.instype = InsType.LOAD
_mm_loadh_pi = ParseOperation(
    "_mm_loadh_pi",
    [("a", "__m128", ""), ("mem", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := mem+1; dst[95:64] := mem; dst[63:32] := a[63:32]; dst[31:0] := a[31:0];",
)
_mm_loadh_pi.instype = InsType.LOAD
_mm_maskload_ps = ParseOperation(
    "_mm_maskload_ps",
    [("a", "float", ""), ("mask", "__m128i", "")],
    "dst",
    "__m128",
    """
FOR j := 0 to 3
i := j*32
IF mask[i+31]
    dst[i+31:i] := a+j
ELSE
    dst[i+31:i] := 0
FI
ENDFOR
dst[MAX:128] := 0
""",
)
_mm_maskload_ps.instype = InsType.LOAD
_mm_maskload_ps.maskvec = True
_mm256_loadu_ps = ParseOperation(
    "_mm256_loadu_ps",
    [("a", "float", "")],
    "dst",
    "__m256",
    "dst[255:224] := a+7; dst[223:192] := a+6; dst[191:160] := a+5; dst[159:128] := a+4; dst[127:96] := a+3; dst[95:64] := a+2; dst[63:32] := a+1; dst[31:0] := a;",
)
_mm256_loadu_ps.instype = InsType.LOAD
_mm256_maskload_ps = ParseOperation(
    "_mm256_maskload_ps",
    [("a", "float", ""), ("mask", "__m256i", "")],
    "dst",
    "__m256",
    """
FOR j := 0 to 7
i := j*32
IF mask[i+31]
    dst[i+31:i] := a+j
ELSE
    dst[i+31:i] := 0
FI
ENDFOR
dst[MAX:256] := 0
""",
)
_mm256_maskload_ps.instype = InsType.LOAD
_mm256_maskload_ps.maskvec = True

load_instructions = [
    globals()[i] for i in dir() if i.startswith("_mm") or i.startswith("_mv")
]

_mv_insert_mem_ps = ParseOperation(
    "_mv_insert_mem_ps",
    [("a", "__m128", ""), ("b", "float", ""), ("imm8", "const int", "")],
    "dst",
    "__m128",
    """
tmp2[127:0] := a[127:0]
CASE (imm8[7:6]) OF
0: tmp1[31:0] := b
1: tmp1[31:0] := b
2: tmp1[31:0] := b
3: tmp1[31:0] := b
ESAC
CASE (imm8[5:4]) OF
0: tmp2[31:0] := tmp1[31:0]
1: tmp2[63:32] := tmp1[31:0]
2: tmp2[95:64] := tmp1[31:0]
3: tmp2[127:96] := tmp1[31:0]
ESAC
FOR j := 0 to 3
	i := j*32
	IF imm8[j%8]
		dst[i+31:i] := 0
	ELSE
		dst[i+31:i] := tmp2[i+31:i]
	FI
ENDFOR
    """,
)
_mv_insert_mem_ps.instype = InsType.INSERT

_mv256_blend_mem_ps = ParseOperation(
    "_mv256_blend_mem_ps",
    [("a", "__m256", ""), ("b", "float", ""), ("imm8", "const int", "")],
    "dst",
    "__m256",
    """
FOR j := 0 to 7
	i := j*32
	IF imm8[j]
		dst[i+31:i] := b+j
	ELSE
		dst[i+31:i] := a[i+31:i]
	FI
ENDFOR
dst[MAX:256] := 0
    """,
)
_mv256_blend_mem_ps.instype = InsType.BLEND

_mv_blend_mem_ps = ParseOperation(
    "_mv_blend_mem_ps",
    [("a", "__m128", ""), ("b", "float", ""), ("imm8", "const int", "")],
    "dst",
    "__m128",
    """
FOR j := 0 to 3
	i := j*32
	IF imm8[j]
		dst[i+31:i] := b+j
	ELSE
		dst[i+31:i] := a[i+31:i]
	FI
ENDFOR
    """,
)
_mv_blend_mem_ps.instype = InsType.BLEND


insert_blend_instructions = [
    globals()[i] for i in dir() if "insert" in i or "blend" in i
]

# Update intrinsics list for iterate
full_custom_ops_list = [
    globals()[i] for i in dir() if i.startswith("_mm") or i.startswith("_mv")
]

regex = "|".join(
    [
        r"_mm_insert_ps",
        r"_mm256_insertf128_ps",
        r"_mm_move(hl|lh)_ps",
        r"_mm(256|)_blend_ps",
    ]
)
intrinsics = parse_whitelist("sat/data-latest.xml", regex=regex)

full_instruction_list = full_custom_ops_list + list(intrinsics.values())


def type_instruction(ins):
    ins.argstype = ArgsType.ALLREG
    if "load" in ins.name:
        ins.instype = InsType.LOAD
    elif "blend" in ins.name:
        ins.instype = InsType.BLEND
    elif "insert" in ins.name:
        ins.instype = InsType.INSERT
    elif "mov" in ins.name:
        ins.instype = InsType.MOVE
    else:
        ins.instype = InsType.ANY
    if "mem" in ins.name:
        ins.argstype = ArgsType.REG_MEM
    ins.hasimm = False
    if "imm" in ins.params[-1].name:
        ins.hasimm = True
    ins.maskvec = False
    if "mask" in ins.params[-1].name:
        ins.maskvec = True
    ins.needsregister = False
    if "loadh" in ins.name or "loadl" in ins.name:
        ins.needsregister = True
    return ins


full_instruction_list = [type_instruction(x) for x in full_instruction_list]

move_instruction_list = [
    ins for ins in full_instruction_list if ins.instype == InsType.MOVE
]

intrinsics = parse_whitelist("sat/data-latest.xml", regex=r"_mm(256|)_set_ps")

# Wrappers
set_4_float_elements = intrinsics["_mm_set_ps"]
set_8_float_elements = intrinsics["_mm256_set_ps"]
intrinsics = parse_whitelist("sat/data-latest.xml", regex="_mm256_set_m128$")
set_hi_lo = intrinsics["_mm256_set_m128"]
