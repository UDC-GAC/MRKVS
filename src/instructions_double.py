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
_mm_load_sd = ParseOperation(
    "_mm_load_sd",
    [("a", "double", "")],
    "dst",
    "__m128d",
    "dst[127:64] := 0; dst[63:0] := a;",
)
_mm_load_sd.instype = InsType.LOAD

_mm_loadu_pd = ParseOperation(
    "_mm_loadu_pd",
    [("a", "double", "")],
    "dst",
    "__m128d",
    "dst[127:64] := a+1; dst[63:0] := a;",
)
_mm_loadu_pd.instype = InsType.LOAD

_mm_loadl_pd = ParseOperation(
    "_mm_loadl_pd",
    [("a", "__m128d", ""), ("mem", "double", "")],
    "dst",
    "__m128d",
    "dst[127:64] := a[127:64]; dst[63:0] := mem;",
)
_mm_loadl_pd.instype = InsType.LOAD

_mm_loadh_pd = ParseOperation(
    "_mm_loadh_pd",
    [("a", "__m128d", ""), ("mem", "double", "")],
    "dst",
    "__m128d",
    "dst[127:64] := mem; dst[63:0] := a[63:0];",
)
_mm_loadh_pd.instype = InsType.LOAD

_mm_maskload_pd = ParseOperation(
    "_mm_maskload_pd",
    [("a", "double", ""), ("mask", "__m128i", "")],
    "dst",
    "__m128d",
    """
FOR j := 0 to 1
i := j*64
IF mask[i+63]
    dst[i+63:i] := a+j
ELSE
    dst[i+63:i] := 0
FI
ENDFOR
dst[MAX:128] := 0
""",
)
_mm_maskload_pd.instype = InsType.LOAD
_mm_maskload_pd.maskvec = True

_mm256_loadu_pd = ParseOperation(
    "_mm256_loadu_pd",
    [("a", "double", "")],
    "dst",
    "__m256d",
    "dst[255:192] := a+3; dst[191:128] := a+2; dst[127:64] := a+1; dst[63:0] := a;",
)
_mm256_loadu_pd.instype = InsType.LOAD

_mm256_maskload_pd = ParseOperation(
    "_mm256_maskload_pd",
    [("a", "double", ""), ("mask", "__m256i", "")],
    "dst",
    "__m256d",
    """
FOR j := 0 to 3
i := j*64
IF mask[i+63]
    dst[i+63:i] := a+j
ELSE
    dst[i+63:i] := 0
FI
ENDFOR
dst[MAX:256] := 0
""",
)
_mm256_maskload_pd.instype = InsType.LOAD
_mm256_maskload_pd.maskvec = True

load_instructions = [
    globals()[i] for i in dir() if i.startswith("_mm") or i.startswith("_mv")
]

_mv_insert_mem_sd = ParseOperation(
    "_mv_insert_mem_sd",
    [("a", "__m128d", ""), ("b", "double", ""), ("imm8", "const int", "")],
    "dst",
    "__m128d",
    """
dst[127:0] := a[127:0]
CASE (imm8[0]) OF
0: dst[63:0] := b
1: dst[127:64] := b
ESAC
    """,
)
_mv_insert_mem_sd.instype = InsType.INSERT

_mv256_blend_mem_pd = ParseOperation(
    "_mv256_blend_mem_pd",
    [("a", "__m256d", ""), ("b", "double", ""), ("imm8", "const int", "")],
    "dst",
    "__m256d",
    """
FOR j := 0 to 3
    i := j*64
    IF imm8[j]
        dst[i+63:i] := b+j
    ELSE
        dst[i+63:i] := a[i+63:i]
    FI
ENDFOR
dst[MAX:256] := 0
    """,
)
_mv256_blend_mem_pd.instype = InsType.BLEND

_mv_blend_mem_pd = ParseOperation(
    "_mv_blend_mem_pd",
    [("a", "__m128d", ""), ("b", "double", ""), ("imm8", "const int", "")],
    "dst",
    "__m128d",
    """
FOR j := 0 to 1
    i := j*64
    IF imm8[j]
        dst[i+63:i] := b+j
    ELSE
        dst[i+63:i] := a[i+63:i]
    FI
ENDFOR
    """,
)
_mv_blend_mem_pd.instype = InsType.BLEND


insert_blend_instructions = [
    globals()[i] for i in dir() if "insert" in i or "blend" in i
]

# Update intrinsics list for iterate
full_custom_ops_list = [
    globals()[i] for i in dir() if i.startswith("_mm") or i.startswith("_mv")
]

regex = "|".join(
    [
        r"_mm256_insertf128_pd",
        r"_mm_move(hl|lh)_pd",
        r"_mm(256|)_blend_pd",
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

intrinsics = parse_whitelist("sat/data-latest.xml", regex=r"_mm(256|)_set_pd")

# Wrappers
set_2_double_elements= intrinsics["_mm_set_pd"]
set_4_double_elements= intrinsics["_mm256_set_pd"]
intrinsics_double = parse_whitelist("sat/data-latest.xml", regex="_mm256_set_m128d$")
set_hi_lo_double = intrinsics_double["_mm256_set_m128d"]
