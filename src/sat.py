from typing import List
from z3 import *
from sat.x86_sat.parse import *
from tqdm import tqdm

MIN_CANDIDATES = 5

DATA_TYPE = {
    "float": 4,
    "double": 8,
}


def max_width(dtype: str = "float", nelems: int = 4):
    return int(DATA_TYPE[dtype] * 8 * nelems)


regex = "|".join(
    [
        # r"_mm(512|256|)_set_(ps|epi32|epi8)",
        r"_mm(256|)_set_ps",
        r"_mm(256|)_blend_ps",
        r"_mm_insert_ps",
        r"_mm256_insertf128_ps",
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


# Custom defined instructions for MACVETH
_mm_load_ss = parse_operation(
    "_mm_load_ss",
    [("a", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := -1; dst[95:64] := -1; dst[63:32] := -1; dst[31:0] := a;",
)
_mm_loadu_ps = parse_operation(
    "_mm_loadu_ps",
    [("a", "float", "")],
    "dst",
    "__m128",
    "dst[127:96] := a+3; dst[95:64] := a+2; dst[63:32] := a+1; dst[31:0] := a;",
)
# _mm_load_ps = _mm_loadu_ps
_mm256_loadu_ps = parse_operation(
    "_mm256_loadu_ps",
    [("a", "float", "")],
    "dst",
    "__m256",
    "dst[255:224] := a+7; dst[223:192] := a+6; dst[191:160] := a+5; dst[159:128] := a+4; dst[127:96] := a+3; dst[95:64] := a+2; dst[63:32] := a+1; dst[31:0] := a;",
)
# _mm256_load_ps = _mm256_loadu_ps

# load_instructions = [i for i in dir() if i.startswith("_mm") or i.startswith("_mv")]
load_instructions = [_mm_load_ss, _mm_loadu_ps, _mm256_loadu_ps]

_mv_insert_mem_ps = parse_operation(
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

_mv256_blend_mem_ps = parse_operation(
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

_mv_blend_mem_ps = parse_operation(
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


# Update intrinsics list for iterate
custom_ops_list = [i for i in dir() if i.startswith("_mm") or i.startswith("_mv")]
dict_custom_ops = dict(zip(custom_ops_list, custom_ops_list))
intrinsics = parse_whitelist("sat/data-latest.xml", regex=regex)
intrinsics.update(dict_custom_ops)


def print_model(m):
    for mi in m:
        print(mi)


def get_all_combinations(n_ins: int, packing, objective, dtype="float"):
    solutions = []

    def new_solutions(case: int, instructions: List, packing):
        result, model = check(objective == instructions[-1])
        if result == sat:
            print("ALRIGHT: ", objective == instructions[-1], model)
            for i in instructions:
                print("\t", i)
            return instructions
        else:
            if case == n_ins:
                return []
            # keep exploring
            new_insert = _mv_insert_mem_ps(
                instructions[-1], packing[0], Var("i" * case, "int")
            )
            instructions.append(new_insert)
            return new_solutions(case + 1, instructions, packing[1:])

    for load in load_instructions:
        new_inst = load(packing[0])
        if new_inst.width > max_width(dtype, len(packing)):
            continue
        new_sol = new_solutions(1, [new_inst], packing[1:])
        if len(new_sol) == 0:
            print(f"\tNo solutions using {n_ins} instructions with seed {new_inst}")
        else:
            solutions.append(new_sol)
    return solutions


packing = [10, 9, 1, 0]


def main(input: List):
    vpacking = intrinsics["_mm_set_ps"](*input)
    packing.reverse()
    MAX_INS = 2 * len(packing)
    solutions_found = 0
    print(f"Searching packing combinations for: {packing}")
    for n_ins in range(1, MAX_INS):
        print(f"- using {n_ins:3} instruction(s)")
        R = get_all_combinations(n_ins, packing, vpacking)
        solutions_found += len(R)
        if len(R) == 0:
            print(f"\tNo solutions using {n_ins} instruction(s)")
        if solutions_found >= MIN_CANDIDATES:
            print(f"** SEARCH FINISHED WITH {solutions_found} CANDIDATES")
            break

    for s in solutions_found:
        # render(s)
        pass
