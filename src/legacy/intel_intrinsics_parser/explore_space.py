#!/env/python3

import intrinsics_parser as ip


class CookingRecipe:
    __uid = 0

    def __init__(self, instructions):
        self.id = self.__uid
        self.instructions = instructions
        self.__uid += 1

    def __str__(self):
        s = f"Cocking recipe = {self.id}\n"
        for ins in self.instructions:
            s += f"{ins}\n"
        return s


class PackingSpace:
    """Wrapper class for packing generation case.

    This class is meant to generate a search space for each
    """

    def load_intrinsics_operations(self):
        self.load_op = []
        self.swizzle_op = []
        intrinsics_list = ip.IntrinsicsParser.get_list_intrinsics(self.isa)

        self.intrinsics_load = []
        self.intrinsics_swizzle = []
        for intrin in intrinsics_list:
            # Avoid macros such as TRANSPOSE and other garbage instructions
            if intrin.intrinsics_name.startswith("_MM_"):
                continue
            if "Load" in intrin.category:
                self.intrinsics_load.append(intrin)
            if "Swizzle" in intrin.category:
                self.intrinsics_swizzle.append(intrin)

    def __init__(self, type, isa):
        self.type = type
        self.isa = isa
        self.load_intrinsics_operations()

    def vpack(self, mem_addr, vect_slots):
        nelems = len(mem_addr)
        # Find load instructions
        for i in range(nelems):
            # print(i)
            pass
        CR = []
        return CR


if __name__ == "__main__":
    mem_addr = [0x0, 0x20, 0x40, 0x80]
    vect_slots = [0, 1, 2, 3]
    P = PackingSpace("float", "avx2")
    CR = P.vpack(mem_addr, vect_slots)
    for C in CR:
        print(C)
