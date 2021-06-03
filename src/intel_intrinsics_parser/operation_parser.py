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

import sys


class OParser:
    class Range:
        def __init__(self, node, name_id):
            if "property" in node.keys():
                raise NotImplementedError()
            if "addr" in name_id:
                self.end = node["start"]["right"]["value"]
            else:
                self.end = node["start"]["value"]
            if node["end"]["type"] == "Identifier" and "addr" in name_id:
                self.start = 0
            else:
                self.start = node["end"]["value"]

        def __str__(self):
            return f"{self.end}:{self.start}"

    def get_for_limits(self, node, ctx):
        init_val = 0
        upperbound = 0
        # Init part
        init_node = node["init"][0]
        if init_node["type"] != "AssignmentExpression":
            raise NotImplementedError()
        lhs = init_node["left"]
        rhs = init_node["right"]
        init_val = rhs["value"]
        ctx["Identifier"][lhs["name"]] = init_val
        # Upper bound
        ub_node = node["varmax"]
        if ub_node["type"] == "Literal" or ub_node["type"] == "NumericLiteral":
            upperbound = ub_node["value"]
        else:
            raise NotImplementedError()
        return init_val, upperbound, lhs["name"]

    def parse_for_stmt(self, node, ctx):
        init_val, upperbound, var = self.get_for_limits(node, ctx)
        for i in range(init_val, upperbound):
            ctx["Identifier"][var] = i
            self.parse(node["body"], ctx)

    def parse_if_stmt(self, node, ctx):
        # "test"
        # "consequent"
        # "alternate"
        pass

    def parse_member_expr(self, node, ctx):
        if not node["computed"]:
            raise NotImplementedError()
        name_id = node["object"]["name"]
        # FIXME: this does not work for _mm256_loadu2_m128, etc.
        if name_id == "MEM":
            name_id = "mem_addr"
        ctx["Identifier"][name_id] = OParser.Range(node["range"], name_id)

    def skip_assignment_expr(self, node):
        try:
            if node["left"]["range"]["start"]["name"] != "MAX":
                return False
            if node["right"]["value"] != 0:
                return False
        except KeyError:
            return False
        return True

    def parse_expr(self, node, ctx):
        ntype = node["type"]
        if ntype == "AssignmentExpression":
            if self.skip_assignment_expr(node):
                return
            rhs = node["right"]
            self.parse_expr(rhs, ctx)
            lhs = node["left"]
            self.parse_expr(lhs, ctx)
        if ntype == "MemberExpression":
            self.parse_member_expr(node, ctx)

    def parse_stmt(self, node, ctx):
        if type(node) == list:
            node = node[0][0]
        ntype = node["type"]
        if ntype == "ForStatement":
            self.parse_for_stmt(node, ctx)
        elif ntype == "IfStatement":
            self.parse_if_stmt(node, ctx)
        elif ntype == "ExpressionStatement":
            self.parse_expr(node["expression"], ctx)
        else:
            self.parse_expr(node, ctx)

    def evaluate(self):
        output_vals = []
        input_vals = []
        root = self.op["operation"]
        if root["type"] != "Program":
            print("Something went wrong")
            sys.exit(1)
        ctx = {"Identifier": {}}
        for leaf in root["body"]:
            if type(leaf) != list:
                self.parse_stmt(leaf, ctx)

        for arg in self.args:
            print(f"MEM VALUES = {ctx['Identifier'][arg.varname]}")
        for ret in self.ret:
            print(f"RET VALUES = {ctx['Identifier'][ret.varname]}")
        return output_vals, input_vals

    def __init__(self, intrinsics):
        self.ret = intrinsics.ret
        self.args = intrinsics.args
        self.op = intrinsics.operation


class StmtParse:
    def parse(self):
        raise NotImplementedError()


# class ForParser(StmtParse):
#    def parse(self)
