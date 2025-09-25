from __future__ import annotations
import ast, astor, autopep8, tokenize, io, sys
import graphviz as gv
from typing import Dict, List, Tuple, Set, Optional, Type
import re
import sys
import src.codetransform.trace_execution as trace_execution
# import trace_execution
import os
import io
import linecache
from PIL import Image
import random
import difflib

# TODO later: graph
'''
1. add a color dictionary for condition calls
2. node shape (may be added into class Block)
'''

class SingletonMeta(type):
    _instance: Optional[BlockId] = None

    def __call__(self) -> BlockId:
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class BlockId(metaclass=SingletonMeta):
    counter: int = 0

    def gen(self) -> int:
        self.counter += 1
        return self.counter

# def a function to check the parenthesis balance
def check_parenthesis(code: str) -> bool:
    stack = []
    for char in code:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack:
                return False
            stack.pop()
    return not stack

class BasicBlock:

    def __init__(self, bid: int):
        self.bid: int = bid
        self.stmts: List[Type[ast.AST]] = []
        self.calls: List[str] = []
        self.prev: List[int] = []
        self.next: List[int] = []
        self.condition = False
        self.for_loop = 0
        self.for_name = Type[ast.AST]

    def is_empty(self) -> bool:
        return len(self.stmts) == 0

    def has_next(self) -> bool:
        return len(self.next) != 0

    def has_previous(self) -> bool:
        return len(self.prev) != 0

    def remove_from_prev(self, prev_bid: int) -> None:
        if prev_bid in self.prev:
            self.prev.remove(prev_bid)

    def remove_from_next(self, next_bid: int) -> None:
        if next_bid in self.next:
            self.next.remove(next_bid)

    def stmts_to_code(self) -> str:
        code_line = ''
        code = ''
        for stmt in self.stmts:
            line = astor.to_source(stmt)
            code_line = line.split('\n')
            content = code_line[0]
            if not check_parenthesis(content):
                for i in range(1,len(code_line)):
                    content += "\n"+code_line[i]
                    if check_parenthesis(content):
                        break
                code += content
            else:
                code += content
        return code

    def calls_to_code(self) -> str:
        return '\n'.join(self.calls)


class CFG:

    def __init__(self, name: str):
        self.name: str = name

        # I am sure that in original code variable asynchr is not used
        # And I think list finalblocks is also not used.

        self.start: Optional[BasicBlock] =  None
        self.func_calls: Dict[str, CFG] = {}
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: Dict[Tuple[int, int], Type[ast.AST]] = {}
        self.graph: Optional[gv.dot.Digraph] = None
        self.execution_path: List[int] = []
        self.error: bool = False
        self.path: List[int] = []
        self.func_name: List[str] = []
        self.matching: Dict[int, int] = {}
        self.revert: Dict[int, int] = {}

    def clean(self):
        des_edges = {}
        for edge in self.edges:
            if edge[0] not in des_edges:
                des_edges[edge[0]] = [edge[1]]
            else:
                des_edges[edge[0]].append(edge[1])
        blank_node = []
        for node in des_edges:
            check = False
            if self.blocks[node].stmts_to_code():
                current = node
                for next in des_edges[node]:
                    condition = self.edges[(node, next)]
                    next_node = next
                    while self.blocks[next_node].stmts_to_code() == '':
                        if next_node not in blank_node:
                            blank_node.append(next_node)
                        current = next_node
                        if current not in des_edges:
                            break
                        next_node = des_edges[current][0]
                    if (node, next_node) not in self.edges:
                        self.edges[(node, next_node)] = condition
                        if next_node != node and next_node not in self.blocks[node].next:
                            self.blocks[node].next.append(next_node)
            else:
                if node not in blank_node:
                    blank_node.append(node)
        for edge in self.edges.copy():
            if edge[0] in blank_node or edge[1] in blank_node:
                self.edges.pop(edge)
        for i in self.blocks:
            for node in blank_node:
                if node in self.blocks[i].next:
                    self.blocks[i].next.remove(node)

    def get_all_nodes(self):
        nodes = self.blocks
        for k, v in self.func_calls.items():
            nodes.update(v.get_all_nodes())
        return nodes

    def get_all_edges(self):
        edges = self.edges
        for k, v in self.func_calls.items():
            edges.update(v.get_all_edges())
        return edges
    
    def get_all_function_name(self):
        functions = {}
        for k, v in self.func_calls.items():
            functions[k] = v.start.bid
            functions.update(v.get_all_function_name())
        return functions
    
    def track_execution(self, filename, func_name = None):
        nodes = []
        blocks = []
        self.revert = {}
        all_nodes = self.get_all_nodes()
        for i in all_nodes:
            if all_nodes[i].stmts_to_code():
                st = all_nodes[i].stmts_to_code()
                st_no_space = re.sub(r"\s+", "", st)
                st_no_space = st_no_space.replace('"', "'")
                blocks.append(st_no_space)
                # if start with if or while, delete these keywords
                if st.startswith('if'):
                    st = st[3:]
                elif st.startswith('while'):
                    st = st[6:]
                if all_nodes[i].condition:
                    st = 'T '+ st
                nodes.append(st)
                self.matching[i] = len(nodes)
                self.revert[len(nodes)] = i
        all_edges = self.get_all_edges()
        edges = {}
        for edge in all_edges:
            if self.matching[edge[0]] not in edges:
                edges[self.matching[edge[0]]] = [self.matching[edge[1]]]
            else:
                edges[self.matching[edge[0]]].append(self.matching[edge[1]])
        t = trace_execution.Trace(ignoredirs=[sys.base_prefix, sys.base_exec_prefix,],

        
                            trace=0, count=1)
        arguments = []
        sys.argv = [filename, arguments]
        sys.path[0] = os.path.dirname(filename)

        with io.open_code(filename) as fp:
            code = compile(fp.read(), filename, 'exec')
        # try to emulate __main__ namespace as much as possible
        globs = {
            '__file__': filename,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }
        error = False
        try:
            t.runctx(code, globs, globs)
        except:
            error = True

        source = linecache.getlines(filename)
        code_line = [element.lstrip().replace('\n', '') for element in source]
        execution_path = []
        for lineno in t.exe_path:
            no_spaces = re.sub(r"\s+", "", code_line[lineno-1])
            if no_spaces.startswith("def") or no_spaces.startswith("assert"):
                continue
            if no_spaces.startswith(f'{func_name}('):
                continue
            if no_spaces.startswith("elif"):
                no_spaces = no_spaces[2:]
            execution_path.append(no_spaces)
        check_True_condition = []
        for i in range(len(execution_path)-1):
            if execution_path[i].startswith('if') or execution_path[i].startswith('while') or execution_path[i].startswith('for'):
                if t.exe_path[i+1] == t.exe_path[i]+1:
                    check_True_condition.append(i)

        current_node = 1

        true_execution = [self.revert[current_node]]
        path = [self.revert[current_node]]
        exit_flag = False
        for s in range(len(execution_path)):
            node = execution_path[s]
            if s == 0:
                continue
            c = 0
            if node == "break" or node == "continue":
                continue
            if len(edges[current_node]) == 2:
                if blocks[edges[current_node][0]-1] == blocks[edges[current_node][1]-1] == node:
                    if (s-1) in check_True_condition:
                        if all_nodes[self.revert[edges[current_node][0]]].condition:
                            current_node = edges[current_node][0]
                        else:
                            current_node = edges[current_node][1]
                    else:
                        if all_nodes[self.revert[edges[current_node][0]]].condition:
                            current_node = edges[current_node][1]
                        else:
                            current_node = edges[current_node][0]
                    if self.revert[current_node] not in path:
                        path.append(self.revert[current_node])
                    continue
            for next_node in edges[current_node]:
                c += 1
                node = node.replace('"', "'")
                # print(node)
                # print(blocks[next_node-1])
                # print("___________")
                if blocks[next_node-1] == node:
                    current_node = next_node
                    break
                if c == len(edges[current_node]):
                    exit_flag = True
                    raise Exception(f"Error: Cannot find the execution path in CFG in file {filename}")
            true_execution.append(self.revert[current_node])
            if exit_flag:
                break
            if self.revert[current_node] not in path:    
                path.append(self.revert[current_node])
        node_max = 0
        for i in all_nodes:
            if i > node_max:
                node_max = i 
        path.append(node_max)

        self.path = path
        return true_execution, error
    
    def track_execution_new(self, source, func_name = None):
        nodes = []
        blocks = []
        self.matching = {}
        all_nodes = self.get_all_nodes()
        for i in all_nodes:
            if all_nodes[i].stmts_to_code():
                st = all_nodes[i].stmts_to_code()
                st_no_space = re.sub(r"\s+", "", st)
                st_no_space = st_no_space.replace('"', "'")
                # remove ( and ) in st_no_space
                st_no_space = st_no_space.replace('(', '').replace(')', '')
                blocks.append(st_no_space)
                # if start with if or while, delete these keywords
                if st.startswith('if'):
                    st = st[3:]
                elif st.startswith('while'):
                    st = st[6:]
                if all_nodes[i].condition:
                    st = 'T '+ st
                nodes.append(st)
                self.matching[i] = len(nodes)
                self.revert[len(nodes)] = i

        all_edges = self.get_all_edges()
        edges = {}
        for edge in all_edges:
            if self.matching[edge[0]] not in edges:
                edges[self.matching[edge[0]]] = [self.matching[edge[1]]]
            else:
                edges[self.matching[edge[0]]].append(self.matching[edge[1]])

        all_functions = self.get_all_function_name()
        for name in all_functions:
            # if element in blocks contain name, then add all_functions[name] to edges
            for i in range(len(blocks)):
                if name in blocks[i]:
                    if i+1 not in edges:
                        edges[i+1] = [self.matching[all_functions[name]]]
                    else:
                        edges[i+1].append(self.matching[all_functions[name]])
        # Create the trace object (adjust the arguments as needed)
        t = trace_execution.Trace(
            ignoredirs=[sys.base_prefix, sys.base_exec_prefix],
            trace=0,
            count=1
        )

        # Prepare the global namespace to emulate __main__
        globs = {
            '__file__': "cfg.py",  # Use <string> to indicate code is from a string
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }
        # Compile the code from the `source` string
        code = compile(source, "cfg.py", 'exec')
        # with io.open_code('test_cfg.py') as fp:
        #     code = compile(fp.read(), filename, 'exec')

        error = False
        try:
            t.runctx(code, globs, globs)
        except:
            error = True
        code_line = [element.lstrip() for element in source.split("\n")]
        execution_path = []
        for lineno in t.exe_path:
            no_spaces = re.sub(r"\s+", "", code_line[lineno-1])
            if no_spaces.startswith("assert"):
                continue
            if no_spaces.startswith(f'{func_name}('):
                continue
            if no_spaces.startswith("elif"):
                no_spaces = no_spaces[2:]
            if no_spaces.startswith("class"):
                continue
            # remove ( and ) in no_spaces
            no_spaces = no_spaces.replace('(', '').replace(')', '')
            execution_path.append(no_spaces)
        for node in execution_path:
            if node == "break" or node == "continue":
                continue
            if node not in blocks:
                find = False
                for i in range(len(blocks)):
                    if node in blocks[i]:
                        self.path.append(i+1)
                        find = True
                        break
                if not find:
                    closest_match = difflib.get_close_matches(node, blocks, n=1, cutoff=0.1)
                    if closest_match:
                        index = blocks.index(closest_match[0])
                        self.path.append(index+1)
            else:
                # get number of element in blocks that match with node 
                all_index = [i+1 for i, x in enumerate(blocks) if x == node]
                if len(all_index) == 1:
                    self.path.append(all_index[0])
                else:
                    if self.path[-1] not in edges:
                        self.path.append(all_index[0])
                    else:
                        find = False
                        for index in all_index:
                            if index in edges[self.path[-1]]:
                                self.path.append(index)
                                find = True
                                break
                        if not find:
                            # get the index closest to self.path[-1]
                            closest_index = min(all_index, key=lambda x: abs(x - self.path[-1]))
                            self.path.append(closest_index)
                            
        return error

    def _traverse(self, block: BasicBlock, get_coverage: bool = False, get_execution: bool = False, visited: Set[int] = set(), calls: bool = True, error: bool = False, path: list[int] = list(), matching: Dict[int, int] = {}) -> None:
        # Add block.bid to the node label
        if block.bid not in visited:
            visited.add(block.bid)
            if get_execution:
                st = f'{matching[block.bid]} \n'
                st += block.stmts_to_code()
            else:
                st = block.stmts_to_code()
            # Check if the block is in the path and highlight it
            node_attributes = {'shape': 'ellipse'}
            if get_coverage:
                if matching[block.bid] in path:
                    if error:
                        node_attributes['color'] = 'red'
                    else:
                        node_attributes['color'] = 'green'
                    node_attributes['style'] = 'filled'

            self.graph.node(str(block.bid), label=st, _attributes=node_attributes)
            # block.next contain some duplicate values, only keep the unique values
            for next_bid in set(block.next):
                self._traverse(self.blocks[next_bid], get_coverage, get_execution, visited, calls=calls, error=error, path=path, matching=matching)
                self.graph.edge(str(block.bid), str(next_bid), label=self.edges[(block.bid, next_bid)] if type(self.edges[(block.bid, next_bid)]) == str else '')
        return visited

    def _show(self, get_coverage: bool = False, get_execution: bool = False, fmt: str = 'png', calls: bool = True, node: int = 0, error: bool = False, path: list[int] = list(), matching: Dict[int, int] = {}) -> gv.dot.Digraph:
        if self.name.endswith('.py'):
            self.graph = gv.Digraph(name='cluster_'+self.name, format=fmt )
        else:
            self.graph = gv.Digraph(name='cluster_'+self.name, format=fmt, graph_attr={'label': self.name})
        if node != 0:
            visited = set()
            visited = self._traverse(self.start, get_coverage, get_execution, visited, calls=calls, error=error, path=[node], matching=matching)
        else:
            self._traverse(self.start, get_coverage = get_coverage, get_execution = get_execution, calls=calls, path=path, matching=matching)
        for k, v in self.func_calls.items():
            v.clean()
            self.graph.subgraph(v._show(get_coverage, get_execution, fmt, calls, node, error, path, matching))
        return self.graph


    def show(self, source: str, get_coverage: bool = False, get_execution: bool = False ,filepath: str = './cfg', fmt: str = 'png', calls: bool = True) -> None:
        if get_coverage or get_execution:
            self.error = self.track_execution_new(source=source)
        self._show(get_coverage, get_execution, fmt, calls, path=self.path, matching=self.matching)
        self.graph.render(filepath, cleanup=True)

    def show_execution(self, source: str, filepath: str = './execution', fmt: str = 'png', calls: bool = True) -> None:
        self.error = self.track_execution_new(source=source)
        count = 1
        filepath = filepath + "_execution"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for node in self.path:
            self.graph = None
            if count == len(self.path):
                self._show(True, False, fmt, calls, node,  error=self.error, matching=self.matching)
            else:
                self._show(True, False, fmt, calls, node, matching=self.matching)
            save_path = f'{filepath}/execution_{count}'
            self.graph.render(save_path, cleanup=True)
            count += 1

        png_files = os.listdir(filepath)
        png_files = sorted(png_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        images = []
        for file_name in png_files:
            file_path = os.path.join(filepath, file_name)
            images.append(Image.open(file_path))

        # Save as GIF
        gif_path = filepath + ".gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:], 
            duration=100,  # Duration between frames in milliseconds
            loop=0  # Loop forever
        )


class CFGVisitor(ast.NodeVisitor):

    invertComparators: Dict[Type[ast.AST], Type[ast.AST]] = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
                                                               ast.LtE: ast.Gt,
                                                               ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Is: ast.IsNot,
                                                               ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}

    def __init__(self):
        super().__init__()
        self.count_for_loop = 0
        self.loop_stack: List[BasicBlock] = []
        self.continue_stack: List[BasicBlock] = []
        self.ifExp = False

    def build(self, name: str, tree: Type[ast.AST]) -> CFG:
        self.cfg = CFG(name)
        begin_block = self.new_block()
        begin_block.stmts = [ast.Expr(value=ast.Str(s='BEGIN'))]
        self.cfg.start = begin_block
        self.curr_block = begin_block
        # self.add_edge(begin_block.bid, self.curr_block.bid)

        self.visit(tree)
        exit_block = self.new_block()
        exit_block.stmts = [ast.Expr(value=ast.Str(s='EXIT'))]
        self.add_edge(self.curr_block.bid, exit_block.bid)
        # self.remove_empty_blocks(self.cfg.start)
        return self.cfg

      
    def new_block(self) -> BasicBlock:
        bid: int = BlockId().gen()
        self.cfg.blocks[bid] = BasicBlock(bid)
        return self.cfg.blocks[bid]

    def add_stmt(self, block: BasicBlock, stmt: Type[ast.AST]) -> None:
        # if block.stmts contain 1 stmt, then create the new node and add the stmt to the new node and add edge from block to new node
        if len(block.stmts) == 1:
            new_block = self.new_block()
            new_block.stmts.append(stmt)
            self.add_edge(block.bid, new_block.bid)
            self.curr_block = new_block
        else:
            block.stmts.append(stmt)

    def add_edge(self, frm_id: int, to_id: int, condition=None) -> BasicBlock:
        self.cfg.blocks[frm_id].next.append(to_id)
        self.cfg.blocks[to_id].prev.append(frm_id)       
        self.cfg.edges[(frm_id, to_id)] = condition
        return self.cfg.blocks[to_id]

    def add_loop_block(self) -> BasicBlock:
        if self.curr_block.is_empty() and not self.curr_block.has_next():
            return self.curr_block
        else:
            loop_block = self.new_block()
            self.add_edge(self.curr_block.bid, loop_block.bid)
            return loop_block

    def add_subgraph(self, tree: Type[ast.AST]) -> None:
        self.cfg.func_name.append(tree.name)
        self.cfg.func_calls[tree.name] = CFGVisitor().build(tree.name, ast.Module(body=tree.body))
        self.cfg.func_calls[tree.name].clean()

    def add_condition(self, cond1: Optional[Type[ast.AST]], cond2: Optional[Type[ast.AST]]) -> Optional[Type[ast.AST]]:
        if cond1 and cond2:
            return ast.BoolOp(ast.And(), values=[cond1, cond2])
        else:
            return cond1 if cond1 else cond2

    # not tested
    def remove_empty_blocks(self, block: BasicBlock, visited: Set[int] = set()) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            if block.is_empty():
                for prev_bid in block.prev:
                    prev_block = self.cfg.blocks[prev_bid]
                    for next_bid in block.next:
                        next_block = self.cfg.blocks[next_bid]
                        self.add_edge(prev_bid, next_bid)
                        if (block.bid, next_bid) in self.cfg.edges:
                            self.cfg.edges.pop((block.bid, next_bid), None)
                        next_block.remove_from_prev(block.bid)
                    if (prev_bid, block.bid) in self.cfg.edges:
                        self.cfg.edges.pop((prev_bid, block.bid))
                    prev_block.remove_from_next(block.bid)
                block.prev.clear()
                for next_bid in block.next:
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)
                block.next.clear()

            else:
                for next_bid in block.next:
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)

    def invert(self, node: Type[ast.AST]) -> Type[ast.AST]:
        if type(node) == ast.Compare:
            if len(node.ops) == 1:
                return ast.Compare(left=node.left, ops=[self.invertComparators[type(node.ops[0])]()], comparators=node.comparators)
            else:
                tmpNode = ast.BoolOp(op=ast.And(), values = [ast.Compare(left=node.left, ops=[node.ops[0]], comparators=[node.comparators[0]])])
                for i in range(0, len(node.ops) - 1):
                    tmpNode.values.append(ast.Compare(left=node.comparators[i], ops=[node.ops[i+1]], comparators=[node.comparators[i+1]]))
                return self.invert(tmpNode)
        elif isinstance(node, ast.BinOp) and type(node.op) in self.invertComparators:
            return ast.BinOp(node.left, self.invertComparators[type(node.op)](), node.right)
        elif type(node) == ast.NameConstant and type(node.value) == bool:
            return ast.NameConstant(value=not node.value)
        elif type(node) == ast.BoolOp:
            return ast.BoolOp(values = [self.invert(x) for x in node.values], op = {ast.And: ast.Or(), ast.Or: ast.And()}.get(type(node.op)))
        elif type(node) == ast.UnaryOp:
            return self.UnaryopInvert(node)
        else:
            return ast.UnaryOp(op=ast.Not(), operand=node)

    def UnaryopInvert(self, node: Type[ast.AST]) -> Type[ast.AST]:
        if type(node.op) == ast.UAdd:
            return ast.UnaryOp(op=ast.USub(),operand = node.operand)
        elif type(node.op) == ast.USub:
            return ast.UnaryOp(op=ast.UAdd(),operand = node.operand)
        elif type(node.op) == ast.Invert:
            return ast.UnaryOp(op=ast.Not(), operand=node)
        else:
            return node.operand

    # def boolinvert(self, node:Type[ast.AST]) -> Type[ast.AST]:
    #     value = []
    #     for item in node.values:
    #         value.append(self.invert(item))
    #     if type(node.op) == ast.Or:
    #         return ast.BoolOp(values = value, op = ast.And())
    #     elif type(node.op) == ast.And:
    #         return ast.BoolOp(values = value, op = ast.Or())

    def combine_conditions(self, node_list: List[Type[ast.AST]]) -> Type[ast.AST]:
        return node_list[0] if len(node_list) == 1 else ast.BoolOp(op=ast.And(), values = node_list)

    def generic_visit(self, node):
        if type(node) in [ast.Import, ast.ImportFrom]:
            self.add_stmt(self.curr_block, node)
            return
        if type(node) in [ast.FunctionDef, ast.AsyncFunctionDef]:
            self.add_stmt(self.curr_block, node)
            self.add_subgraph(node)
            return
        if type(node) in [ast.AnnAssign, ast.AugAssign]:
            self.add_stmt(self.curr_block, node)
        super().generic_visit(node)

    def get_function_name(self, node: Type[ast.AST]) -> str:
        if type(node) == ast.Name:
            return node.id
        elif type(node) == ast.Attribute:
            return self.get_function_name(node.value) + '.' + node.attr
        elif type(node) == ast.Str:
            return node.s
        elif type(node) == ast.Subscript:
            return node.value.id
        elif type(node) == ast.Lambda:
            return 'lambda function'

    def populate_body(self, body_list: List[Type[ast.AST]], to_bid: int) -> None:
        for child in body_list:
            self.visit(child)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, to_bid)

    # assert type check
    def visit_Assert(self, node):
        # self.add_stmt(self.curr_block, node)
        # # If the assertion fails, the current flow ends, so the fail block is a
        # # final block of the CFG.
        # # self.cfg.finalblocks.append(self.add_edge(self.curr_block.bid, self.new_block().bid, self.invert(node.test)))
        # # If the assertion is True, continue the flow of the program.
        # # success block
        # self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        # self.generic_visit(node)
        pass

    # TODO: change all those registers to stacks!
    def visit_Assign(self, node): 
        #edit
        # if type(node.value) in [ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda] and len(node.targets) == 1 and type(node.targets[0]) == ast.Name: # is this entire statement necessary?
        #     if type(node.value) == ast.ListComp:
        #         self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
        #         self.listCompReg = (node.targets[0].id, node.value)
        #     elif type(node.value) == ast.SetComp:
        #         self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Call(func=ast.Name(id='set', ctx=ast.Load()), args=[], keywords=[])))
        #         self.setCompReg = (node.targets[0].id, node.value)
        #     elif type(node.value) == ast.DictComp:
        #         self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Dict(keys=[], values=[])))
        #         self.dictCompReg = (node.targets[0].id, node.value)
        #     elif type(node.value) == ast.GeneratorExp:
        #         self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Call(func=ast.Name(id='__' + node.targets[0].id + 'Generator__', ctx=ast.Load()), args=[], keywords=[])))
        #         self.genExpReg = (node.targets[0].id, node.value)
        #     else:
        #         self.lambdaReg = (node.targets[0].id, node.value)
        # else:
        self.add_stmt(self.curr_block, node)
        self.generic_visit(node)

    def visit_Await(self, node):
        afterawait_block = self.new_block()
        self.add_edge(self.curr_block.bid, afterawait_block.bid)
        self.generic_visit(node)
        self.curr_block = afterawait_block

    def visit_Break(self, node):
        assert len(self.loop_stack), "Found break not inside loop"
        self.add_edge(self.curr_block.bid, self.loop_stack[-1].bid, ast.Break())

    def visit_Call(self, node):
        if type(node.func) == ast.Lambda:
            self.lambdaReg = ('Anonymous Function', node.func)
            self.generic_visit(node)

    def visit_Continue(self, node):
        assert len(self.continue_stack), "Found continue not inside loop"
        self.add_edge(self.curr_block.bid, self.continue_stack[-1].bid, ast.Continue())

    def visit_DictComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            if self.dictCompReg[0]: # bug if there is else statement in comprehension
                return [ast.Assign(targets=[ast.Subscript(value=ast.Name(id=self.dictCompReg[0], ctx=ast.Load()), slice=ast.Index(value=self.dictCompReg[1].key), ctx=ast.Store())], value=self.dictCompReg[1].value)]
            # else: # not supported yet
            #     return [ast.Expr(value=self.dictCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_DictComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_DictComp_Rec(generators[:-1]), orelse=[])]

    def visit_DictComp(self, node):
        try: # try may change to checking if self.dictCompReg exists
            self.generic_visit(ast.Module(self.visit_DictComp_Rec(self.dictCompReg[1].generators)))
        except:
            pass
        finally:
            self.dictCompReg = None

    # ignore the case when using set or dict comprehension or generator expression but the result is not assigned to a variable
    def visit_Expr(self, node):
        #edit
        # if type(node.value) == ast.ListComp and type(node.value.elt) == ast.Call:
        #     self.listCompReg = (None, node.value)
        # elif type(node.value) == ast.Lambda:
        #     self.lambdaReg = ('Anonymous Function', node.value)
        # # elif type(node.value) == ast.Call and type(node.value.func) == ast.Lambda:
        # #     self.lambdaReg = ('Anonymous Function', node.value.func)
        # else:
        self.add_stmt(self.curr_block, node)
        self.generic_visit(node)

    def visit_For(self, node):
        loop_guard = self.add_loop_block()
        self.continue_stack.append(loop_guard)
        self.curr_block = loop_guard
        self.add_stmt(self.curr_block, node)
        # New block for the body of the for-loop.
        for_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "T")
        if not node.orelse:
            # Block of code after the for loop.
            afterfor_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "F")
            self.loop_stack.append(afterfor_block)
            self.curr_block = for_block

            self.curr_block = self.populate_For_body(node.body)
            self.add_edge(self.curr_block.bid, loop_guard.bid)

        else:
            # Block of code after the for loop.
            afterfor_block = self.new_block()
            orelse_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "F")
            self.loop_stack.append(afterfor_block)
            self.curr_block = for_block

            self.curr_block = self.populate_For_body(node.body)
            self.add_edge(self.curr_block.bid, loop_guard.bid)


            self.curr_block = orelse_block
            for child in node.orelse:
                self.visit(child)
            self.add_edge(orelse_block.bid, afterfor_block.bid, "For")
        
        # if not node.orelse:
        #     # Block of code after the for loop.
        #     afterfor_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        #     self.loop_stack.append(afterfor_block)
        #     self.curr_block = for_block

        #     self.curr_block = self.populate_For_body(node.body)
        #     self.add_edge(self.curr_block.bid, loop_guard.bid)

        # else:
        #     # Block of code after the for loop.
        #     afterfor_block = self.new_block()
        #     orelse_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "F")
        #     self.loop_stack.append(afterfor_block)
        #     self.curr_block = for_block

        #     self.curr_block = self.populate_For_body(node.body)
        #     self.add_edge(self.curr_block.bid, loop_guard.bid)

        #     self.curr_block = orelse_block
        #     for child in node.orelse:
        #         self.visit(child)
        #     self.add_edge(orelse_block.bid, afterfor_block.bid, "For")

        # Continue building the CFG in the after-for block.
        self.curr_block = afterfor_block
        self.continue_stack.pop()
        self.loop_stack.pop()

    def populate_For_body(self, body_list: List[Type[ast.AST]]) -> None:
        for child in body_list:
            self.visit(child)
        new_node = self.new_block()
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, new_node.bid)
        return new_node
    
    def visit_GeneratorExp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.genExpReg[1].elt) # the location of the node may be wrong
            if self.genExpReg[0]: # bug if there is else statement in comprehension
                return [ast.Expr(value=ast.Yield(value=self.genExpReg[1].elt))]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_GeneratorExp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_GeneratorExp_Rec(generators[:-1]), orelse=[])]

    def visit_GeneratorExp(self, node):
        try: # try may change to checking if self.genExpReg exists
            self.generic_visit(ast.FunctionDef(name='__' + self.genExpReg[0] + 'Generator__', 
                args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body = self.visit_GeneratorExp_Rec(self.genExpReg[1].generators), 
                decorator_list=[], returns=None))
        except:
            pass
        finally:
            self.genExpReg = None

    def visit_If(self, node):
        # Add the If statement at the end of the current block.
        self.add_stmt(self.curr_block, node)

        # Create a block for the code after the if-else.
        afterif_block = self.new_block()
        # Create a new block for the body of the if.
        body_block = self.new_block()
        body_block.condition = True
        if_block = self.add_edge(self.curr_block.bid, body_block.bid, "T")

        # New block for the body of the else if there is an else clause.
        if node.orelse:
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "F")

            # Visit the children in the body of the else to populate the block.
            self.populate_body(node.orelse, afterif_block.bid)
        else:
            self.add_edge(self.curr_block.bid, afterif_block.bid, "F")

        # Visit children to populate the if block.
        self.curr_block = if_block

        self.populate_body(node.body, afterif_block.bid)

        # Continue building the CFG in the after-if block.
        self.curr_block = afterif_block

    def visit_IfExp_Rec(self, node: Type[ast.AST]) -> List[Type[ast.AST]]:
        return [ast.If(test=node.test, body=[ast.Return(value=node.body)], orelse=self.visit_IfExp_Rec(node.orelse) if type(node.orelse) == ast.IfExp else [ast.Return(value=node.orelse)])]

    def visit_IfExp(self, node):
        if self.ifExp:
            self.generic_visit(ast.Module(self.visit_IfExp_Rec(node)))

    def visit_Lambda(self, node): # deprecated since there is autopep8
        self.add_subgraph(ast.FunctionDef(name=self.lambdaReg[0], args=node.args, body = [ast.Return(value=node.body)], decorator_list=[], returns=None))
        self.lambdaReg = None

    def visit_ListComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.listCompReg[1].elt) # the location of the node may be wrong
            if self.listCompReg[0]: # bug if there is else statement in comprehension
                return [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.listCompReg[0], ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[self.listCompReg[1].elt], keywords=[]))]
            else:
                return [ast.Expr(value=self.listCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_ListComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_ListComp_Rec(generators[:-1]), orelse=[])]

    def visit_ListComp(self, node):
        try: # try may change to checking if self.listCompReg exists
            self.generic_visit(ast.Module(self.visit_ListComp_Rec(self.listCompReg[1].generators)))
        except:
            pass
        finally:
            self.listCompReg = None

    def visit_Pass(self, node):
        self.add_stmt(self.curr_block, node)

    def visit_Raise(self, node):
        self.add_stmt(self.curr_block, node)
        self.curr_block = self.new_block()

    # ToDO: final blocks to be add
    def visit_Return(self, node):
        #edit
        # if type(node.value) == ast.IfExp:
        #     self.ifExp = True
        #     self.generic_visit(node)
        #     self.ifExp = False
        # else:
        self.add_stmt(self.curr_block, node)
        # self.cfg.finalblocks.append(self.curr_block)
        # Continue in a new block but without any jump to it -> all code after
        # the return statement will not be included in the CFG.
        self.curr_block = self.new_block()

    def visit_SetComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.setCompReg[1].elt) # the location of the node may be wrong
            if self.setCompReg[0]:
                return [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.setCompReg[0], ctx=ast.Load()), attr='add', ctx=ast.Load()), args=[self.setCompReg[1].elt], keywords=[]))]
            else: # not supported yet
                return [ast.Expr(value=self.setCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_SetComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_SetComp_Rec(generators[:-1]), orelse=[])]

    def visit_SetComp(self, node):
        try: # try may change to checking if self.setCompReg exists
            self.generic_visit(ast.Module(self.visit_SetComp_Rec(self.setCompReg[1].generators)))
        except:
            pass
        finally:
            self.setCompReg = None

    def visit_Try(self, node):
        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        self.add_stmt(loop_guard, ast.Try(body=[], handlers=[], orelse=[], finalbody=[]))

        after_try_block = self.new_block()
        self.add_stmt(after_try_block, ast.Name(id='handle errors', ctx=ast.Load()))
        self.populate_body(node.body, after_try_block.bid)

        self.curr_block = after_try_block

        if node.handlers:
            for handler in node.handlers:
                before_handler_block = self.new_block()
                self.curr_block = before_handler_block
                self.add_edge(after_try_block.bid, before_handler_block.bid)

                after_handler_block = self.new_block()
                self.add_stmt(after_handler_block, ast.Name(id='end except', ctx=ast.Load()))
                self.populate_body(handler.body, after_handler_block.bid)
                self.add_edge(after_handler_block.bid, after_try_block.bid)

        if node.orelse:
            before_else_block = self.new_block()
            self.curr_block = before_else_block
            self.add_edge(after_try_block.bid, before_else_block.bid)

            after_else_block = self.new_block()
            self.add_stmt(after_else_block, ast.Name(id='end no error', ctx=ast.Load()))
            self.populate_body(node.orelse, after_else_block.bid)
            self.add_edge(after_else_block.bid, after_try_block.bid)

        finally_block = self.new_block()
        self.curr_block = finally_block

        if node.finalbody:
            self.add_edge(after_try_block.bidxf, finally_block.bid)
            after_finally_block = self.new_block()
            self.populate_body(node.finalbody, after_finally_block.bid)
            self.curr_block = after_finally_block
        else:
            self.add_edge(after_try_block.bid, finally_block.bid)

    def visit_While(self, node):
        loop_guard = self.add_loop_block()
        self.continue_stack.append(loop_guard)
        self.curr_block = loop_guard
        self.add_stmt(loop_guard, node)
        # New block for the case where the test in the while is False.
        afterwhile_block = self.new_block()
        self.loop_stack.append(afterwhile_block)
        inverted_test = self.invert(node.test)

        if not node.orelse:
            # Skip shortcut loop edge if while True:
            if not (isinstance(inverted_test, ast.NameConstant) and inverted_test.value == False):
                self.add_edge(self.curr_block.bid, afterwhile_block.bid, "F")

            # New block for the case where the test in the while is True.
            # Populate the while block.
            body_block = self.new_block()
            body_block.condition = True
            self.curr_block = self.add_edge(self.curr_block.bid, body_block.bid, "T")

            self.populate_body(node.body, loop_guard.bid)
        else:
            orelse_block = self.new_block()
            if not (isinstance(inverted_test, ast.NameConstant) and inverted_test.value == False):
                self.add_edge(self.curr_block.bid, orelse_block.bid, inverted_test)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid, "T")

            self.populate_body(node.body, loop_guard.bid)
            self.curr_block = orelse_block
            for child in node.orelse:
                self.visit(child)
            self.add_edge(orelse_block.bid, afterwhile_block.bid)
        
        # Continue building the CFG in the after-while block.
        self.curr_block = afterwhile_block
        self.loop_stack.pop()
        self.continue_stack.pop()

    def visit_Yield(self, node):
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)


class PyParser:

    def __init__(self, script):
        self.script = script

    def formatCode(self):
        self.script = autopep8.fix_code(self.script)

    # https://github.com/liftoff/pyminifier/blob/master/pyminifier/minification.py
    def removeCommentsAndDocstrings(self):
        io_obj = io.StringIO(self.script)  # ByteIO for Python2?
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        # Note regarding NEWLINE vs NL: The tokenize module
                        # differentiates between newlines that start a new statement
                        # and newlines inside of operators such as parens, brackes,
                        # and curly braces.  Newlines inside of operators are
                        # NEWLINE and newlines that start new code are NL.
                        # Catch whole-module docstrings:
                        if start_col > 0:
                            # Unlabelled indentation means we're inside an operator
                            out += token_string
                        # Note regarding the INDENT token: The tokenize module does
                        # not label indentation inside of an operator (parens,
                        # brackets, and curly braces) as actual indentation.
                        # For example:
                        # def foo():
                        #     "The spaces before this docstring are tokenize.INDENT"
                        #     test = [
                        #         "The spaces before this string do not get a token"
                        #     ]
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        self.script = out

def generate_random_string(input_string, length=64):
    # Ensure input_string is not empty to avoid ValueError
    if not input_string:
        raise ValueError("Input string must not be empty.")
    
    # Generate a random string of the specified length
    random_string = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
    return random_string

def code2cfgimage(source: str, get_execution=False, get_coverage = False, filename = "cfg", random_file = False):
    if random_file:
        file_path = generate_random_string(source, 32)
        file_path = f"/tmp/{file_path}"
    else:
        file_path = "./output_image/" + filename
    
    parser = PyParser(source)
    parser.removeCommentsAndDocstrings()
    parser.formatCode()
    cfg = CFGVisitor().build("code.py", ast.parse(parser.script))  
    cfg.clean()
    cfg.show(source= source, get_coverage = get_coverage, get_execution = get_execution, filepath = file_path)
    return file_path + ".png", cfg.path

def code2cfgvid(source: str, filename = "cfg", random_file = False):
    if random_file:
        file_path = generate_random_string(source, 32)
        file_path = f"/tmp/{file_path}"
    else:
        file_path = "./output_image/" + filename
    parser = PyParser(source)
    parser.removeCommentsAndDocstrings()
    parser.formatCode()
    cfg = CFGVisitor().build("code.py", ast.parse(parser.script))  
    cfg.clean()
    cfg.show_execution(source = source, filepath = file_path)
    return file_path + "_execution.gif", cfg.path

if __name__ == '__main__':
    filename = sys.argv[1]
    if "get_execution" in sys.argv:
        get_execution = True
    else:
        get_execution = False
    if "get_coverage" in sys.argv:
        get_coverage = True
    else:
        get_coverage = False

    source = open(filename, 'r').read()
    if get_execution:
        code2cfgvid(source)
    else:
        code2cfgimage(source, get_execution = get_execution, get_coverage = get_coverage)
