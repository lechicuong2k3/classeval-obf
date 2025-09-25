from __future__ import annotations
import sys
import src.codetransform.trace_execution as trace_execution
import copy
from types import FunctionType

class ExecutionTracer:
    def __init__(self):
        self.tracer = trace_execution.Trace(count=False, trace=True, countfuncs=False, countcallers=False)
        self.execution_trace = []
        self.idx = 0
        self.error = None
        self.asserterror = False

    def trace_execution(self, frame, event, arg):
        if '__builtins__' not in frame.f_locals and ".0" not in frame.f_locals and len(frame.f_locals) > 0 and 'module' not in frame.f_locals:
            code = frame.f_code
            lineno = frame.f_lineno
            locals_snapshot = copy.deepcopy(frame.f_locals) 
            self.execution_trace.append([lineno, locals_snapshot])
        return self.trace_execution

    def start_tracing(self, code):
        sys.settrace(self.trace_execution)
        try:
            exec(code, {})
        except AssertionError as e:
            self.error = e
            self.asserterror = True
        except Exception as e:
            self.error = e
        sys.settrace(None)

    def get_execution_trace(self):
        return self.execution_trace

def generate_commented_code(source, in_line_cmt):
    code_lines = source.split('\n')
    code_lines = [element.rstrip() for element in code_lines]
    commented_code = []

    for lineno, line in enumerate(code_lines, 1):
        if lineno in in_line_cmt:
            comments = in_line_cmt[lineno]
            
            if comments:
                first_comment = comments[0]
                last_comment = comments[-1]

                # Handle NO_CHANGE scenario
                if isinstance(first_comment[1], str):
                    first_comment_str = f"({first_comment[0]}) {first_comment[1]}"
                else:
                    first_comment_str = f"({first_comment[0]}) " + "; ".join([f"{k}={v}" for k, v in first_comment[1].items()])
                
                if first_comment == last_comment:
                    inline_comment = f" # {first_comment_str}"
                else:
                    if isinstance(last_comment[1], str):
                        last_comment_str = f"({last_comment[0]}) {last_comment[1]}"
                    else:
                        last_comment_str = f"({last_comment[0]}) " + "; ".join([f"{k}={v}" for k, v in last_comment[1].items()])
                    
                    inline_comment = f" # {first_comment_str}; ...; {last_comment_str}"
                
                commented_code.append(line + inline_comment)
            else:
                commented_code.append(line)
        else:
            commented_code.append(line)

    return "\n".join(commented_code)

def execute_and_trace(source: str):
    code = compile(source, "next.py", 'exec')
    tracer = ExecutionTracer()
    tracer.start_tracing(code)
    execution_trace = tracer.get_execution_trace()
    error = tracer.error
    asserterror = tracer.asserterror

    intermediate_value = []
    code_line = [element.lstrip() for element in source.split("\n")]
    condition_line = []
    def_line = 0
    for i in range(len(code_line)):
        if code_line[i].startswith('if') or code_line[i].startswith('elif') or code_line[i].startswith('else'):
            condition_line.append(i+1)
        if code_line[i].startswith('assert'):
            assert_line = i+1
        if code_line[i].startswith('def'):
            if def_line == 0:
                def_line = i+1
    for i in range(len(execution_trace)-1):
        if execution_trace[i][0] not in condition_line:
            if i > 0:
                if execution_trace[i][0] == execution_trace[i-1][0]:
                    if execution_trace[i][1] == execution_trace[i-1][1]:
                        continue
            intermediate_value.append([execution_trace[i][0], execution_trace[i+1][1]])

    if error != None:
        if asserterror:
            intermediate_value.append([assert_line , f'__exception__ = AssertionError()'])
        else:
            if len(intermediate_value)>0:
                intermediate_value[-1][1] = f'__exception__ = {error}'
            else:
                intermediate_value.append([def_line,f'__exception__ = {error}'])
    
    symbol_table = {}
    values = []
    for i in range(len(intermediate_value)):
        if i == len(intermediate_value)-1 and error ==None:
            temp_dict = {}
            for var in intermediate_value[i][1]:
                if var.startswith("__class__"):
                    continue
                if var.startswith("self"):
                    continue
                if var.startswith("__module__"):
                    continue
                if var.startswith("__qualname__"):
                    continue
                if isinstance(intermediate_value[i][1][var], FunctionType):
                    continue
                temp_dict[var] = intermediate_value[i][1][var]
            values.append([intermediate_value[i][0], temp_dict])
        elif i == len(intermediate_value)-2 and error != None:
            temp_dict = {}
            for var in intermediate_value[i][1]:
                if var.startswith("__class__"):
                    continue
                if var.startswith("self"):
                    continue
                if var.startswith("__module__"):
                    continue
                if var.startswith("__qualname__"):
                    continue
                if isinstance(intermediate_value[i][1][var], FunctionType):
                    continue
                temp_dict[var] = intermediate_value[i][1][var]
            values.append([intermediate_value[i][0], temp_dict])
        elif i == len(intermediate_value)-1 and error != None:
            values.append([intermediate_value[i][0], intermediate_value[i][1]])
        else:
            temp_dict = {}
            for var in intermediate_value[i][1]:
                if var.startswith("__class__"):
                    continue
                if var.startswith("self"):
                    continue
                if var.startswith("__module__"):
                    continue
                if var.startswith("__qualname__"):
                    continue
                if isinstance(intermediate_value[i][1][var], FunctionType):
                    continue
                if var not in symbol_table:
                    symbol_table[var] = intermediate_value[i][1][var]
                    temp_dict[var] = intermediate_value[i][1][var]
                else:
                    if symbol_table[var] != intermediate_value[i][1][var]:
                        symbol_table[var] = intermediate_value[i][1][var]
                        temp_dict[var] = intermediate_value[i][1][var]
            if len(temp_dict) > 0:
                values.append([intermediate_value[i][0], temp_dict])
            else:
                values.append([intermediate_value[i][0], "NO_CHANGE"])

    in_line_cmt = {}
    for i in range(len(values)):
        if values[i][0] not in in_line_cmt:
            in_line_cmt[values[i][0]] = [[i, values[i][1]]]
        else:
            in_line_cmt[values[i][0]].append([i, values[i][1]])
        
    commented_code = generate_commented_code(source, in_line_cmt)

    return commented_code


if __name__ == '__main__':
    source ="""from collections import defaultdict

def find_covered_people(N, M, parents, insurances):
    # Create family tree
    children = defaultdict(list)
    for i in range(2, N + 1):
        children[parents[i - 2]].append(i)
    
    # Initialize coverage set
    covered = set()
    
    def cover_descendants(person, generations):
        if generations < 0 or person in covered:
            return
        covered.add(person)
        for child in children[person]:
            cover_descendants(child, generations - 1)
    
    # Process insurances
    for x, y in insurances:
        cover_descendants(x, y)
    
    return len(covered)

assert find_covered_people(7, 3, [1, 2, 1, 3, 3, 3], [(1, 1), (1, 2), (4, 3)]) == 4"""
    print(execute_and_trace(source))