
def make_cot_output_prompt(s):
    code, input, _ = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in <ANSWER> and </ANSWER> tags, following the examples.

<PYTHON>
def performOperation(s):
    s = s + s
    return "b" + s + "a"
assert performOperation(s = "hi") == ??
</PYTHON>
<THOUGHT>
Let's execute the code step by step:

1. The function performOperation is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
</THOUGHT>
<ANSWER>
assert performOperation(s = "hi") == "bhihia"
</ANSWER>

<PYTHON>
{code}
assert {input} == ??
</PYTHON>
<THOUGHT>
"""


def make_direct_output_prompt(s):
    code, input, _ = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in <ANSWER> and </ANSWER> tags, following the examples.

<PYTHON>
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
</PYTHON>
<ANSWER>
assert repeatNumber(number = 17) == 17
</ANSWER>

<PYTHON>
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
</PYTHON>
<ANSWER>
assert addCharacterA(string = "x9j") == "x9ja"
</ANSWER>

<PYTHON>
{code}
assert {input} == ??
</PYTHON>
<ANSWER>
"""

def make_cot_input_prompt(s):
    code, output, function_name = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with <ANSWER> and </ANSWER> tags. Express your answer as a passing assertion containing the input and the given output.

<PYTHON>
def f(x):
    return x + 1
assert f(??) == 17
</PYTHON>
<THOUGHT>
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
</THOUGHT>
<ANSWER>
assert f(16) == 17
</ANSWER>

<PYTHON>
{code}
assert {function_name}(??) == {output}
</PYTHON>
<THOUGHT>
"""

def make_cot_output_cruxeval_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in <ANSWER> and </ANSWER> tags, following the examples.

<PYTHON>
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
</PYTHON>
<THOUGHT>
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
</THOUGHT>
<ANSWER>
assert f("hi") == "bhihia"
</ANSWER>

<PYTHON>
{code}
assert f({input}) == ??
</PYTHON>
<THOUGHT>
"""

def make_cot_input_cruxeval_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with <ANSWER> and </ANSWER> tags. Express your answer as a passing assertion containing the input and the given output.

<PYTHON>
def f(x):
    return x + 1
assert f(??) == 17
</PYTHON>
<THOUGHT>
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
</THOUGHT>
<ANSWER>
assert f(16) == 17
</ANSWER>

<PYTHON>
{code}
assert f(??) == {output}
</PYTHON>
<THOUGHT>
"""