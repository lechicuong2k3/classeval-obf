# Copyright (c) Meta Platforms, Inc. and affiliates.

# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
import json

import ast
import json
import sys
import faulthandler
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal
import numpy as np
# for capturing the stdout
from io import StringIO
# used for testing the code that reads from input
from unittest.mock import patch, mock_open
from pyext import RuntimeModule

from enum import Enum
from tqdm import tqdm
import ipdb


def truncatefn(s, length=300):
    assert isinstance(s, str)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
# timeout = 6  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    final_res = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        final_res.append([f'{sum(all_correct)}/{len(all_correct)}'] + all_correct)
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    # ipdb.set_trace()
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": final_res
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def only_int_check(val):
    return isinstance(val, int)


def string_int_check(val):
    return isinstance(val, str) and val.isdigit()


def combined_int_check(val):
    return only_int_check(val) or string_int_check(val)


def run_test(sample, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]
    

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        assert False, "should not happen: test code is none"
        return in_outs, {"error": "no test code provided"}
    elif test is not None:
        results = []
        sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:

            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    "error_code": -1,
                    "error_message": "Compilation Error",
                }
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it
            try:
                astree = ast.parse(test)
                last_block = astree.body[-1]
                if isinstance(last_block, ast.If):
                    condition = last_block.test
                    if ast.unparse(condition).strip() == "__name__ == '__main__'":
                        test = (
                            ast.unparse(astree.body[:-1])
                            + "\n"
                            + ast.unparse(last_block.body)
                        )
            except:
                pass

            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    "error_code": -1,
                    "error_message": "Compilation Error",
                }
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            return results, {
                "error": repr(e),
                "error_code": -1,
                "error_message": "Unable to extract code",
            }
        
        for index, inputs in enumerate(in_outs["inputs"]):
            raw_inputs = inputs
            raw_outputs = in_outs["outputs"][index]
            if which_type == CODE_TYPE.call_based:
                inputs = [json.loads(line) for line in inputs.split("\n")]
                in_outs["outputs"][index] = json.loads(in_outs["outputs"][index])

                truncate_line_size = 300 // (raw_inputs.count("\n") + 1)
                raw_inputs = "\n".join(
                    [
                        truncatefn(line, truncate_line_size)
                        for line in raw_inputs.strip().split("\n")
                    ]
                )
                raw_outputs = truncatefn(raw_outputs, 200)
            else:
                raw_inputs = truncatefn(raw_inputs)
                raw_outputs = truncatefn(raw_outputs, 200)
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index].items()}
                    ]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index][0].items()}
                    ]
            except:
                True
            

            if debug:
                print(
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}"
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)
                    raw_true_output = output

                    raw_true_output_copy = json.dumps(output)
                    raw_true_output_copy = truncatefn(raw_true_output_copy, 200)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if (
                        isinstance(in_outs["outputs"][index], list)
                        and in_outs["outputs"][index]
                    ):
                        tmp_result = tmp_result or (
                            output == in_outs["outputs"][index][0]
                        )

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or (
                                [list(x) for x in output]
                                == in_outs["outputs"][index][0]
                            )
                    except:
                        True
                    results.append(tmp_result)
                    if tmp_result != True:
                        return results, {
                            "output": raw_true_output_copy,
                            "expected": raw_outputs,
                            "inputs": raw_inputs,
                            "error_code": -2,
                            "error_message": "Wrong Answer",
                        }
                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug:
                        print(
                            f"Standard input runtime error or time limit exceeded error = {e}"
                        )
                    results.append(-1)
                    if "timeoutexception" in repr(e).lower():
                        return results, {
                            "error": repr(e),
                            "error_code": -3,
                            "error_message": "Time Limit Exceeded",
                            "inputs": raw_inputs,
                            "expected": raw_outputs,
                        }
                    else:
                        return results, {
                            "error": repr(e),
                            "error_code": -4,
                            "error_message": "Runtime Error",
                            "inputs": raw_inputs,
                            "expected": raw_outputs,
                        }
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
            

            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                signal.alarm(timeout)
                with Capturing() as output:
                    try:    
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(
                            f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}"
                        )
                        results.append(-1)
                        if "timeoutexception" in repr(e).lower():
                            return results, {
                                "error": repr(e),
                                "error_code": -3,
                                "error_message": "Time Limit Exceeded",
                                "inputs": raw_inputs,
                                "expected": raw_outputs,
                            }
                        else:
                            return results, {
                                "error": repr(e),
                                "error_code": -4,
                                "error_message": "Runtime Error",
                                "inputs": raw_inputs,
                                "expected": raw_outputs,
                            }
                    signal.alarm(0)
                raw_true_output = output[0]
                raw_true_output_copy = truncatefn(raw_true_output, 200)
                output = raw_true_output.splitlines()
                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                    continue

                if passed and debug:
                    print(
                        f"==> output = {output}, test outputs = {in_outs['outputs'][index]}"
                    )

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or (
                                [e.strip() for e in output] == in_outs["outputs"][index]
                            )
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(
                        filter(len, in_outs["outputs"][index])
                    )
                    in_outs["outputs"][index] = list(
                        map(lambda x: x.strip(), in_outs["outputs"][index])
                    )

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}"
                        )
                    else:
                        print(
                            f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}"
                        )

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                if debug:
                    print(f"{tmp_result=} @a")

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")
                    pass

                if debug:
                    print(f"{tmp_result=} @b")

                try:
                    all_ints = all(
                        combined_int_check(e1) and combined_int_check(e2)
                        for e1, e2 in zip(output, in_outs["outputs"][index])
                    )
                    if not all_ints:
                        if debug:
                            print(
                                [
                                    combined_int_check(e1) and combined_int_check(e2)
                                    for e1, e2 in zip(output, in_outs["outputs"][index])
                                ]
                            )
                        output_float = [float(e) for e in output]
                        gt_float = [float(e) for e in in_outs["outputs"][index]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float))
                            and np.allclose(output_float, gt_float)
                        )
                except Exception as e:
                    pass

                if debug:
                    print(f"{tmp_result=} @c")

                try:
                    if isinstance(output[0], list):
                        all_ints = all(
                            combined_int_check(e1) and combined_int_check(e2)
                            for e1, e2 in zip(output[0], in_outs["outputs"][index])
                        )
                        if not all_ints:
                            output_float = [float(e) for e in output[0]]
                            gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                            tmp_result = tmp_result or (
                                (len(output_float) == len(gt_float))
                                and np.allclose(output_float, gt_float)
                            )
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                if debug:
                    print(f"{tmp_result=} @d")
                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                if debug:
                    print(f"{tmp_result=} @e")

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                if debug:
                    print(f"{tmp_result=} @f")

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                if debug:
                    print(f"{tmp_result=} @g")
                # try:
                #     tmp_result = set(frozenset(s) for s in output) == set(
                #         frozenset(s) for s in in_outs["outputs"][index]
                #     )
                # except Exception as e:
                #     if debug:
                #         print(f"Failed check5 exception = {e}")

                # if they are all numbers, round so that similar numbers are treated as identical
                # try:
                #     all_ints = all(
                #         combined_int_check(e1) and combined_int_check(e2)
                #         for e1, e2 in zip(output, in_outs["outputs"][index])
                #     )
                #     tmp_result = tmp_result or (
                #         set(frozenset(round(float(t), 3) for t in s) for s in output)
                #         == set(
                #             frozenset(round(float(t), 3) for t in s)
                #             for s in in_outs["outputs"][index]
                #         )
                #     )
                # except Exception as e:
                #     if debug:
                #         print(f"Failed check6 exception = {e}")

                if debug:
                    print(f"{tmp_result=} @h")

                if tmp_result == True and debug:
                    print("PASSED")

                results.append(tmp_result)
                if tmp_result != True:
                    return results, {
                        "output": raw_true_output_copy,
                        "expected": raw_outputs,
                        "inputs": raw_inputs,
                        "error_code": -2,
                        "error_message": "Wrong Answer",
                    }

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

                    print(f"results = {results}")

    return results, {}


def custom_compare_(output, ground_truth):

    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def check_correctness_lcb(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness_lcb(
                sample, o, timeout=timeout, debug=debug
            )
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            curr_metadata = {}
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)}...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        # assert len(final_metadata[i]) == len(
        #     generations_list[0]
        # ), f"{len(final_metadata[i])=}"

    return [metrics]


def write(content, file):
    with open(file, 'a') as f:
        f.write(content + '\n')