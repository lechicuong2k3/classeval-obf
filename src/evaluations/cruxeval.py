from src.utils.util_crux_eval import check_correctness, check, BASE_IMPORTS

def evaluate_score_cruxeval(args):
    gs, (c, i, o), mode = args
    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            pass
        elif mode == "output" and f"f({i})" in g:
            pass
        else:
            if mode == "input":
                code_to_execute = f"{BASE_IMPORTS}\n{c}\nassert {g} == {o}"
            else:
                code_to_execute = f"{BASE_IMPORTS}\n{c}\nassert {i} == {g}"
            # execution_results.append(check_correctness(code_to_execute, 3))
            execution_results.append(check(code_to_execute))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results