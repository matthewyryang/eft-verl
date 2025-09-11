try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def compute_score(data_source, solution_str, ground_truth, extra_info):


    def last_boxed_only_string(string):
        idx = string.rfind("boxed{")
        if idx < 0:
            return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0

        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None

    def remove_boxed(s: str) -> str:
        left = "boxed{"
        if s[:len(left)] != left:
            return None
        if s[-1] != "}":
            return None
        return s[len(left):-1]

    model_answer = last_boxed_only_string(solution_str)
    if model_answer is None:
        return 0.
    model_answer = remove_boxed(model_answer)
    if model_answer is None:
        return 0.
    

    verify_func = math_metric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    model_answer_boxed = "\\boxed{" + model_answer + "}"
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_answer_boxed])
    except Exception as e:
        return 0.
    except TimeoutException:
        return 0.

    return ret_score
