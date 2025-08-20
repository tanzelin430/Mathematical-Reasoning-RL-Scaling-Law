"""
Daytona-based code execution for VERL reward computation
"""

import json
import re
import os
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from daytona import Daytona, DaytonaConfig
except ImportError:
    Daytona = None
    DaytonaConfig = None

#Use the Code1 
CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
ANSWER_PATTERN = re.compile(r"</think>(.*)", re.DOTALL)


_daytona_client = None
_config = None

def initialize_daytona(api_key: str, max_concurrent: int = 8):
    global _daytona_client, _config
    
    if Daytona is None or DaytonaConfig is None:
        raise ImportError("Daytona package not installed. Please install with: pip install daytona")
    
    if _daytona_client is None:
        _config = DaytonaConfig(api_key=api_key)
        _daytona_client = Daytona(_config)
    return _daytona_client

def try_extract_solution(solution_str: str) -> str:
    match = re.search(ANSWER_PATTERN, solution_str)
    if match:
        return match.group(1).strip()
    return solution_str

def extract_code_from_string(solution_str: str) -> str:
    solution_str = try_extract_solution(solution_str)  # step 1: process </think>
    code_blocks = CODE_PATTERN.findall(solution_str)  # step 2: extract code blocks
    return "\n".join(code_blocks).strip()  # step 3: merge all code blocks

def fuzzy_equal(actual: str, expected: str, tolerance: float = 1e-6) -> bool:
    actual = actual.strip().replace("\r\n", "\n")
    expected = expected.strip().replace("\r\n", "\n")

    # If exact match after normalization, return early
    if actual == expected:
        return True

    # Split into lines
    actual_lines = actual.split("\n")
    expected_lines = expected.split("\n")

    # If different number of lines, they're definitely not equal
    if len(actual_lines) != len(expected_lines):
        return False

    # Compare each line
    for i, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        # If lines match exactly, continue
        if actual_line == expected_line:
            continue

        # Split into tokens by whitespace
        actual_tokens = actual_line.split()
        expected_tokens = expected_line.split()

        # If different number of tokens, they're not equal
        if len(actual_tokens) != len(expected_tokens):
            return False

        # Compare each token
        for j, (actual_token, expected_token) in enumerate(zip(actual_tokens, expected_tokens)):
            # If tokens match exactly, continue
            if actual_token == expected_token:
                continue

            # For yes/no, use case-insensitive comparison
            if actual_token.lower() in ["yes", "no"] and expected_token.lower() in ["yes", "no"]:
                if actual_token.lower() == expected_token.lower():
                    continue
                else:
                    return False

            # Try numeric comparison
            try:
                actual_num = float(actual_token)
                expected_num = float(expected_token)
                diff = abs(actual_num - expected_num)

                if diff <= tolerance:
                    continue
                else:
                    return False
            except ValueError:
                # Not numeric values
                return False

    # If we made it here, all lines are approximately equal
    return True

def execute_code_in_daytona(code: str, stdin: Optional[str] = None, timeout: int = 30) -> tuple[bool, str]:
    global _daytona_client
    
    if _daytona_client is None:
        raise RuntimeError("Daytona client not initialized. Call initialize_daytona() first.")
    
    sandbox = None
    try:
        # create sandbox
        sandbox = _daytona_client.create()
        
        # prepare to execute code - automatically add common imports
        common_imports = f"from typing import *\n"
        exec_code = common_imports + code
        if stdin:
            # if there is stdin, add input simulation
            exec_code += f"\n\nimport io\nimport sys\nsys.stdin = io.StringIO('''{stdin}''')\n"
        
        # execute code
        response = sandbox.process.code_run(exec_code, timeout=timeout)
        
        success = response.exit_code == 0
        output = response.result if hasattr(response, 'result') else ""
        

        if not success and hasattr(response, 'stderr') and response.stderr:
            output = f"ERROR: {response.stderr}\n{output}"
            
        return success, output
        
    except Exception as e:
        return False, f"Execution failed: {str(e)}"
    finally:
        if sandbox:
            try:
                sandbox.delete()
            except:
                pass  

def remote_check_stdio(code: str, stdin: str, stdout: str) -> tuple[bool, str, str, str]:
    success, output = execute_code_in_daytona(code, stdin, timeout=5)
    return success, output, stdin, stdout

def _compute_score(solution_str: str, ground_truth: str, extra_info: Dict[str, Any], 
                  format_reward: float = 0.0, answer_reward: float = 1.0) -> float:

    

    solution_code = extract_code_from_string(solution_str)
    
    if len(solution_code) == 0:
        return 0.0
    
    ground_truth_data = json.loads(ground_truth)
    

    if "pytest" in ground_truth_data or "functional" in ground_truth_data or "solution_file" in ground_truth_data:
        if "functional" in ground_truth_data:
            if "prefix" in extra_info and extra_info["prefix"] is not None:
                solution_code = extra_info["prefix"] + "\n" + solution_code
            success, output = execute_code_in_daytona(solution_code + "\n" + ground_truth_data["functional"], timeout=5)
        elif "solution_file" in ground_truth_data:
            success, output = execute_code_in_daytona(solution_code + "\n" + ground_truth_data["solution_file"], timeout=5)
        else:  # pytest
            success, output = execute_code_in_daytona(solution_code + "\n" + ground_truth_data["pytest"], timeout=5)
        
        if not success:
            return format_reward
            
    elif "inputs" in ground_truth_data and "outputs" in ground_truth_data:
        stdin_list = ground_truth_data["inputs"]
        stdout_list = ground_truth_data["outputs"]
        
        for i, (stdin, stdout) in enumerate(zip(stdin_list, stdout_list)):
            success, output = execute_code_in_daytona(solution_code, stdin, timeout=5)
            if not success or not fuzzy_equal(output.strip(), stdout.strip()):
                return format_reward
    else:
        raise ValueError(f"Current supports for ground-truth are ['pytest', 'functional', 'solution_file', 'inputs/outputs'] -- No idea what's: {ground_truth_data = }")
    
    return format_reward + answer_reward

def compute_score(solution_str: str, ground_truth: str, extra_info: Any, 
                 format_reward: float = 0.0, answer_reward: float = 1.0) -> Dict[str, Any]:

    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    

    api_key = os.environ.get('DAYTONA_API_KEY') or extra_info.get('daytona_api_key')
    if not api_key:
        raise ValueError("DAYTONA_API_KEY not found in environment or extra_info")

    initialize_daytona(api_key)
    
    score = _compute_score(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )
    
    return {"score": score, "acc": score}