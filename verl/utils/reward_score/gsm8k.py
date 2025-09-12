# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Optional
from math_verify import parse, verify


def _last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    Args:
        string: Input string containing LaTeX code
        
    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.
    
    Args:
        s: String with format "\\boxed{content}"
        
    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def _normalize_number(num_str: str) -> str:
    """Normalize a number string by removing commas, dollar signs, and other common units."""
    if num_str is None:
        return None
    
    # Start with the input string
    result = num_str.strip()
    
    # Handle LaTeX formatting (similar to naive_dapo.py)
    # Remove LaTeX text commands: \text{...} -> content
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    
    # Remove LaTeX textbf commands: \textbf{...} -> content  
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    
    # Handle LaTeX escaped characters
    result = result.replace('\\$', '$').replace('\\%', '%')
    
    # Remove dollar signs and percent signs
    result = result.replace("$", "").replace("%", "")
    
    # Remove commas in numbers
    result = result.replace(",", "")
    
    # Remove common unit words at the end (similar to naive_dapo.py approach)
    for unit in ["dollars", "dollar", "cents", "cent", "percent", "percentage", 
                 "points", "point", "units", "unit", "degrees", "degree",
                 "feet", "foot", "inches", "inch", "meters", "meter", "miles", "mile",
                 "cm", "mm", "kg", "pounds", "pound", "hours", "hour",
                 "minutes", "minute", "seconds", "second", "days", "day"]:
        # Remove unit words at the end with optional 's' and whitespace
        pattern = rf'\s*{unit}s?\s*$'
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up any remaining whitespace
    result = result.strip()
    
    # Handle decimal numbers - if it's a valid float that's also an integer, convert to int string
    try:
        if '.' in result:
            float_val = float(result)
            if float_val.is_integer():
                result = str(int(float_val))
    except ValueError:
        pass
        
    return result


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract answer from boxed format in GSM8K solutions.
    
    Args:
        solution_str: The solution string
        
    Returns:
        Extracted and normalized answer or None if not found
    """
    # First try to find boxed answer
    boxed = _last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            extracted = _remove_boxed(boxed)
            return _normalize_number(extracted)
        except:
            pass
    
    # Fallback: try to find #### format (for backward compatibility)
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is not None:
        answer = solution.group(1)
        return _normalize_number(answer)
    
    # Last resort: find the last number in the text
    numbers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
    if numbers:
        # Filter out invalid strings
        invalid_strs = ["", ".", "0."]
        for num in reversed(numbers):
            normalized = _normalize_number(num)
            if normalized not in invalid_strs and len(normalized) > 0:
                return normalized
    
    return None


def compute_score(solution_str: str, ground_truth: str, method="boxed", format_score=0.0, score=1.0):
    """The scoring function for GSM8k using boxed format.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth answer
        method: the method to extract the solution (kept for compatibility, always uses boxed)
        format_score: the score for partial credit (format correct but answer wrong)
        score: the score for the correct answer
    """
    # Extract answer using boxed format
    extracted_answer = extract_boxed_answer(solution_str)
    
    if extracted_answer is None:
        # No answer found at all
        return {"score": 0.0, "acc": 0.0}
    
    # Normalize ground truth
    normalized_gt = _normalize_number(str(ground_truth))
    
    # Check if answers match
    if extracted_answer == normalized_gt:
        return {"score": score, "acc": 1.0}
    else:
        # If our parser says it's wrong, try Math-Verify as fallback
        try:
            gold_parsed = parse(str(ground_truth))
            answer_parsed = parse(str(solution_str))
            if verify(gold_parsed, answer_parsed):
                return {"score": score, "acc": 1.0}
        except:
            pass
        
        # Answer found but incorrect - give partial credit for format
        return {"score": format_score, "acc": 0.0}
