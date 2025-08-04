"""
Reward function for guru-RL-92k dataset focusing on 4 domains: math, code, science, logic
"""
from verl.utils.reward_score.math import compute_score as math_score
import re
import json

def extract_answer(solution_str: str, domain: str):
    """Extract answer based on domain-specific patterns"""
    if domain == 'math':
        # Look for boxed answer
        boxed = re.search(r'\\boxed\{([^}]+)\}', solution_str)
        if boxed:
            return boxed.group(1).strip()
        # Look for final answer pattern
        final = re.search(r'(?:final answer|answer is|answer:)\s*([^\n.]+)', solution_str, re.IGNORECASE)
        if final:
            return final.group(1).strip()
            
    elif domain == 'code':
        # Look for code blocks
        code_block = re.search(r'```(?:python|java|cpp|c\+\+|javascript|js)?\n(.*?)\n```', solution_str, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        # Look for function definitions
        func_def = re.search(r'def\s+\w+\s*\([^)]*\):[^}]+', solution_str, re.DOTALL)
        if func_def:
            return func_def.group(0)
            
    elif domain == 'logic':
        # Look for structured answer formats
        answer_patterns = [
            r'(?:answer|solution|result):\s*([^\n]+)',
            r'Therefore,?\s+([^\n]+)',
            r'(?:Yes|No|True|False)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, solution_str, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
                
    return solution_str.strip()

def normalize_answer(answer: str, domain: str):
    """Normalize answer for comparison"""
    answer = str(answer).strip()
    
    if domain == 'math':
        # Remove spaces and normalize math expressions
        answer = answer.replace(' ', '')
        # Handle fractions
        answer = answer.replace('\\frac', '')
        
    elif domain == 'logic':
        # Normalize boolean answers
        answer_lower = answer.lower()
        if answer_lower in ['yes', 'true', '1', 'correct']:
            return 'yes'
        elif answer_lower in ['no', 'false', '0', 'incorrect']:
            return 'no'
            
    return answer

def compute_score(solution_str: str, ground_truth, extra_info=None, **kwargs):
    """
    Compute reward score for 4 domains: math, code, science, logic
    
    The scoring is tailored for each domain while maintaining consistency
    """
    # Get domain information
    data_source = kwargs.get('data_source', '')
    domain = kwargs.get('domain', '')
    
    # Determine domain from data_source if domain not provided
    if not domain:
        if 'math' in data_source:
            domain = 'math'
        elif 'codegen' in data_source or 'code' in data_source:
            domain = 'code'
        elif 'science' in data_source:
            domain = 'science'
        elif 'logic' in data_source:
            domain = 'logic'
        else:
            domain = 'unknown'
    
    # Extract and normalize answers
    extracted_answer = extract_answer(solution_str, domain)
    normalized_solution = normalize_answer(extracted_answer, domain)
    normalized_truth = normalize_answer(ground_truth, domain)
    
    # Domain-specific scoring
    if domain == 'math':
        # Try verl's math scorer first
        try:
            return math_score(solution_str, ground_truth)
        except:
            # Fallback to normalized comparison
            return 1.0 if normalized_solution == normalized_truth else 0.0
    
    elif domain == 'code':
        # Code evaluation - check if ground truth appears in solution
        # For more sophisticated evaluation, integrate with code execution
        if normalized_truth in solution_str:
            return 1.0
        # Check if it's a function that might have different implementation
        if 'def ' in normalized_truth and 'def ' in solution_str:
            # Extract function name and check if it's implemented
            func_name = re.search(r'def\s+(\w+)', normalized_truth)
            if func_name and func_name.group(1) in solution_str:
                return 0.5  # Partial credit for attempting the right function
        return 0.0
    
    elif domain == 'science':
        # Science often has structured answers, try math scorer
        try:
            return math_score(solution_str, ground_truth)
        except:
            # Exact match for non-numeric answers
            return 1.0 if normalized_solution == normalized_truth else 0.0
    
    elif domain == 'logic':
        # Logic puzzles - normalized comparison
        return 1.0 if normalized_solution == normalized_truth else 0.0
    
    else:
        # Unknown domain - try exact match
        return 1.0 if normalized_solution == normalized_truth else 0.0