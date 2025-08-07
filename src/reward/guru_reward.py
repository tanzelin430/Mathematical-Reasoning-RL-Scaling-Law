"""
Improved reward function for guru-RL-92k dataset with proper code execution evaluation
"""
from verl.utils.reward_score.math import compute_score as math_score
import re
import json
import sys
import io
import contextlib
from contextlib import redirect_stdout, redirect_stderr

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
        # Extract code blocks with improved patterns
        # First try to find code blocks with explicit language markers
        code_block = re.search(r'```(?:python|py)\s*\n(.*?)\n```', solution_str, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
            
        # Try generic code blocks
        code_block = re.search(r'```\s*\n(.*?)\n```', solution_str, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
            
        # Look for class definitions (for LeetCode-style problems)
        class_def = re.search(r'class\s+Solution.*?(?=\n(?:\S|$))', solution_str, re.DOTALL)
        if class_def:
            return class_def.group(0).strip()
            
        # Look for complete function definitions
        func_def = re.search(r'def\s+\w+\s*\([^)]*\):\s*.*?(?=\n(?:\S|$))', solution_str, re.DOTALL)
        if func_def:
            return func_def.group(0).strip()
            
        # If solution_str itself looks like code, return it
        if ('class ' in solution_str or 'def ' in solution_str) and len(solution_str.strip()) > 10:
            return solution_str.strip()
            
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

def safe_execute_code(code: str, test_code: str, timeout_seconds=5):
    """
    Safely execute code with test cases and return pass rate
    """
    try:
        # Create a comprehensive but restricted environment
        import math
        import itertools
        import operator
        import functools
        import collections
        
        # Create a safe builtins dictionary based on the default one
        safe_builtins = {
            # Core language constructs
            '__build_class__': __builtins__['__build_class__'],
            '__import__': __builtins__['__import__'],
            '__name__': __builtins__['__name__'],
            
            # Basic types and constructors
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'pow': pow,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'ord': ord, 'chr': chr, 'bin': bin, 'hex': hex, 'oct': oct,
            
            # Type checking and reflection
            'isinstance': isinstance, 'hasattr': hasattr, 'callable': callable,
            'getattr': getattr, 'setattr': setattr, 'type': type,
            
            # Iteration and slicing
            'slice': slice, 'iter': iter, 'next': next, 'reversed': reversed,
            
            # Math and comparison
            'round': round, 'divmod': divmod, 'all': all, 'any': any,
            
            # IO (limited)
            'print': print,
            
            # Exception handling
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'IndexError': IndexError, 'KeyError': KeyError, 'AttributeError': AttributeError,
            'AssertionError': AssertionError, 'NameError': NameError,
        }
        
        exec_globals = {
            '__builtins__': safe_builtins,
            # Standard library modules (safe subset)
            'math': math,
            'itertools': itertools,
            'operator': operator,
            'functools': functools,
            'collections': collections,
        }
        
        # Execute the solution code first
        exec(code, exec_globals)
        
        # Now execute the test code to get the test function
        test_globals = exec_globals.copy()
        exec(test_code, test_globals)
        
        # All tests passed if we reach here
        return 1.0
        
    except AssertionError as e:
        # Some tests failed - try to count passed vs failed tests
        return count_passed_tests(code, test_code, exec_globals)
        
    except SyntaxError as e:
        # Syntax error in code
        return 0.0
        
    except Exception as e:
        # Runtime error (NameError, TypeError, etc.)
        return 0.0


def count_passed_tests(code: str, test_code: str, base_globals: dict = None) -> float:
    """
    Count how many individual tests pass by running them one by one
    """
    try:
        # Extract individual assert statements from test code
        assert_pattern = r'assert\s+candidate\([^)]+\)\s*==\s*[^\n]+'
        asserts = re.findall(assert_pattern, test_code)
        
        if not asserts:
            return 0.0
            
        passed = 0
        total = len(asserts)
        
        for assert_stmt in asserts:
            try:
                # Create a fresh environment for each test
                test_globals = base_globals.copy()
                exec(code, test_globals)
                
                # Replace 'candidate' with the actual function call
                # Try to find Solution().methodName pattern directly
                solution_call_match = re.search(r'Solution\(\)\.(\w+)', test_code)
                if solution_call_match:
                    method_name = solution_call_match.group(1)
                    func_call = f'Solution().{method_name}'
                    test_stmt = assert_stmt.replace('candidate', func_call)
                    exec(test_stmt, test_globals)
                    passed += 1
                else:
                    # Fallback: look for any function call pattern in check()
                    check_call_match = re.search(r'check\((.+?)\)', test_code)
                    if check_call_match:
                        func_call = check_call_match.group(1).strip()
                        # If it's just 'candidate', use Solution() default
                        if func_call == 'candidate':
                            func_call = 'Solution().solve'  # Generic method name
                        test_stmt = assert_stmt.replace('candidate', func_call)
                        exec(test_stmt, test_globals)
                        passed += 1
                    
            except AssertionError:
                # This specific test failed
                continue
            except Exception:
                # Runtime error on this test
                continue
                
        return passed / total if total > 0 else 0.0
        
    except Exception:
        return 0.0

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
        elif 'science' in data_source or 'stem' in data_source:
            domain = 'science'
        elif 'logic' in data_source:
            domain = 'logic'
        else:
            domain = 'unknown'
    
    # Parse ground_truth if it's JSON (common in MATH domain)
    actual_ground_truth = ground_truth
    if isinstance(ground_truth, str) and domain in ['math', 'science']:
        try:
            # Try to parse as JSON
            parsed = json.loads(ground_truth)
            # If parsed successfully, convert back to string for comparison
            actual_ground_truth = str(parsed)
        except:
            # Not JSON, use as is
            actual_ground_truth = ground_truth
    
    # Extract and normalize answers
    extracted_answer = extract_answer(solution_str, domain)
    normalized_solution = normalize_answer(extracted_answer, domain)
    normalized_truth = normalize_answer(actual_ground_truth, domain)
    
    # Domain-specific scoring
    if domain == 'math':
        # Try verl's math scorer first with actual_ground_truth
        try:
            verl_score = math_score(solution_str, actual_ground_truth)
            # Only trust VeRL if it gives a positive score
            if verl_score > 0:
                return verl_score
        except:
            pass
        
        # Fallback to normalized comparison
        return 1.0 if normalized_solution == normalized_truth else 0.0
    
    elif domain == 'code':
        # Code evaluation with unit test execution
        try:
            # Parse ground truth to extract test code
            if isinstance(ground_truth, dict) and 'functional' in ground_truth:
                test_code = ground_truth['functional']
            elif isinstance(ground_truth, str):
                # Try to parse JSON if it's a string
                try:
                    gt_dict = json.loads(ground_truth)
                    if 'functional' in gt_dict:
                        test_code = gt_dict['functional']
                    else:
                        test_code = ground_truth
                except:
                    test_code = ground_truth
            else:
                test_code = str(ground_truth)
            
            # Validate that we have proper code structure
            if not extracted_answer or not extracted_answer.strip():
                return 0.0
                
            # Check for basic code structure
            has_class = 'class Solution' in extracted_answer
            has_method = 'def ' in extracted_answer
            
            if not (has_class and has_method):
                # No proper structure
                return 0.0
                
            # Execute code with tests
            score = safe_execute_code(extracted_answer, test_code)
            return score
            
        except Exception as e:
            # Fallback evaluation
            # Give minimal credit for having proper code structure
            if 'class Solution' in extracted_answer and 'def ' in extracted_answer:
                return 0.05  # Very small credit for structure
            return 0.0
    
    elif domain == 'science':
        # Science often has structured answers, try math scorer
        try:
            verl_score = math_score(solution_str, actual_ground_truth)
            # Only trust VeRL if it gives a positive score
            if verl_score > 0:
                return verl_score
        except:
            pass
        
        # Fallback to exact match for non-numeric answers
        return 1.0 if normalized_solution == normalized_truth else 0.0
    
    elif domain == 'logic':
        # Logic puzzles - normalized comparison
        return 1.0 if normalized_solution == normalized_truth else 0.0
    
    else:
        # Unknown domain - try exact match
        return 1.0 if normalized_solution == normalized_truth else 0.0