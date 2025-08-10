#!/usr/bin/env python3
"""
Test script for VeRL's built-in reward functions with generated sample responses.
Tests each domain (math, code, logic, stem) with realistic LLM-generated answers.
"""

import sys
import os
from pathlib import Path

# Add verl path
sys.path.append(str(Path(__file__).parent / "verl"))

from verl.utils.reward_score import default_compute_score

def test_math_domain():
    """Test math domain reward scoring."""
    print("\n" + "="*60)
    print("MATH DOMAIN TESTS")
    print("="*60)
    
    tests = [
        {
            "name": "Correct boxed answer",
            "data_source": "lighteval/MATH",
            "response": "The answer is \\boxed{42}",
            "ground_truth": "42",
            "expected": "high"
        },
        {
            "name": "Wrong boxed answer",
            "data_source": "lighteval/MATH",
            "response": "The answer is \\boxed{40}",
            "ground_truth": "42",
            "expected": "low"
        },
        {
            "name": "Correct answer no box",
            "data_source": "lighteval/MATH",
            "response": "The final answer is 42",
            "ground_truth": "42",
            "expected": "medium"
        },
        {
            "name": "GSM8K correct",
            "data_source": "openai/gsm8k",
            "response": "Step 1: Calculate...\nStep 2: ...\nThe answer is 125.\n #### 125",
            "ground_truth": "125",
            "expected": "high"
        }
    ]
    
    for test in tests:
        try:
            score = default_compute_score(
                data_source=test["data_source"],
                solution_str=test["response"],
                ground_truth=test["ground_truth"]
            )
            print(f"✅ {test['name']}: score = {score}")
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {e}")

def test_code_domain():
    """Test code domain reward scoring."""
    print("\n" + "="*60)
    print("CODE DOMAIN TESTS")
    print("="*60)
    
    import json
    
    tests = [
        {
            "name": "LeetCode format - correct solution",
            "data_source": "codegen__leetcode2k",  # Our actual data source
            "response": """```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == s[i-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-2]
                    if p[j-2] == s[i-1] or p[j-2] == '.':
                        dp[i][j] = dp[i][j] or dp[i-1][j]
        
        return dp[m][n]
```""",
            # Use the exact JSON format from the data
            "ground_truth": json.dumps({
                "functional": """def check(candidate):
    assert candidate(s = "aa", p = "a") == False
    assert candidate(s = "aa", p = "a*") == True
    assert candidate(s = "ab", p = ".*") == True

check(Solution().isMatch)"""
            }),
            "expected": "high",
            "extra_info": {}
        },
        {
            "name": "LeetCode format - wrong solution",
            "data_source": "codegen__leetcode2k",
            "response": """```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return s == p  # Wrong: doesn't handle regex
```""",
            # Use the exact JSON format from the data
            "ground_truth": json.dumps({
                "functional": """def check(candidate):
    assert candidate(s = "aa", p = "a*") == True
    assert candidate(s = "ab", p = ".*") == True

check(Solution().isMatch)"""
            }),
            "expected": "low",
            "extra_info": {}
        },
        {
            "name": "TACO format - correct solution",
            "data_source": "codegen__taco",
            "response": """```python
n, s, k = map(int, input().split())
r = list(map(int, input().split()))
colors = input().strip()

# Simple solution for testing
print(4)
```""",
            # Use actual JSON format from fixed data
            "ground_truth": json.dumps({
                "inputs": ["5 3 10\n1 2 3 4 5\nRGBRR\n"],
                "outputs": ["4\n"]
            }),
            "expected": "high",
            "extra_info": {}
        },
        {
            "name": "LiveCodeBench format - correct",
            "data_source": "codegen__livecodebench",
            "response": """```python
t = int(input())
for _ in range(t):
    s = input()
    if s == "abc":
        print("YES")
    elif s in ["acb", "bac", "cba"]:
        print("YES")
    else:
        print("NO")
```""",
            # Use actual JSON format from fixed data
            "ground_truth": json.dumps({
                "inputs": ["6\nabc\nacb\nbac\nbca\ncab\ncba\n"],
                "outputs": ["YES\nYES\nYES\nNO\nNO\nYES\n"]
            }),
            "expected": "high",
            "extra_info": {}
        },
        {
            "name": "PrimeIntellect format - wrong solution",
            "data_source": "codegen__primeintellect",
            "response": """```python
a, b = map(int, input().split())
MOD = 10**9 + 7
result = 0  # Wrong: always outputs 0
print(result % MOD)
```""",
            # Use actual JSON format from fixed data
            "ground_truth": json.dumps({
                "inputs": ["1 4\n", "2 2\n"],
                "outputs": ["30\n", "8\n"]
            }),
            "expected": "low",
            "extra_info": {}
        }
    ]
    
    for test in tests:
        try:
            print(f"\n🧪 Testing: {test['name']}")
            # All ground_truth are already in proper JSON format
            score = default_compute_score(
                data_source=test["data_source"],
                solution_str=test["response"],
                ground_truth=test["ground_truth"],  # Already JSON string
                extra_info=test.get("extra_info", {})
            )
            
            if isinstance(score, dict):
                score_val = score.get("score", score.get("acc", 0))
            else:
                score_val = score
                
            status = "✅" if ((test["expected"] == "high" and score_val > 0.5) or 
                             (test["expected"] == "low" and score_val < 0.5) or
                             test["expected"] == "unknown") else "⚠️"
            print(f"{status} {test['name']}: score = {score}")
        except Exception as e:
            print(f"❌ {test['name']}: {e}")

def test_logic_domain():
    """Test logic domain reward scoring."""
    print("\n" + "="*60)
    print("LOGIC DOMAIN TESTS")
    print("="*60)
    
    import numpy as np
    
    tests = [
        {
            "name": "ARC-AGI correct grid",
            "data_source": "logic__arcagi1",
            "response": """Looking at the pattern, I can see that...
            
            <answer>[[0, 0, 0, 3, 0], [0, 3, 0, 0, 0], [0, 3, 3, 0, 3], [0, 0, 0, 3, 0], [3, 3, 3, 0, 0]]</answer>""",
            # Use proper 2D list format that the scorer expects
            "ground_truth": [[0, 0, 0, 3, 0], [0, 3, 0, 0, 0], [0, 3, 3, 0, 3], [0, 0, 0, 3, 0], [3, 3, 3, 0, 0]],
            "expected": "high"
        },
        {
            "name": "ARC-AGI wrong grid",
            "data_source": "logic__arcagi1",
            "response": """My analysis shows the pattern is...
            
            <answer>[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]</answer>""",
            "ground_truth": [[0, 0, 0, 3, 0], [0, 3, 0, 0, 0], [0, 3, 3, 0, 3], [0, 0, 0, 3, 0], [3, 3, 3, 0, 0]],
            "expected": "low"
        },
        {
            "name": "BARC puzzle correct",
            "data_source": "logic__barc",
            "response": """After analyzing the examples...

        <answer>[[8, 8, 8, 3], [8, 0, 0, 0], [1, 0, 0, 0]]</answer>""",
            "ground_truth": [[8, 8, 8, 3], [8, 0, 0, 0], [1, 0, 0, 0]],
            "expected": "high"
        },
        {
            "name": "Graph logical - correct",
            "data_source": "logic__graph_logical_dataset",
            "response": "The next step is <answer>vovojl</answer>",
            "ground_truth": "vovojl",
            "expected": "high"
        },
        {
            "name": "Ordering puzzle",
            "data_source": "logic__ordering_puzzle_dataset",
            "response": "<answer>['seal', 'rabbit', 'horse', 'kangaroo', 'dolphin', 'wolf']</answer>",
            "ground_truth": ['seal', 'rabbit', 'horse', 'kangaroo', 'dolphin', 'wolf'],
            "expected": "high"
        }
    ]
    
    for test in tests:
        try:
            score = default_compute_score(
                data_source=test["data_source"],
                solution_str=test["response"],
                ground_truth=test["ground_truth"]
            )
            print(f"✅ {test['name']}: score = {score}")
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {e}")

def test_stem_domain():
    """Test STEM domain reward scoring."""
    print("\n" + "="*60)
    print("STEM DOMAIN TESTS")
    print("="*60)
    
    tests = [
        {
            "name": "GPQA correct",
            "data_source": "stem__gpqa",
            "response": "The answer is A",
            "ground_truth": "A",
            "expected": "high"
        },
        {
            "name": "GPQA wrong",
            "data_source": "stem__gpqa",
            "response": "The answer is B",
            "ground_truth": "A",
            "expected": "low"
        },
        {
            "name": "SuperGPQA with box",
            "data_source": "stem__supergpqa",
            "response": "\\boxed{C}",
            "ground_truth": "C",
            "expected": "high"
        }
    ]
    
    for test in tests:
        try:
            score = default_compute_score(
                data_source=test["data_source"],
                solution_str=test["response"],
                ground_truth=test["ground_truth"]
            )
            print(f"✅ {test['name']}: score = {score}")
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {e}")

def main():
    """Run all VeRL built-in reward tests."""
    
    print("="*80)
    print("TESTING VERL BUILT-IN REWARD FUNCTIONS")
    print("="*80)
    print("\nNote: Using VeRL's default_compute_score from verl.utils.reward_score")
    
    # Test each domain
    test_math_domain()
    test_code_domain()
    test_logic_domain()
    test_stem_domain()
    

if __name__ == "__main__":
    main()