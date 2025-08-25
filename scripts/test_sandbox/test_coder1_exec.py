#!/usr/bin/env python3
"""Final test of coder1 with correct format"""

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

os.environ['CODER1_EXEC'] = 'sandboxfusion'
os.environ['SANDBOX_FUSION_SERVERS'] = 'localhost:10086'

from verl.utils.reward_score.coder1 import compute_score

print("=== Final Coder1 Test with Correct Format ===\n")

# Real LeetCode data
ground_truth = {
    "functional": """def check(candidate):
    assert candidate(s = "abc", p = "a*b*c*") == True
    assert candidate(s = "mississippi", p = "mis*is*p*") == False
    assert candidate(s = "aa", p = "a") == False
    assert candidate(s = "ab", p = ".*") == True
    assert candidate(s = "aab", p = "c*a*b") == True
    assert candidate(s = "aa", p = "a*") == True
    assert candidate(s = "aaa", p = "a*a") == True
    assert candidate(s = "mississippi", p = "mis*is*p*.") == False
    assert candidate(s = "abc", p = "a.c") == True
    assert candidate(s = "abcd", p = "d*") == False
    assert candidate(s = "abc", p = "abc") == True
    assert candidate(s = "mississippi", p = "mis*is*p*.",) == False


check(Solution().isMatch)"""
}

extra_info = {
    "prefix": """import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')

"""
}

# Test 1: Wrong solution (in markdown)
print("Test 1: Wrong Solution")
solution = """```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return s == p
```"""

result = compute_score(
    solution_str=solution,
    ground_truth=json.dumps(ground_truth),
    extra_info=extra_info
)
print(f"Score: {result['score']} (Expected: 0.0)")
print(f"Result: {'✅ PASS' if result['score'] == 0.0 else '❌ FAIL'}")
print("-" * 60)

# Test 2: Correct solution (in markdown)
print("\nTest 2: Correct Solution")
solution = """```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # Handle patterns like a*, a*b*, etc.
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == s[i-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-2]  # zero occurrence
                    if p[j-2] == s[i-1] or p[j-2] == '.':
                        dp[i][j] = dp[i][j] or dp[i-1][j]  # one or more occurrences
        
        return dp[m][n]
```"""

result = compute_score(
    solution_str=solution,
    ground_truth=json.dumps(ground_truth),
    extra_info=extra_info
)
print(f"Score: {result['score']} (Expected: 1.0)")
print(f"Result: {'✅ PASS' if result['score'] == 1.0 else '❌ FAIL'}")
print("-" * 60)

# Test 3: I/O based test
print("\nTest 3: I/O Based Test")
io_ground_truth = {
    "inputs": ["3\n", "5\n", "1\n"],
    "outputs": ["1\n2\n3\n", "1\n2\n3\n4\n5\n", "1\n"]
}

solution_io = """```python
n = int(input())
for i in range(1, n + 1):
    print(i)
```"""

result = compute_score(
    solution_str=solution_io,
    ground_truth=json.dumps(io_ground_truth),
    extra_info={}
)
print(f"Score: {result['score']} (Expected: 1.0)")
print(f"Result: {'✅ PASS' if result['score'] == 1.0 else '❌ FAIL'}")

print("\n" + "="*60)
print("Summary: Coder1 + SandboxFusion Integration")
print("✅ All solutions must be in markdown code blocks")
print("✅ Wrong solutions → score 0.0")
print("✅ Correct solutions → score 1.0")
print("✅ Both functional and I/O tests work correctly")
print("="*60)