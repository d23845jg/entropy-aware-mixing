# Copyright 2024 PRIME team and/or its affiliates
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

import ast
import json
import multiprocessing
import os
import queue
import re
import platform
import traceback


LEETCODE_PRELUDE = r"""
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6 * 10**5)


class SortedList(list):
    def __init__(self, iterable=()):
        super().__init__(sorted(iterable))

    def add(self, value):
        insort(self, value)

    def discard(self, value):
        index = bisect_left(self, value)
        if index < len(self) and self[index] == value:
            self.pop(index)

    def bisect_left(self, value):
        return bisect_left(self, value)

    def bisect_right(self, value):
        return bisect_right(self, value)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def list_node(values):
    if values is None:
        return None
    dummy = ListNode()
    cur = dummy
    for value in values:
        cur.next = ListNode(value)
        cur = cur.next
    return dummy.next


def list_node_to_list(node):
    values = []
    seen = set()
    while node is not None:
        node_id = id(node)
        if node_id in seen:
            raise ValueError("Cycle detected in ListNode")
        seen.add(node_id)
        values.append(node.val)
        node = node.next
    return values


def is_same_list(left, right):
    return list_node_to_list(left) == list_node_to_list(right)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def tree_node(values):
    if values is None or len(values) == 0:
        return None
    nodes = [None if value is None else TreeNode(value) for value in values]
    kids = nodes[::-1]
    root = kids.pop()
    for node in nodes:
        if node is not None:
            if kids:
                node.left = kids.pop()
            if kids:
                node.right = kids.pop()
    return root


def tree_node_to_list(root):
    if root is None:
        return []
    values = []
    q = deque([root])
    while q:
        node = q.popleft()
        if node is None:
            values.append(None)
            continue
        values.append(node.val)
        q.append(node.left)
        q.append(node.right)
    while values and values[-1] is None:
        values.pop()
    return values


def is_same_tree(left, right):
    return tree_node_to_list(left) == tree_node_to_list(right)
"""


def _extract_python_code(completion):
    if not isinstance(completion, str):
        completion = str(completion)

    python_block = re.search(
        r"```python\s*(.*?)```", completion, flags=re.DOTALL | re.IGNORECASE
    )
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", completion, flags=re.DOTALL)
    if generic_block:
        block = generic_block.group(1)
        lines = block.splitlines()
        if lines and re.fullmatch(r"[A-Za-z0-9_+-]+", lines[0].strip()):
            lines = lines[1:]
        return "\n".join(lines).strip()

    class_index = completion.find("class Solution")
    if class_index != -1:
        return completion[class_index:].strip()

    return completion.strip()


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _leetcode_reliability_guard(maximum_memory_bytes=None):
    import builtins
    import os
    import resource
    import shutil
    import subprocess
    import sys

    if maximum_memory_bytes is not None:
        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if platform.uname().system != "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"

    for name in [
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "chdir",
    ]:
        if hasattr(os, name):
            setattr(os, name, None)

    if platform.uname().system != "Darwin":
        for name in ["lchflags", "lchmod", "lchown"]:
            if hasattr(os, name):
                setattr(os, name, None)

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None  # type: ignore
    __builtins__["help"] = None

    for mod in [
        "subprocess",
        "ctypes",
        "ipdb",
        "joblib",
        "resource",
        "psutil",
        "tkinter",
    ]:
        sys.modules[mod] = None


class _AssertRecorder(ast.NodeTransformer):
    def __init__(self):
        self.total = 0

    def visit_Assert(self, node):
        self.total += 1
        return ast.copy_location(
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_leetcode_record_assert", ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[],
                                vararg=None,
                                kwonlyargs=[],
                                kw_defaults=[],
                                kwarg=None,
                                defaults=[],
                            ),
                            body=node.test,
                        )
                    ],
                    keywords=[
                        ast.keyword(arg="line_no", value=ast.Constant(node.lineno))
                    ],
                )
            ),
            node,
        )


def _instrument_leetcode_tests(test_code):
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return test_code, None

    recorder = _AssertRecorder()
    tree = recorder.visit(tree)
    ast.fix_missing_locations(tree)
    if recorder.total == 0:
        return test_code, None
    return ast.unparse(tree), recorder.total


def _run_leetcode_program(code, result_queue):
    try:
        memory_mb = int(os.environ.get("CODE_VERIFY_MEMORY_MB", "0"))
        memory_bytes = None if memory_mb <= 0 else memory_mb * 1024 * 1024
        _leetcode_reliability_guard(memory_bytes)
        namespace = {}
        exec(code, namespace)
        result_queue.put(
            namespace.get("_LEETCODE_RESULT", {"score": 0.0, "error": "missing result"})
        )
    except Exception:
        result_queue.put(
            {
                "score": 0.0,
                "passed": 0,
                "total": 0,
                "error": traceback.format_exc(limit=10),
            }
        )


def _as_optional_code(value):
    if not value:
        return ""
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
            else:
                chunks.append(str(item))
        value = "\n".join(chunks)
    value = str(value)
    if (
        "class ListNode" in value
        or "class TreeNode" in value
        or "def list_node" in value
        or "def tree_node" in value
    ):
        lines = []
        for line in value.splitlines():
            stripped = line.strip()
            if stripped.startswith(
                "from sortedcontainers import"
            ) or stripped.startswith("import sortedcontainers"):
                continue
            lines.append(line)
        return "\n".join(lines)
    return ""


def compute_leetcode_score(completion, test_cases, timeout=10):
    if not isinstance(test_cases, dict):
        test_cases = json.loads(test_cases)

    solution = _extract_python_code(completion)
    test_code = test_cases.get("test") or ""
    entry_point = test_cases.get("entry_point")
    if not test_code or not entry_point:
        return 0.0, [{"error": "LeetCode test_cases must include test and entry_point"}]

    instrumented_test, total_asserts = _instrument_leetcode_tests(test_code)
    pre_code = _as_optional_code(test_cases.get("pre_code") or test_cases.get("prompt"))
    runner = f"""
_LEETCODE_PASSED = 0
_LEETCODE_TOTAL = 0
_LEETCODE_FAILURES = []


def _leetcode_record_assert(check_fn, line_no=None):
    global _LEETCODE_PASSED, _LEETCODE_TOTAL
    _LEETCODE_TOTAL += 1
    try:
        if check_fn():
            _LEETCODE_PASSED += 1
        else:
            _LEETCODE_FAILURES.append({{"line": line_no, "error": "assertion returned false"}})
    except Exception as exc:
        _LEETCODE_FAILURES.append({{"line": line_no, "error": repr(exc)}})


try:
    check({entry_point})
    if _LEETCODE_TOTAL == 0:
        _LEETCODE_TOTAL = 1
        _LEETCODE_PASSED = 1
except Exception as exc:
    if _LEETCODE_TOTAL == 0:
        _LEETCODE_TOTAL = {total_asserts or 1}
    _LEETCODE_FAILURES.append({{"line": None, "error": repr(exc)}})

_LEETCODE_RESULT = {{
    "score": float(_LEETCODE_PASSED / _LEETCODE_TOTAL) if _LEETCODE_TOTAL else 0.0,
    "passed": _LEETCODE_PASSED,
    "total": _LEETCODE_TOTAL,
    "failures": _LEETCODE_FAILURES[:5],
}}
"""
    program = "\n\n".join(
        [
            LEETCODE_PRELUDE,
            pre_code,
            "# Model solution",
            solution,
            "# LeetCode tests",
            instrumented_test,
            runner,
        ]
    )

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_leetcode_program, args=(program, result_queue)
    )
    process.start()
    process.join(timeout + 1)
    if process.is_alive():
        process.kill()
        process.join()
        return 0.0, [
            {"score": 0.0, "passed": 0, "total": total_asserts or 0, "error": "timeout"}
        ]

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "score": 0.0,
            "passed": 0,
            "total": total_asserts or 0,
            "error": "no result",
        }

    return float(result.get("score", 0.0)), [_json_safe(result)]


def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = _extract_python_code(completion)
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        if (
            isinstance(test_cases, dict)
            and "test" in test_cases
            and "entry_point" in test_cases
        ):
            return compute_leetcode_score(solution, test_cases)

        from .utils import check_correctness as apps_check_correctness

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases, generation=solution, timeout=5, debug=False
            )
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(
                    in_outs=test_case, generation=solution, timeout=10, debug=False
                )
                try:
                    metadata = dict(enumerate(metadata))[
                        0
                    ]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
