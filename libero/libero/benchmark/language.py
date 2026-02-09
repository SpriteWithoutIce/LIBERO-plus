#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build language-instruction task ID groups by suite and difficulty level.

Output:
- Print task IDs for each (suite, difficulty)
- Save all groups to language_task_ids.json
"""

import json
from collections import defaultdict
from typing import Dict, List


def build_language_difficulty_groups(
    json_path: str,
) -> Dict[str, Dict[int, List[int]]]:
    """
    Returns:
    {
        suite_name: {
            difficulty_level (1-5): [task_id, ...]
        }
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    groups = defaultdict(lambda: defaultdict(list))

    for suite_name, task_list in data.items():
        for task in task_list:
            if task.get("category") != "Language Instructions":
                continue

            difficulty = task.get("difficulty_level")
            if difficulty not in [1, 2, 3, 4, 5]:
                continue

            task_id = task.get("id")
            if task_id is None:
                continue

            groups[suite_name][difficulty].append(task_id)

    return groups


def main():
    json_path = (
        "/home/jwhe/linyihan/LIBERO-plus/libero/libero/benchmark/"
        "task_classification.json"
    )

    groups = build_language_difficulty_groups(json_path)

    print("=" * 80)
    print("Language Instructions task IDs by suite and difficulty")
    print("=" * 80)

    total_buckets = 0
    total_tasks = 0

    for suite in sorted(groups.keys()):
        print(f"\nSuite: {suite}")
        for d in range(1, 6):
            ids = groups[suite].get(d, [])
            ids = sorted(ids)  # 排个序，方便看 & diff
            print(f"  Difficulty L{d} ({len(ids)} tasks):")
            print(f"    {ids}")

            total_buckets += 1
            total_tasks += len(ids)

    print("\n" + "=" * 80)
    print(f"Total buckets (should be 20): {total_buckets}")
    print(f"Total language tasks: {total_tasks}")
    print("=" * 80)

    # 额外：存成 json，后面 eval 直接用
    output_path = "language_task_ids.json"
    with open(output_path, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"\nSaved language task IDs to: {output_path}")

    return groups


if __name__ == "__main__":
    main()
