#!/usr/bin/env python3
"""Generate LIBERO object-layout variants by swapping target object with a random distractor.

Rule:
- Keep language instruction unchanged.
- Do not add continuous perturbation.
- Swap initial regions between the task target object and one randomly selected distractor object.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _extract_goal_target_object(bddl_text: str) -> str:
    m = re.search(r"\(:goal\s*\(And\s*\((In|On)\s+([^\s\)]+)", bddl_text, flags=re.S)
    if not m:
        raise ValueError("Cannot infer target object from goal section")
    return m.group(2)


def _extract_init_section(bddl_text: str) -> Tuple[str, str, str]:
    m = re.search(r"(\(:init\s*)(.*?)(\)\s*\(:goal)", bddl_text, flags=re.S)
    if not m:
        raise ValueError("Cannot locate :init section")
    return m.group(1), m.group(2), m.group(3)


def _parse_on_states(init_body: str) -> List[Tuple[str, str]]:
    return re.findall(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", init_body)


def _pick_distractor(on_states: List[Tuple[str, str]], target_obj: str, rng: random.Random) -> str:
    candidates = [obj for obj, region in on_states if obj != target_obj and "other_object_region" in region]
    if not candidates:
        candidates = [obj for obj, _ in on_states if obj != target_obj]
    if not candidates:
        raise ValueError("No distractor object found")
    return rng.choice(candidates)


def _swap_regions(on_states: List[Tuple[str, str]], obj_a: str, obj_b: str) -> Dict[str, str]:
    region_map = dict(on_states)
    if obj_a not in region_map or obj_b not in region_map:
        raise ValueError(f"Cannot find regions for swap pair: {obj_a}, {obj_b}")
    region_map[obj_a], region_map[obj_b] = region_map[obj_b], region_map[obj_a]
    return region_map


def _rewrite_init(init_body: str, swapped_region_map: Dict[str, str]) -> str:
    def repl(match):
        obj = match.group(1)
        old_region = match.group(2)
        if obj in swapped_region_map:
            return f"(On {obj} {swapped_region_map[obj]})"
        return f"(On {obj} {old_region})"

    return re.sub(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", repl, init_body)


def _find_suite_list_bounds(text: str, suite: str) -> Tuple[int, int]:
    key = f'"{suite}": ['
    start = text.find(key)
    if start == -1:
        raise ValueError(f"Cannot find suite key {suite} in suite map")
    i = start + len(key)
    depth = 1
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return start, i
        i += 1
    raise ValueError(f"Cannot locate closing bracket for suite {suite}")


def register_in_suite_task_map(task_name: str, suite: str = "libero_object"):
    map_path = Path("libero/libero/benchmark/libero_suite_task_map.py")
    text = map_path.read_text()
    if f'"{task_name}"' in text:
        return False

    _start, end = _find_suite_list_bounds(text, suite)
    insertion = f'        "{task_name}",\n'
    new_text = text[:end] + insertion + text[end:]
    map_path.write_text(new_text)
    return True


def register_in_task_classification(task_name: str, suite: str = "libero_object", category: str = "Objects Layout", difficulty_level: int = 3):
    cls_path = Path("libero/libero/benchmark/task_classification.json")
    obj = json.loads(cls_path.read_text())
    suite_list = obj[suite]
    if any(item["name"] == task_name for item in suite_list):
        return False

    next_id = max(item["id"] for item in suite_list) + 1 if suite_list else 1
    suite_list.append(
        {
            "id": next_id,
            "name": task_name,
            "category": category,
            "difficulty_level": int(difficulty_level),
        }
    )
    cls_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    return True


def update_task_num_for_suite(suite: str = "libero_object", delta: int = 1):
    init_path = Path("libero/libero/benchmark/__init__.py")
    text = init_path.read_text()

    m_suite = re.search(r"suite_order\s*=\s*\[(.*?)\]", text, flags=re.S)
    m_num = re.search(r"task_num\s*=\s*\[(.*?)\]", text, flags=re.S)
    if not m_suite or not m_num:
        raise ValueError("Cannot find suite_order/task_num in benchmark __init__.py")

    suites = [s.strip().strip('"\'') for s in m_suite.group(1).split(",") if s.strip()]
    nums = [int(x.strip()) for x in m_num.group(1).split(",") if x.strip()]
    if suite not in suites:
        raise ValueError(f"Suite {suite} is not in suite_order")

    idx = suites.index(suite)
    nums[idx] += delta
    new_num_literal = "[" + ", ".join(str(x) for x in nums) + "]"
    new_text = text[: m_num.start()] + f"task_num = {new_num_literal}" + text[m_num.end() :]
    init_path.write_text(new_text)


def generate_one(src_bddl: Path, dst_bddl: Path, seed: int, dry_run: bool = False):
    bddl_text = src_bddl.read_text()
    target_obj = _extract_goal_target_object(bddl_text)

    init_prefix, init_body, init_suffix = _extract_init_section(bddl_text)
    on_states = _parse_on_states(init_body)
    rng = random.Random(f"{seed}:{src_bddl.stem}")
    distractor_obj = _pick_distractor(on_states, target_obj, rng)

    swapped_region_map = _swap_regions(on_states, target_obj, distractor_obj)
    new_init_body = _rewrite_init(init_body, swapped_region_map)
    new_bddl_text = bddl_text.replace(init_prefix + init_body + init_suffix, init_prefix + new_init_body + init_suffix)

    if not dry_run:
        dst_bddl.parent.mkdir(parents=True, exist_ok=True)
        dst_bddl.write_text(new_bddl_text)

    return {
        "base_task": src_bddl.stem,
        "new_task": dst_bddl.stem,
        "target_object": target_obj,
        "swapped_with": distractor_obj,
        "source_bddl": str(src_bddl),
        "output_bddl": str(dst_bddl),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", default=None, help="single task name without .bddl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default="table_1")
    parser.add_argument("--manifest", default="/tmp/libero_object_swap_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--register-benchmark", action="store_true", help="register generated tasks into suite map and task classification")
    parser.add_argument("--classification-category", default="Objects Layout")
    parser.add_argument("--classification-difficulty", type=int, default=3)
    parser.add_argument("--update-task-num", action="store_true", help="increment benchmark task_num for the suite")
    args = parser.parse_args()

    bddl_root = Path("libero/libero/bddl_files") / args.suite
    src_files = [bddl_root / f"{args.task}.bddl"] if args.task else sorted(bddl_root.glob("*.bddl"))

    report = []
    for src in src_files:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = src.with_name(f"{src.stem}_{args.suffix}.bddl")
        report.append(generate_one(src, dst, seed=args.seed, dry_run=args.dry_run))

    if args.register_benchmark and not args.dry_run:
        added_count = 0
        for item in report:
            added_map = register_in_suite_task_map(item["new_task"], suite=args.suite)
            added_cls = register_in_task_classification(
                item["new_task"],
                suite=args.suite,
                category=args.classification_category,
                difficulty_level=args.classification_difficulty,
            )
            if added_map or added_cls:
                added_count += 1
        if args.update_task_num and added_count > 0:
            update_task_num_for_suite(suite=args.suite, delta=added_count)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Generated {len(report)} tasks. Manifest: {args.manifest}")
    if report:
        print("Example:", json.dumps(report[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
