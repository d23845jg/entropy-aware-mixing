#!/usr/bin/env python3
"""Evaluate a model on rule-verifiable benchmarks with vLLM."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    k: int
    max_tokens: int
    path: Path


def load_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation parquet: {path}")

    rows = []
    for row_idx, row in enumerate(pd.read_parquet(path).to_dict(orient="records")):
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        rows.append(
            {
                "row_idx": row_idx,
                "problem_id": str(row.get("extra_info", {}).get("index", row_idx)),
                "messages": prompt,
                "ability": row.get("ability", "math"),
                "ground_truth": row["reward_model"]["ground_truth"],
            }
        )
    return rows


def score_response(response: str, ground_truth: Any, ability: str) -> float:
    if ability == "math":
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        verify = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )
        try:
            score, _ = verify([f"\\boxed{{{ground_truth}}}"], [response])
            return float(score)
        except TimeoutException:
            return 0.0
        except Exception:
            return 0.0

    if ability == "code":
        from verl.utils.reward_score.prime_code import compute_score

        result = compute_score(response, ground_truth)
        if isinstance(result, tuple):
            return float(result[0])
        if isinstance(result, dict):
            return float(result.get("score", 0.0))
        return float(result)

    raise ValueError(f"Unsupported evaluation ability: {ability}")


def set_rank_cache_env(rank: int) -> None:
    for env_name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        cache_root = os.environ.get(env_name)
        if cache_root:
            rank_dir = Path(cache_root) / f"rank_{rank}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(rank_dir)

    if os.environ.get("VLLM_CACHE_ROOT"):
        os.environ["VLLM_ASSETS_CACHE"] = str(
            Path(os.environ["VLLM_CACHE_ROOT"]) / "assets"
        )


def generate_samples(
    args: argparse.Namespace,
    llm: Any,
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    if not rows:
        return []

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=spec.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=spec.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=args.show_progress,
    )

    samples = []
    for row, prompt, request_output in zip(rows, prompts, outputs, strict=True):
        if len(request_output.outputs) != spec.k:
            raise RuntimeError(
                f"{spec.name} row {row['row_idx']} produced "
                f"{len(request_output.outputs)} samples, expected {spec.k}"
            )

        for sample_idx, output in enumerate(request_output.outputs):
            response = output.text
            score = score_response(response, row["ground_truth"], row["ability"])
            samples.append(
                {
                    "dataset": spec.name,
                    "row_idx": row["row_idx"],
                    "problem_id": row["problem_id"],
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "ground_truth": row["ground_truth"],
                    "response": response,
                    "score": score,
                    "correct": (
                        score >= 1.0 if row["ability"] == "code" else score > 0.0
                    ),
                    "num_tokens": len(output.token_ids or []),
                    "finish_reason": output.finish_reason,
                    "cumulative_logprob": float(output.cumulative_logprob or 0.0),
                }
            )
    return samples


def run_rank(args: argparse.Namespace, model: str, output_base_dir: Path) -> None:
    import pandas as pd
    from vllm import LLM

    rank = args.dp_rank
    set_rank_cache_env(rank)
    print(
        f"[rank {rank}] visible devices="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
        flush=True,
    )

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)

    for spec in args.dataset_spec:
        rows = load_rows(spec.path)[rank :: args.dp_size]
        output_dir = output_base_dir / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rank {rank}] generating {spec.name}: rows={len(rows)}, "
            f"k={spec.k}, max_tokens={spec.max_tokens}",
            flush=True,
        )
        rank_path = output_dir / f"rank_{rank}.parquet"
        samples = generate_samples(args, llm, spec, rows)
        pd.DataFrame(samples).to_parquet(rank_path, index=False)
        print(f"[rank {rank}] wrote {rank_path}", flush=True)


def aggregate_metrics(
    rows: list[dict[str, Any]], dataset: str, requested_k: int
) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)

    lengths = sorted(int(row["num_tokens"]) for row in rows)
    metrics: dict[str, Any] = {
        "requested_k": requested_k,
        "problem_count": len(by_problem),
        "sample_count": len(rows),
        "mean_sample_acc": (
            sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
        ),
        "num_tokens_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "num_tokens_min": lengths[0] if lengths else 0,
        "num_tokens_max": lengths[-1] if lengths else 0,
    }

    report_ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= requested_k]
    if not report_ks or report_ks[-1] != requested_k:
        report_ks.append(requested_k)
    for k in report_ks:
        metrics[f"pass@{k}"] = (
            sum(
                1
                - comb(
                    len(problem_rows) - sum(row["correct"] for row in problem_rows),
                    k,
                )
                / comb(len(problem_rows), k)
                for problem_rows in by_problem.values()
            )
            / len(by_problem)
            if by_problem
            else 0.0
        )

    overall = {key: value for key, value in metrics.items() if key != "requested_k"}
    return {"datasets": {dataset: metrics}, "overall": overall}


def parse_dataset_spec(raw_spec: str) -> DatasetSpec:
    try:
        name, k, max_tokens, path = raw_spec.split(":", 3)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dataset specs must be formatted as NAME:K:MAX_TOKENS:PATH"
        ) from exc

    try:
        parsed_k = int(k)
        parsed_max_tokens = int(max_tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"K and MAX_TOKENS must be integers in dataset spec: {raw_spec}"
        ) from exc
    if parsed_k <= 0 or parsed_max_tokens <= 0:
        raise argparse.ArgumentTypeError(
            f"K and MAX_TOKENS must be positive in dataset spec: {raw_spec}"
        )

    return DatasetSpec(name, parsed_k, parsed_max_tokens, Path(path))


def resolve_model(model: str, seed: int) -> tuple[str, Path]:
    model_path = Path(model).expanduser()
    if model_path.exists() or model.startswith(("/", ".")):
        resolved = model_path.resolve()
        if not (resolved / "config.json").is_file():
            raise FileNotFoundError(
                "Expected a Hugging Face model directory containing config.json: "
                f"{resolved}"
            )
        return str(resolved), resolved / f"evals_{seed}"
    return model, Path(f"evals_{seed}") / model.replace("/", "__")


def aggregate_outputs(args: argparse.Namespace, output_base_dir: Path) -> None:
    import pandas as pd

    for spec in args.dataset_spec:
        output_dir = output_base_dir / spec.name
        samples = pd.concat(
            [
                pd.read_parquet(output_dir / f"rank_{rank}.parquet")
                for rank in range(args.dp_size)
            ],
            ignore_index=True,
        ).sort_values(["row_idx", "sample_idx"])

        samples_path = output_dir / "samples.parquet"
        metrics_path = output_dir / "metrics.json"
        samples.to_parquet(samples_path, index=False)
        metrics_path.write_text(
            json.dumps(
                aggregate_metrics(samples.to_dict(orient="records"), spec.name, spec.k),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {metrics_path}")
        print(f"Wrote {samples_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model directory or model ID")
    parser.add_argument(
        "--dataset-spec",
        action="append",
        required=True,
        type=parse_dataset_spec,
        help="NAME:K:MAX_TOKENS:PATH; may be repeated",
    )
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=os.environ.get("SLURM_PROCID", 0),
        help="Global rank. Defaults to SLURM_PROCID, or 0 outside Slurm.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Merge completed rank files and write metrics without inference.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dp_size <= 0:
        raise ValueError(f"dp-size must be positive, got {args.dp_size}")
    if not 0 <= args.dp_rank < args.dp_size:
        raise ValueError(f"dp-rank must be in [0, {args.dp_size}), got {args.dp_rank}")

    model, output_base_dir = resolve_model(args.model, args.seed)
    if args.aggregate_only:
        aggregate_outputs(args, output_base_dir)
    else:
        run_rank(args, model, output_base_dir)


if __name__ == "__main__":
    main()
