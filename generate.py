import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

TARGET_DECODING = "standard_target"
DRAFT_DECODING = "standard_draft"
STANDARD_SD = "standard_sd"
ENTROPY_AWARE_SD = "entropy_aware_sd"


@dataclass(frozen=True)
class RunConfig:
    draft_model: str
    target_model: str
    enable_thinking: bool
    algorithm: str
    entropy_aware_mixing: str
    entropy_transform: str
    seed: int
    temperature: float
    top_k: int
    top_p: float
    gamma: int
    tensor_parallel_size: int
    max_prompt_length: int
    max_new_tokens: int
    num_runs: int
    batch_size: int | None

    @property
    def model_name(self) -> str:
        if self.algorithm == DRAFT_DECODING:
            return self.draft_model
        return self.target_model

    @property
    def uses_entropy_aware_mixing(self) -> bool:
        return bool(
            ALGO_SPECULATIVE_CONFIG.get(self.algorithm, {}).get(
                "use_entropy_aware_mixing"
            )
        )

    @property
    def speculative_config(self) -> dict[str, Any] | None:
        base_config = ALGO_SPECULATIVE_CONFIG.get(self.algorithm)
        if base_config is None:
            return None

        config = {
            "model": self.draft_model,
            "num_speculative_tokens": self.gamma,
            **base_config,
        }
        if self.uses_entropy_aware_mixing:
            config["entropy_aware_mixing"] = self.entropy_aware_mixing
            config["entropy_aware_alpha"] = self.entropy_transform
        return config


ALGORITHMS = [
    TARGET_DECODING,
    DRAFT_DECODING,
    STANDARD_SD,
    ENTROPY_AWARE_SD,
]
ENTROPY_AWARE_MIXINGS = ("geometric", "convex")
ENTROPY_TRANSFORMS = ("linear", "sqrt", "sqrt2", "sq", "constant")


ALGO_SPECULATIVE_CONFIG: dict[str, dict] = {
    STANDARD_SD: {"method": "draft_model"},
    ENTROPY_AWARE_SD: {
        "method": "draft_model",
        "use_entropy_aware_mixing": True,
        "entropy_top_k": 64,
    },
}


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_problems(path: str) -> list[dict]:
    ds = load_dataset("parquet", data_files=path, split="train", streaming=False)

    return [
        {
            "id": idx,
            "messages": row["prompt"],
            "problem": row["prompt"][0]["content"],
            "answer": row["reward_model"]["ground_truth"],
            "reward_type": row["reward_model"]["style"],
            "ability": row.get("ability", "math"),
        }
        for idx, row in enumerate(ds)
    ]


def _default_data_tag(path: str) -> str:
    data_path = Path(path)
    if data_path.name in {"train.parquet", "test.parquet"}:
        tag = data_path.parent.name
    else:
        tag = data_path.stem

    for suffix in ("_500_boxed", "_boxed"):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _completed_request_ids(out_file: Path, num_runs: int) -> set[int]:
    if not out_file.exists():
        return set()

    counts: dict[int, set[int]] = {}
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts.setdefault(int(row["id"]), set()).add(int(row["run_id"]))

    return {idx for idx, run_ids in counts.items() if len(run_ids) >= num_runs}


def _set_rank_cache_env(dp_rank: int) -> None:
    for env_name in (
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        base_dir = os.environ.get(env_name)
        if not base_dir:
            continue
        rank_dir = Path(base_dir) / f"rank_{dp_rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(rank_dir)

    if os.environ.get("VLLM_CACHE_ROOT"):
        os.environ["VLLM_ASSETS_CACHE"] = str(
            Path(os.environ["VLLM_CACHE_ROOT"]) / "assets"
        )


def _filter_overlong_requests(
    requests: list[dict[str, Any]], config: RunConfig
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    kept_requests: list[dict[str, Any]] = []
    skipped_ids: list[int] = []

    for req in requests:
        token_ids = tokenizer.apply_chat_template(
            req["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )
        if len(token_ids) <= config.max_prompt_length:
            kept_requests.append(req)
        else:
            skipped_ids.append(int(req["id"]))

    print(
        f"Prompt-length filter <= {config.max_prompt_length}: "
        f"kept {len(kept_requests)}/{len(requests)} prompts, "
        f"skipped {len(skipped_ids)}",
        flush=True,
    )
    if skipped_ids:
        preview = ", ".join(map(str, skipped_ids[:10]))
        suffix = " ..." if len(skipped_ids) > 10 else ""
        print(
            f"Skipped prompt ids: {preview}{suffix}",
            flush=True,
        )

    return kept_requests


def _create_vllm(config: RunConfig):
    from vllm import LLM

    return LLM(
        model=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_prompt_length + config.max_new_tokens,
        gpu_memory_utilization=0.8,
        seed=config.seed,
        speculative_config=config.speculative_config,
    )


def _shutdown_vllm(llm: Any, dp_rank: int) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:
        print(f"[rank {dp_rank}] vLLM shutdown warning: {exc}", flush=True)
    finally:
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _write_batch_results(
    f,
    request_batch: list[dict[str, Any]],
    batch_results: list[Any],
    elapsed: float,
) -> None:
    total_generations = max(sum(len(result.outputs) for result in batch_results), 1)
    per_gen_time = round(elapsed / total_generations, 4)

    for req, result in zip(request_batch, batch_results):
        for run_idx, output in enumerate(result.outputs):
            f.write(
                json.dumps(
                    {
                        "id": int(req["id"]),
                        "run_id": run_idx,
                        "problem": req["problem"],
                        "generation": output.text,
                        "reference": req["answer"],
                        "reward_type": req["reward_type"],
                        "ability": req["ability"],
                        "runtime": per_gen_time,
                        "num_tokens": len(output.token_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    f.flush()


def _rank_file(out_root: Path, data_tag: str, rank: int) -> Path:
    return out_root / f"{data_tag}_rank{rank}.jsonl"


def _run_single_rank(
    requests: list[dict[str, Any]],
    config: RunConfig,
    dp_rank: int,
    out_file: Path,
    local_device: int | None,
) -> None:
    if local_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_device)
    _set_rank_cache_env(dp_rank)
    print(
        f"[rank {dp_rank}] visible devices = "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
        flush=True,
    )
    if os.environ.get("VLLM_CACHE_ROOT"):
        print(
            f"[rank {dp_rank}] cache root = {os.environ['VLLM_CACHE_ROOT']}",
            flush=True,
        )
    print(f"[rank {dp_rank}] starting with {len(requests)} prompts", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _completed_request_ids(out_file, config.num_runs)
    if completed_ids:
        requests = [req for req in requests if int(req["id"]) not in completed_ids]
        print(
            f"[rank {dp_rank}] skipping {len(completed_ids)} completed prompts; "
            f"{len(requests)} remaining",
            flush=True,
        )
    if not requests:
        print(f"[rank {dp_rank}] shard already complete: {out_file}", flush=True)
        return

    mix_desc = (
        config.entropy_aware_mixing if config.uses_entropy_aware_mixing else "n/a"
    )

    t_load0 = time.time()
    print(
        f"[rank {dp_rank}] loading engine: {config.algorithm}"
        f" (entropy_aware_mixing={mix_desc}, seed={config.seed})",
        flush=True,
    )
    llm = None
    try:
        from vllm import SamplingParams

        llm = _create_vllm(config)
        tokenizer = llm.get_tokenizer()
        stop_token_ids = []
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            stop_token_ids.append(eos_token_id)
        elif isinstance(eos_token_id, list):
            stop_token_ids.extend(
                token_id for token_id in eos_token_id if isinstance(token_id, int)
            )

        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(im_end_token_id, int) or im_end_token_id < 0:
            raise ValueError("Tokenizer does not define the <|im_end|> stop token.")
        stop_token_ids.append(im_end_token_id)
        stop_token_ids = list(dict.fromkeys(stop_token_ids))
        print(
            f"[rank {dp_rank}] engine loaded in {time.time() - t_load0:.2f}s",
            flush=True,
        )
        print(f"[rank {dp_rank}] stop token ids = {stop_token_ids}", flush=True)

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            n=config.num_runs,
            seed=config.seed,
            stop_token_ids=stop_token_ids,
        )
        total_elapsed = 0.0
        processed = 0
        batch_size = config.batch_size or len(requests)
        num_batches = (len(requests) + batch_size - 1) // batch_size
        with open(out_file, "a", encoding="utf-8") as f:
            for batch_index, start in enumerate(
                range(0, len(requests), batch_size), start=1
            ):
                request_batch = requests[start : start + batch_size]
                messages_list = [req["messages"] for req in request_batch]
                print(
                    f"[rank {dp_rank}] llm.chat batch {batch_index}/{num_batches}: "
                    f"{len(messages_list)} prompts with n={config.num_runs}",
                    flush=True,
                )

                t_gen0 = time.time()
                batch_results = llm.chat(
                    messages=messages_list,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": config.enable_thinking},
                )
                elapsed = round(time.time() - t_gen0, 4)
                total_elapsed += elapsed

                _write_batch_results(f, request_batch, batch_results, elapsed)

                processed += len(request_batch)
                print(
                    f"[rank {dp_rank}] wrote batch in {elapsed}s "
                    f"({processed}/{len(requests)} prompts)",
                    flush=True,
                )
                del batch_results

        print(
            f"[rank {dp_rank}] generation done in {round(total_elapsed, 4)}s",
            flush=True,
        )
        print(f"[rank {dp_rank}] wrote {out_file}", flush=True)
    finally:
        if llm is not None:
            _shutdown_vllm(llm, dp_rank)


def _finalize_outputs(
    out_root: Path,
    data_tag: str,
    dp_size: int,
    enable_thinking: bool,
    seed: int,
) -> None:
    from datasets import Dataset
    from verify import score_response

    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for rank in range(dp_size):
        rank_file = _rank_file(out_root, data_tag, rank)
        with open(rank_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = (int(row["id"]), int(row["run_id"]))
                    rows_by_key.setdefault(key, row)

    rows = sorted(
        rows_by_key.values(), key=lambda row: (int(row["id"]), int(row["run_id"]))
    )

    merged = out_root / f"{data_tag}.jsonl"
    verified = out_root / f"{data_tag}_verified.jsonl"
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (
        open(merged, "w", encoding="utf-8") as merged_file,
        open(verified, "w", encoding="utf-8") as verified_file,
    ):
        num_correct = 0
        for row in rows:
            merged_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            ability = row.get("ability", "math")
            reward = score_response(
                row["generation"], row["reference"], ability
            )
            correct = reward >= 1.0 if ability == "code" else reward > 0.0
            scored = {
                **row,
                "reward": reward,
                "correct": correct,
            }
            num_correct += correct
            verified_file.write(json.dumps(scored, ensure_ascii=False) + "\n")
            grouped[int(row["id"])].append(scored)

    rng = random.Random(seed)
    records = []
    for problem_id, examples in sorted(grouped.items()):
        max_reward = max(float(example["reward"]) for example in examples)
        chosen = rng.choice(
            [row for row in examples if float(row["reward"]) == max_reward]
        )
        records.append(
            {
                "data_source": data_tag,
                "prompt": [
                    {"role": "user", "content": chosen["problem"]},
                    {"role": "assistant", "content": chosen["generation"]},
                ],
                "reward_model": {
                    "style": chosen["reward_type"],
                    "ground_truth": chosen["reference"],
                },
                "enable_thinking": enable_thinking,
                "ability": chosen.get("ability", "math"),
                "extra_info": {
                    "id": problem_id,
                    "selection_mode": (
                        "correct" if chosen["correct"] else "incorrect"
                    ),
                    "used_privileged": False,
                },
            }
        )

    train_file = out_root / "train.parquet"
    Dataset.from_list(records).to_parquet(str(train_file))
    for rank in range(dp_size):
        _rank_file(out_root, data_tag, rank).unlink()

    print(f"Wrote {merged}")
    print(f"Wrote {verified} ({num_correct}/{len(rows)} correct)")
    print(f"Wrote {train_file} ({len(records)} rows)")


# ── Core ──────────────────────────────────────────────────────────────────────


def generate(
    draft_model: str,
    target_model: str,
    enable_thinking: bool,
    data_file: str,
    data_tag: str | None,
    output_dir: str,
    num_runs: int,
    algorithm: str,
    seed: int = 0,
    entropy_aware_mixing: str = "geometric",
    entropy_transform: str = "linear",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    gamma: int = 5,
    max_prompt_length: int = 512,
    max_new_tokens: int = 512,
    batch_size: int | None = None,
    tensor_parallel_size: int = 1,
    dp_size: int = 1,
    dp_rank: int | None = None,
    finalize_only: bool = False,
) -> None:
    if dp_size < 1:
        raise ValueError("dp_size must be >= 1")
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    if dp_rank is not None and (dp_rank < 0 or dp_rank >= dp_size):
        raise ValueError(f"dp_rank must be in [0, {dp_size - 1}]")
    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be >= 1")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    config = RunConfig(
        draft_model=draft_model,
        target_model=target_model,
        enable_thinking=enable_thinking,
        algorithm=algorithm,
        entropy_aware_mixing=entropy_aware_mixing,
        entropy_transform=entropy_transform,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        gamma=gamma,
        tensor_parallel_size=tensor_parallel_size,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        num_runs=num_runs,
        batch_size=batch_size,
    )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    data_tag = data_tag or _default_data_tag(data_file)

    if finalize_only:
        _finalize_outputs(
            out_root=out_root,
            data_tag=data_tag,
            dp_size=dp_size,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        return

    print(
        f"\n{'=' * 30}\n"
        f"{config.algorithm} | {data_tag} | seed={config.seed}\n"
        f"output root: {out_root}\n"
        f"{'=' * 30}",
        flush=True,
    )

    requests_all = load_problems(data_file)
    print(f"Loaded {len(requests_all)} prompts", flush=True)
    requests_all = _filter_overlong_requests(requests_all, config)

    if dp_rank is not None:
        rank_requests = requests_all[dp_rank::dp_size]
        print(f"rank {dp_rank}: {len(rank_requests)} prompts", flush=True)
        _run_single_rank(
            requests=rank_requests,
            config=config,
            dp_rank=dp_rank,
            out_file=_rank_file(out_root, data_tag, dp_rank),
            local_device=None,
        )
        return

    if dp_size == 1:
        _run_single_rank(
            requests=requests_all,
            config=config,
            dp_rank=0,
            out_file=_rank_file(out_root, data_tag, 0),
            local_device=0,
        )
        _finalize_outputs(
            out_root=out_root,
            data_tag=data_tag,
            dp_size=dp_size,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        return

    rank_requests_list = [requests_all[rank::dp_size] for rank in range(dp_size)]
    for rank, rank_requests in enumerate(rank_requests_list):
        print(f"rank {rank}: {len(rank_requests)} prompts", flush=True)

    ctx = mp.get_context("spawn")
    procs: list[tuple[int, mp.Process]] = []

    for rank in range(dp_size):
        rank_file = _rank_file(out_root, data_tag, rank)
        proc = ctx.Process(
            target=_run_single_rank,
            kwargs={
                "requests": rank_requests_list[rank],
                "config": config,
                "dp_rank": rank,
                "out_file": rank_file,
                "local_device": rank,
            },
        )
        proc.start()
        procs.append((rank, proc))

    failures = 0
    for rank, proc in procs:
        proc.join()
        print(f"rank {rank} exit code = {proc.exitcode}", flush=True)
        if proc.exitcode != 0:
            failures += 1

    if failures:
        raise RuntimeError(f"{failures} DP ranks failed")

    _finalize_outputs(
        out_root=out_root,
        data_tag=data_tag,
        dp_size=dp_size,
        enable_thinking=enable_thinking,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses with vLLM, verify them, and build an SFT parquet."
        )
    )
    parser.add_argument("--draft-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--target-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallelism size.")
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=os.environ.get("SLURM_PROCID"),
        help="Global rank for Slurm-launched data parallelism.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Merge completed rank files, verify every response, and write the "
            "final train.parquet without running inference."
        ),
    )
    parser.add_argument("--enable-thinking", type=_str_to_bool, default=False)
    parser.add_argument("--data-file", required=True, help="Input parquet file.")
    parser.add_argument(
        "--data-tag",
        default=None,
        help="Output filename stem. Defaults to the parquet parent directory.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--entropy-aware-mixing",
        choices=ENTROPY_AWARE_MIXINGS,
        default="geometric",
    )
    parser.add_argument(
        "--entropy-transform",
        choices=ENTROPY_TRANSFORMS,
        default="linear",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum prompts per llm.chat call. Omit to process each rank shard at once.",
    )
    return parser


def main() -> None:
    options = vars(build_parser().parse_args())
    generate(**options)


if __name__ == "__main__":
    main()
