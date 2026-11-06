"""Lightweight execution metadata for batched evaluation (no model imports)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


DEFAULT_TEST_LOC_BATCH_SIZE = 32
DEFAULT_TEST_GEN_BATCH_SIZE = 16
BATCH_IMPLEMENTATION = "unilip_inference_batch_v1"
RNG_POLICY = "legacy_ordered_initial_noise_draws_v1"


def positive_batch_size(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Inference batch size must be a positive integer")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inference batch size must be a positive integer") from exc
    if size < 1:
        raise ValueError("Inference batch size must be a positive integer")
    return size


def execution_contract(task: str, batch_size: int) -> dict:
    if task not in ("loc", "gen", "gen_continuous"):
        raise ValueError(f"Unknown inference task: {task}")
    return {
        "schema_version": 1,
        "task": "loc" if task == "loc" else "gen",
        "batch_size": positive_batch_size(batch_size),
        "implementation": BATCH_IMPLEMENTATION,
        "rng_policy": RNG_POLICY,
    }


def validate_execution_metadata(payload: dict, task: str, batch_size: int) -> None:
    """A legacy manifest without execution metadata is valid only for batch 1."""
    expected = execution_contract(task, batch_size)
    actual = payload.get("inference_execution")
    if actual is None and expected["batch_size"] == 1:
        return
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("Inference batch size, implementation or RNG policy differs from saved output")


def _checkpoint_identity(value) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().as_posix()
    # Managed checkpoints are portable between server checkout roots.
    marker = "outputs/csgo_1b/"
    if marker in path:
        return marker + path.split(marker, 1)[1]
    return str(Path(value).expanduser().resolve())


def guard_inference_output(output_dir, task: str, batch_size: int, *,
                           checkpoint=None, split=None, seed=42, maps=None) -> None:
    """Check identity before touching predictions; mark new batched partial runs.

    A partial batched run is pinned too, so changing batch/RNG in an interrupted
    output cannot silently mix predictions. Legacy batch-1 outputs are not
    rewritten by this helper.
    """
    root = Path(output_dir)
    batch_size = positive_batch_size(batch_size)
    contract = execution_contract(task, batch_size)
    identity = {
        "inference_execution": contract,
        "checkpoint": _checkpoint_identity(checkpoint),
        "split": split,
        "seed": int(seed),
        "maps": list(maps) if maps is not None else None,
    }
    marker = root / "inference_execution_plan.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Interrupted/completed inference execution differs: {marker}")
    manifest = root / "inference_manifest.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        validate_execution_metadata(old, task, batch_size)
        if (checkpoint is not None and _checkpoint_identity(old.get("checkpoint", old.get("ckpt_path"))) != identity["checkpoint"]
            or split is not None and old.get("benchmark_v2_split", old.get("split")) != split
            or old.get("seed", old.get("inference_seed")) != int(seed)
            or maps is not None and old.get("maps") != list(maps)):
            raise ValueError(f"Existing inference identity differs: {manifest}")
    elif batch_size > 1 and root.exists() and not marker.exists():
        if ((root / "gen_imgs").is_dir() and any((root / "gen_imgs").rglob("*.jpg"))
            or any(root.glob("loc_results*.json"))):
            raise ValueError(f"Unidentified partial predictions in {root}; use a new output directory")
    if batch_size > 1 and not marker.exists():
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class InferenceExecutionTracker:
    """Track ordered sample identities and actual batch sizes, not model RNG."""

    def __init__(self, task: str, batch_size: int):
        self.contract = execution_contract(task, batch_size)
        self.hasher = hashlib.sha256()
        self.sample_count = 0
        self.batch_count = 0
        self.last_batch_size = 0
        self.max_batch_size = 0
        self.started = time.perf_counter()

    def update(self, identities) -> None:
        identities = list(identities)
        size = len(identities)
        if not 1 <= size <= self.contract["batch_size"]:
            raise ValueError("Actual inference batch is empty or exceeds requested size")
        for identity in identities:
            self.hasher.update((json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.sample_count += size
        self.batch_count += 1
        self.last_batch_size = size
        self.max_batch_size = max(self.max_batch_size, size)

    def finish(self, **extra) -> dict:
        if set(extra) & set(self.contract):
            raise ValueError("Cannot replace inference execution identity fields")
        elapsed = max(0.0, time.perf_counter() - self.started)
        return {
            **self.contract,
            "sample_order_sha256": self.hasher.hexdigest(),
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "last_batch_size": self.last_batch_size,
            "max_batch_size": self.max_batch_size,
            "elapsed_seconds": elapsed,
            "samples_per_second": self.sample_count / elapsed if elapsed else 0.0,
            **extra,
        }
