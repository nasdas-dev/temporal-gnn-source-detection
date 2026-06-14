"""Local checkpoint helpers for resumable trainable-model runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely used."""


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values to canonical JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() > 64:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
            }
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation for hashing."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def compatibility_hash(metadata: dict[str, Any]) -> str:
    """Hash metadata that determines whether a checkpoint is reusable."""
    return hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()


def array_fingerprint(values: Any) -> dict[str, Any]:
    """Return a compact fingerprint for index arrays used in compatibility metadata."""
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    return {
        "dtype": "int64",
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "head": arr[:8].tolist(),
        "tail": arr[-8:].tolist(),
    }


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch checkpoint atomically via temp file + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json_dump(payload: dict[str, Any], path: str | Path) -> None:
    """Write JSON atomically via temp file + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp, "w") as f:
            json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_json(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None when it does not exist."""
    target = Path(path)
    if not target.exists():
        return None
    with open(target) as f:
        return json.load(f)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_compatible(payload: dict[str, Any], expected_hash: str, path: str | Path) -> None:
    """Raise when a checkpoint payload does not match the expected hash."""
    found = payload.get("compatibility_hash")
    if found != expected_hash:
        raise CheckpointError(
            f"Incompatible checkpoint {path}: expected compatibility hash "
            f"{expected_hash}, found {found}. Use --fresh to start a new attempt."
        )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG states for epoch-boundary resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG state captured by capture_rng_state."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        # latest.pt is loaded with map_location=cuda, which moves the saved RNG
        # ByteTensors onto the GPU; set_rng_state_all requires CPU ByteTensors
        # (same reason the torch CPU state above is forced to .cpu()).
        cuda_states = [
            s.cpu() if torch.is_tensor(s) else s for s in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def checkpoint_timestamp() -> str:
    """Return a simple UTC timestamp for checkpoint metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
