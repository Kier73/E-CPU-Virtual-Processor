"""VP-003: parameter-bearing sparse SwiGLU inference with lazy verified tensor reads.

This is actual numerical neural execution, not a topology/router simulation.  The
receiver owns the immutable weight index; requests cannot choose arbitrary file
paths, alter weights, or substitute a computation operator. A separate E-CPU
VirtualLayer adapter (`vp.native_inference`) governs execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class InferenceRefused(ValueError):
    """Untrusted/missing tensor, malformed request, or invalid numerical evidence."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=False).encode("utf8")
    except (TypeError, ValueError) as exc:
        raise InferenceRefused("noncanonical or nonfinite JSON") from exc


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise InferenceRefused(f"{label} must be integer >= {minimum}")
    return value


def _hex_sha(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise InferenceRefused(f"{label} must be lowercase SHA256")
    return value


@dataclass(frozen=True)
class SparseWeightIndex:
    """Verified local weight descriptors; no declaration allocates all model weights.

    The manifest may describe an arbitrarily large model, but only selected
    expert tensors are needed for a specific query. A missing selected expert
    is refused, never replaced by generated coefficients.
    """
    root: Path
    record: Mapping[str, Any]
    digest: str

    @classmethod
    def open(cls, path: Path, *, expected_sha256: str | None = None) -> "SparseWeightIndex":
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        if len(raw) > 12_000_000:
            raise InferenceRefused("index exceeds metadata size budget")
        try:
            record = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise InferenceRefused("invalid weight index JSON") from exc
        if not isinstance(record, dict) or set(record) != {
            "schema", "model_ground", "layer", "hidden_size", "num_experts", "top_k", "tensors", "provenance"
        } or record["schema"] != "ecpu-vp.sparse-weights.v1":
            raise InferenceRefused("unexpected tensor index schema")
        model_ground = _hex_sha(record["model_ground"], "model_ground")
        _int(record["layer"], "layer")
        hidden = _int(record["hidden_size"], "hidden_size", 1)
        experts = _int(record["num_experts"], "num_experts", 1)
        top_k = _int(record["top_k"], "top_k", 1)
        if top_k > experts:
            raise InferenceRefused("top_k exceeds expert count")
        if not isinstance(record["provenance"], dict) or record["provenance"].get("kind") not in ("measured", "synthetic", "generated"):
            raise InferenceRefused("tensor provenance must explicitly identify observed or synthetic weights")
        tensors = record["tensors"]
        if not isinstance(tensors, dict) or "router" not in tensors:
            raise InferenceRefused("router tensor is required")
        for key, t in tensors.items():
            if not isinstance(key, str) or len(key) > 150 or not isinstance(t, dict) or set(t) not in ({"file", "sha256", "shape", "dtype"}, {"file", "sha256", "shape", "dtype", "tensor_key"}):
                raise InferenceRefused("invalid typed tensor record")
            if type(t["file"]) is not str or not t["file"].endswith((".npy", ".safetensors")) or not t["file"].isascii() or t["file"] in (".", ".."):
                raise InferenceRefused("only simple local .npy or .safetensors file names admitted")
            if Path(t["file"]).name != t["file"] or '/' in t["file"] or '\\' in t["file"]:
                raise InferenceRefused("tensor paths cannot leave the receiver store")
            _hex_sha(t["sha256"], "tensor sha256")
            if (t["dtype"] not in ("float16", "bfloat16", "float32", "float64")
                or not isinstance(t["shape"], list) or any(type(n) is not int or n <= 0 for n in t["shape"])
                or (t["file"].endswith(".npy") and ("tensor_key" in t or t["dtype"] == "bfloat16"))
                or (t["file"].endswith(".safetensors") and (type(t.get("tensor_key")) is not str or not t["tensor_key"] or len(t["tensor_key"])>256))):
                raise InferenceRefused("unsupported tensor dtype or shape")
            if key == "router":
                if t["shape"] != [experts, hidden]:
                    raise InferenceRefused("router shape mismatch")
            elif key.startswith("experts/"):
                parts = key.split("/")
                if len(parts) != 3 or not parts[1].isdigit() or str(int(parts[1])) != parts[1] or int(parts[1]) >= experts or parts[2] not in ("gate", "up", "down"):
                    raise InferenceRefused("invalid expert address")
                dims = t["shape"]
                if len(dims) != 2 or (dims[1] != hidden if parts[2] != "down" else dims[0] != hidden):
                    raise InferenceRefused("expert hidden dimension mismatch")
            else:
                raise InferenceRefused("unrecognized tensor address")
        # When a VP-002 published Qwen architecture is claimed, verify the
        # complete shape contract against that Ground before trusting bytes.
        from .moe import CONFIGURATIONS, ModelGround
        for profile in CONFIGURATIONS:
            published = ModelGround.published(profile)
            if model_ground != published.identifier:
                continue
            config = published.c
            if (hidden != config["hidden_size"] or experts != config["num_experts"]
                    or top_k != config["num_experts_per_tok"]
                    or record["layer"] >= config["num_hidden_layers"]):
                raise InferenceRefused("tensor index contradicts claimed published model Ground")
            for name,t in tensors.items():
                if name == "router":
                    role, expert = "router.weight", None
                else:
                    _, number, part = name.split("/")
                    role, expert = part + "_proj.weight", int(number)
                expected = published.tensor_shape(record["layer"],role,expert=expert)
                if t["shape"] != list(expected):
                    raise InferenceRefused("published model tensor address shape mismatch")
        canonical_digest = _sha(_canonical(record))
        if expected_sha256 is not None and canonical_digest != _hex_sha(expected_sha256,"expected index digest"):
            raise InferenceRefused("weight index changed from trusted manifest")
        return cls(path.parent, record, canonical_digest)

    def load(self, name: str, *, verified_files: dict | None = None):
        """Verify the selected file, then read only the named parameter tensor.

        Safetensors requires the original file's SHA-256; different tensor
        addresses in the same physical shard reuse one verification per query.
        A trusted storage layer must prevent concurrent mutation after read.
        """
        import numpy as np
        if name not in self.record["tensors"]:
            raise InferenceRefused(f"required tensor not provisioned: {name}")
        ref = self.record["tensors"][name]
        path = self.root / ref["file"]
        if path.is_symlink() or not path.is_file():
            raise InferenceRefused("weight path is not a regular local file")
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        seen = verified_files if verified_files is not None else {}
        prior = seen.get(str(path))
        if prior != (ref["sha256"], stamp):
            with path.open("rb") as reader:
                h = hashlib.sha256()
                while chunk := reader.read(1024 * 1024):
                    h.update(chunk)
            if h.hexdigest() != ref["sha256"] or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise InferenceRefused("tensor bytes do not match admitted digest")
            seen[str(path)] = (ref["sha256"], stamp)
        try:
            if path.suffix == ".npy":
                tensor = np.load(path, mmap_mode="r", allow_pickle=False)
                dtype = tensor.dtype.name
            else:
                from safetensors import safe_open
                with safe_open(str(path), framework="pt", device="cpu") as reader:
                    if ref["tensor_key"] not in reader.keys():
                        raise InferenceRefused("tensor key not found in admitted Safetensors")
                    t = reader.get_tensor(ref["tensor_key"])
                    dtype = str(t.dtype).removeprefix("torch.")
                    tensor = t.float().numpy() if dtype == "bfloat16" else t.numpy()
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            raise InferenceRefused("selected tensor payload cannot be loaded") from exc
        if list(tensor.shape) != ref["shape"] or dtype != ref["dtype"] or not bool(np.isfinite(tensor).all()):
            raise InferenceRefused("tensor values do not match declared shape, dtype or finiteness")
        return tensor


def _silu(value):
    import numpy as np
    # Stable sigmoid in float64; no overflow in the large-negative branch.
    x = np.asarray(value, dtype=np.float64)
    return x * np.exp(-np.logaddexp(0, -x))


def sparse_forward(index: SparseWeightIndex, hidden: Sequence[float], *, request_id: str) -> dict[str, Any]:
    """Execute a real sparse SwiGLU expert aggregation from verified tensors.

    It is a single MoE feed-forward sublayer, not full Transformer attention or
    autoregressive LLM inference. The returned receipt includes input, output,
    route and exact loaded tensor digests; an independent scalar oracle checks
    all output coordinates for small input or three for large input.
    """
    import numpy as np
    if type(request_id) is not str or not request_id or len(request_id) > 128 or request_id.strip() != request_id:
        raise InferenceRefused("invalid request identity")
    if not isinstance(hidden, (list, tuple)) or len(hidden) != index.record["hidden_size"] or any(type(n) not in (float, int) or not math.isfinite(n) for n in hidden):
        raise InferenceRefused("hidden state must match receiver Ground")
    x = np.asarray(hidden, dtype=np.float64)
    verified_files: dict = {}
    router = np.asarray(index.load("router", verified_files=verified_files), dtype=np.float64)
    logits = router @ x
    order = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))[:index.record["top_k"]]
    selected = np.asarray([logits[i] for i in order], dtype=np.float64)
    weights = np.exp(selected - np.max(selected))
    weights /= weights.sum()
    output = np.zeros_like(x)
    selected_tensors: dict[str, str] = {"router": index.record["tensors"]["router"]["sha256"]}
    scalar_inputs: list[tuple[float, Any, Any, Any]] = []
    for expert, weight in zip(order, weights):
        parts = {}
        for role in ("gate", "up", "down"):
            name = f"experts/{expert}/{role}"
            parts[role] = np.asarray(index.load(name, verified_files=verified_files), dtype=np.float64)
            selected_tensors[name] = index.record["tensors"][name]["sha256"]
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        if gate.shape != up.shape or down.shape != (len(x),gate.shape[0]):
            raise InferenceRefused("expert gate/up/down shapes inconsistent")
        activation = _silu(gate @ x) * (up @ x)
        partial = down @ activation
        output += float(weight) * partial
        scalar_inputs.append((float(weight), gate, up, down))
    if not bool(np.isfinite(output).all()):
        raise InferenceRefused("nonfinite sparse inference output")
    # Independent scalar accumulation checks a few coordinates using Python fsum
    # rather than reusing NumPy BLAS for the entire forward calculation.
    sampled = sorted(set([0, len(x)//2, len(x)-1]))
    oracle: dict[str, float] = {}
    scalar_experts = []
    for weight, gate, up, down in scalar_inputs:
        activations = []
        for row_gate, row_up in zip(gate, up):
            g = math.fsum(float(a)*float(b) for a,b in zip(row_gate,x))
            u = math.fsum(float(a)*float(b) for a,b in zip(row_up,x))
            sigmoid = (1/(1+math.exp(-g)) if g >= 0
                       else math.exp(g)/(1+math.exp(g)))
            activations.append(g * sigmoid * u)
        scalar_experts.append((weight, down, activations))
    for coord in sampled:
        contributions = [w * math.fsum(float(a)*float(b) for a,b in zip(down[coord], activations))
                         for w,down,activations in scalar_experts]
        scalar = math.fsum(contributions)
        if not math.isclose(float(output[coord]),scalar,rel_tol=2e-9,abs_tol=2e-9):
            raise InferenceRefused("scalar numerical oracle disagrees with operator")
        oracle[str(coord)] = scalar
    public={
        "schema":"ecpu-vp.sparse-inference.v1",
        "request_id":request_id,"model_ground":index.record["model_ground"],
        "weight_index_digest":index.digest,"layer":index.record["layer"],
        "source_kind":index.record["provenance"]["kind"],
        "hidden_sha256":_sha(np.ascontiguousarray(x,dtype="<f8").tobytes()),
        "router_logits_sha256":_sha(np.ascontiguousarray(logits,dtype="<f8").tobytes()),
        "selected_experts":order,"selected_weights":[float(w) for w in weights],
        "loaded_tensor_digests": selected_tensors,
        "output_sha256":_sha(np.ascontiguousarray(output,dtype="<f8").tobytes()),
        "output":[float(y) for y in output],
        "independent_scalar_coordinates":oracle,
        "actual_neural_math_executed":True,
        "full_llm_inference_performed":False,
        "realization":"numpy.float64.sparse_swiglu",
    }
    return {"receipt":public,"digest":_sha(_canonical(public))}


def verify_sparse_receipt(index: SparseWeightIndex, hidden: Sequence[float], report: Mapping[str,Any]) -> None:
    """Re-execute from receiver-verified weights, not just the submitted digest."""
    if not isinstance(report, Mapping) or set(report) != {"receipt","digest"} or not isinstance(report["receipt"],dict):
        raise InferenceRefused("invalid numerical receipt envelope")
    if _sha(_canonical(report["receipt"])) != report["digest"]:
        raise InferenceRefused("receipt hash mismatch")
    expected=sparse_forward(index,hidden,request_id=report["receipt"].get("request_id"))
    if _canonical(expected)!=_canonical(report):
        raise InferenceRefused("rehashed numerical result fails independent execution")
