"""VP-002: verified, lazily addressed MoE computational structures.

This is a descriptor/router/placement *control plane*, not a tensor inference
engine. It never claims to evaluate a model's router, expert or attention weights.

Realizations are admitted later through independently authorized E-CPU operators.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


class MoERefusal(ValueError):
    """An unsupported/ambiguous model, route or parameter address is refused."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MoERefusal("not canonical finite JSON") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise MoERefusal(f"{label} must be positive integer")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MoERefusal(f"{label} must be non-negative integer")
    return value


def _hex_digest(value: str, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MoERefusal(f"{label} must be a lowercase 256-bit hex digest")
    return value


# Pinned published configuration values, not inferred tensor/weight contents.
# See https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/e187836637046dedaf8a1f27c91bf65c5175cc97/config.json
# and https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/58ccb6e3fb22606f13e07de68b71f8fbce40a375/config.json
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "qwen3-30b-a3b": {
        "source_model": "Qwen/Qwen3-30B-A3B",
        "revision": "e187836637046dedaf8a1f27c91bf65c5175cc97",
        "model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
        "vocab_size": 151936, "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "norm_topk_prob": True, "tie_word_embeddings": False, "decoder_sparse_step": 1,
        "mlp_only_layers": [], "torch_dtype": "bfloat16",
        "attention_bias": False, "use_sliding_window": False, "rope_scaling": None,
    },
    "qwen3-235b-a22b": {
        "source_model": "Qwen/Qwen3-235B-A22B",
        "revision": "58ccb6e3fb22606f13e07de68b71f8fbce40a375",
        "model_type": "qwen3_moe", "hidden_size": 4096, "num_hidden_layers": 94,
        "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
        "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 1536,
        "vocab_size": 151936, "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "norm_topk_prob": True, "tie_word_embeddings": False, "decoder_sparse_step": 1,
        "mlp_only_layers": [], "torch_dtype": "bfloat16",
        "attention_bias": False, "use_sliding_window": False, "rope_scaling": None,
    },
}


@dataclass(frozen=True)
class ModelGround:
    """An immutable *architectural* identity, not a loaded or certified model."""

    profile: str
    revision: str
    source_model: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Defensive copy: no mutable caller-owned profile can alter a Ground digest.
        frozen = {k: tuple(v) if isinstance(v, list) else v for k, v in self.parameters.items()}
        object.__setattr__(self, "parameters", MappingProxyType(frozen))

    @classmethod
    def published(cls, profile: str) -> "ModelGround":
        if profile not in CONFIGURATIONS:
            raise MoERefusal(f"unknown published model profile: {profile}")
        source = CONFIGURATIONS[profile]
        g = cls(profile, source["revision"], source["source_model"],
                {key: val for key, val in source.items() if key not in ("source_model", "revision")})
        g.validate()
        return g

    def validate(self) -> None:
        c = self.parameters
        if not isinstance(c, Mapping) or c.get("model_type") != "qwen3_moe":
            raise MoERefusal("only audited qwen3_moe architecture supported")
        if type(self.revision) is not str or len(self.revision) != 40 or any(x not in "0123456789abcdef" for x in self.revision):
            raise MoERefusal("model must have pinned source revision")
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "num_experts", "num_experts_per_tok",
                    "moe_intermediate_size", "vocab_size", "max_position_embeddings",
                    "decoder_sparse_step"):
            _positive(c.get(key), key)
        if (c["num_experts_per_tok"] > c["num_experts"]
                or c["num_attention_heads"] % c["num_key_value_heads"]
                or c["head_dim"] % 2):
            raise MoERefusal("invalid grouping, top-k or rotary head dimension")
        if (c.get("norm_topk_prob") is not True or c.get("tie_word_embeddings") is not False
                or c.get("torch_dtype") != "bfloat16" or c.get("attention_bias") is not False
                or c.get("use_sliding_window") is not False or c.get("rope_scaling") is not None
                or c.get("mlp_only_layers") != () or c.get("decoder_sparse_step") != 1):
            raise MoERefusal("unsupported model variant; declare a distinct operator Ground")
        if type(c.get("rms_norm_eps")) not in (int, float) or not (0 < c["rms_norm_eps"] < 1):
            raise MoERefusal("invalid RMSNorm epsilon")
        if type(c.get("rope_theta")) not in (int, float) or not math.isfinite(c["rope_theta"]) or c["rope_theta"] <= 0:
            raise MoERefusal("invalid rotary frequency")
        if self.profile in CONFIGURATIONS:
            expected = CONFIGURATIONS[self.profile]
            if self.revision != expected["revision"] or self.source_model != expected["source_model"] or canonical(dict(c)) != canonical({k: v for k, v in expected.items() if k not in ("revision", "source_model")}):
                raise MoERefusal("published profile changed without a new pinned identity")
        else:
            raise MoERefusal("unrecognized architecture profile")

    @property
    def identifier(self) -> str:
        return digest({"kind": "ecpu-vp.moe-architecture.v1", "profile": self.profile,
                       "revision": self.revision, "source": self.source_model,
                       "config": dict(self.parameters)})

    @property
    def c(self) -> Mapping[str, Any]:
        return self.parameters

    def tensor_shape(self, layer: int, role: str, *, expert: int | None = None) -> tuple[int, ...]:
        """Lazy tensor addresses, not allocation of tensors or real weight bindings."""
        _nonnegative(layer, "layer")
        if layer >= self.c["num_hidden_layers"]:
            raise MoERefusal("layer out of range")
        h, hd, nq, nk, mid = (self.c[k] for k in ("hidden_size", "head_dim",
                                    "num_attention_heads", "num_key_value_heads", "moe_intermediate_size"))
        shapes = {
            "q_proj.weight": (nq * hd, h), "k_proj.weight": (nk * hd, h),
            "v_proj.weight": (nk * hd, h), "o_proj.weight": (h, nq * hd),
            "q_norm.weight": (hd,), "k_norm.weight": (hd,),
            "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
            "router.weight": (self.c["num_experts"], h),
        }
        if role in shapes:
            if expert is not None:
                raise MoERefusal("expert specified for shared tensor")
            return shapes[role]
        if role in ("gate_proj.weight", "up_proj.weight"):
            shape = (mid, h)
        elif role == "down_proj.weight":
            shape = (h, mid)
        else:
            raise MoERefusal("unknown tensor role")
        if expert is None or type(expert) is not int or not 0 <= expert < self.c["num_experts"]:
            raise MoERefusal("expert tensor needs an in-range expert address")
        return shape

    def tensor_ref(self, layer: int, role: str, *, expert: int | None = None) -> dict[str, Any]:
        shape = self.tensor_shape(layer, role, expert=expert)
        path = f"model.layers.{layer}." + (
            f"mlp.experts.{expert}.{role}" if expert is not None else
            "mlp.gate.weight" if role == "router.weight" else
            f"self_attn.{role}" if role in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight") else role)
        return {"model_ground": self.identifier, "tensor": path, "shape": list(shape),
                "elements": math.prod(shape), "dtype": self.c["torch_dtype"],
                "weight_binding": "unbound"}

    def kv_budget(self, tokens: int, *, byte_width: int = 2) -> dict[str, int]:
        _nonnegative(tokens, "tokens")
        if tokens > self.c["max_position_embeddings"]:
            raise MoERefusal("requested cache exceeds declared context")
        if byte_width not in (1, 2, 4):
            raise MoERefusal("unsupported cache element width")
        per_token = 2 * self.c["num_hidden_layers"] * self.c["num_key_value_heads"] * self.c["head_dim"] * byte_width
        return {"tokens": tokens, "kv_bytes_per_token": per_token,
                "kv_bytes_total": per_token * tokens, "cache_element_bytes": byte_width}

    def parameter_inventory(self) -> dict[str, int]:
        """Compute exact declared matrix element counts without instantiating tensors.

        This reflects the supported architectural layout only; it is not a
        verification of pretrained tensor availability or checkpoint completeness.
        """
        c = self.c
        h, nq, nk, hd, mid, experts, layers = (
            c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"],
            c["head_dim"], c["moe_intermediate_size"], c["num_experts"],
            c["num_hidden_layers"])
        embedding = c["vocab_size"] * h
        attention_per_layer = h * (2 * nq * hd + 2 * nk * hd) + 2 * hd
        expert_per_layer = experts * 3 * h * mid
        common_per_layer = 2 * h + experts * h
        total = 2 * embedding + h + layers * (attention_per_layer + expert_per_layer + common_per_layer)
        active = 2 * embedding + h + layers * (attention_per_layer +
            c["num_experts_per_tok"] * 3 * h * mid + common_per_layer)
        return {"declared_elements": total, "active_elements_per_token_upper_scope": active,
                "declared_bf16_bytes": total * 2,
                "active_bf16_bytes_per_token_upper_scope": active * 2,
                "declared_expert_instances": experts * layers}

    def description(self) -> dict[str, Any]:
        c = self.c
        return {"schema": "ecpu-vp.moe-architecture.v1", "profile": self.profile,
                "source_model": self.source_model, "source_revision": self.revision,
                "model_ground": self.identifier, "layers": c["num_hidden_layers"],
                "experts_per_layer": c["num_experts"],
                "expert_instances": c["num_experts"] * c["num_hidden_layers"],
                "active_experts_per_token_per_layer": c["num_experts_per_tok"],
                "active_expert_invocations_per_token": c["num_hidden_layers"] * c["num_experts_per_tok"],
                "query_heads": c["num_attention_heads"], "kv_heads": c["num_key_value_heads"],
                "head_dim": c["head_dim"], "hidden_size": c["hidden_size"],
                "context_limit": c["max_position_embeddings"],
                "kv_budget_at_context_limit": self.kv_budget(c["max_position_embeddings"]),
                "parameter_inventory": self.parameter_inventory(),
                "weights_loaded": False, "neural_inference_performed": False}


@dataclass(frozen=True)
class RoutingReceipt:
    """Reconstructible routing decision computed from *supplied*, not model-evaluated, logits."""
    value: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return dict(self.value)

    @property
    def receipt_digest(self) -> str:
        return digest(self.value)


def route_experts(ground: ModelGround, *, layer: int, token: int,
                  logits: Sequence[float], weight_manifest_digest: str | None = None) -> RoutingReceipt:
    ground.validate()
    _nonnegative(token, "token")
    if token >= ground.c["max_position_embeddings"]:
        raise MoERefusal("token position exceeds declared model context")
    _nonnegative(layer, "layer")
    if layer >= ground.c["num_hidden_layers"]:
        raise MoERefusal("layer out of range")
    if type(logits) not in (list, tuple) or len(logits) != ground.c["num_experts"]:
        raise MoERefusal("router logits must supply one score per expert")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
        raise MoERefusal("router logits must be finite numbers")
    if weight_manifest_digest is not None:
        _hex_digest(weight_manifest_digest, "weight manifest digest")
    try:
        vals = [float(x) for x in logits]
    except (ValueError, OverflowError) as exc:
        raise MoERefusal("router values cannot be represented as finite reference floats") from exc
    if any(not math.isfinite(x) for x in vals):
        raise MoERefusal("router values overflow reference precision")
    ceiling = max(vals)
    exp_values = [math.exp(x - ceiling) for x in vals]
    denom = math.fsum(exp_values)
    probabilities = [v / denom for v in exp_values]
    chosen = sorted(range(len(vals)), key=lambda idx: (-probabilities[idx], idx))[:ground.c["num_experts_per_tok"]]
    sum_chosen = math.fsum(probabilities[idx] for idx in chosen)
    normalized = [probabilities[idx] / sum_chosen for idx in chosen]
    # Numeric re-normalization uses the model's `norm_topk_prob` intent; weights
    # here are Python float64 reference values, not asserted bfloat16 forward values.
    routes = [{"expert": idx, "weight": weight,
               "tensor_addresses": [ground.tensor_ref(layer, role, expert=idx)["tensor"]
                                    for role in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")]}
              for idx, weight in zip(chosen, normalized)]
    record = {"schema": "ecpu-vp.moe-route.v1", "model_ground": ground.identifier,
              "source_model": ground.source_model, "layer": layer, "token_position": token,
              "router_logits": vals, "router_logits_digest": digest(vals),
              "top_k": ground.c["num_experts_per_tok"], "routes": routes,
              "weight_manifest_digest": weight_manifest_digest,
              "tensor_payloads_loaded": False, "neural_inference_performed": False,
              "scope": "reference route of externally supplied logits; not a model forward pass"}
    return RoutingReceipt(record)


def verify_route(ground: ModelGround, report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping) or set(report) != {"receipt", "digest"}:
        raise MoERefusal("receipt wrapper must have exact fields")
    event = report["receipt"]
    if not isinstance(event, Mapping):
        raise MoERefusal("invalid receipt")
    if digest(event) != report["digest"]:
        raise MoERefusal("receipt digest mismatch")
    try:
        expected = route_experts(ground, layer=event["layer"], token=event["token_position"],
                                 logits=event["router_logits"],
                                 weight_manifest_digest=event["weight_manifest_digest"])
    except (KeyError, TypeError) as exc:
        raise MoERefusal("missing typed routing fields") from exc
    if canonical(event) != canonical(expected.snapshot()):
        raise MoERefusal("route does not reconstruct under pinned Ground")


def compile_layer_frontier(ground: ModelGround, report: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one dependency-bounded, lazily addressed MoE execution frontier.

    The upstream router score is *supplied evidence*, not an inferred output
    of this graph. Each task is a declared operator invocation, never a claim
    that the numerical operator ran. This avoids the 256-node global DAG bound.
    """
    verify_route(ground, report)
    event = report["receipt"]
    layer, token = event["layer"], event["token_position"]
    prefix = f"l{layer}:t{token}:"
    tasks: list[dict[str, Any]] = []

    def node(name: str, operator: str, parents: Sequence[str],
             refs: Sequence[dict[str, Any]] = ()) -> str:
        identity = prefix + name
        if any(i["id"] == identity for i in tasks):
            raise MoERefusal("duplicate frontier operation")
        predecessors = [prefix + x for x in parents]
        if any(p not in {i["id"] for i in tasks} for p in predecessors):
            raise MoERefusal("frontier not topologically ordered")
        tasks.append({"id": identity, "operator": operator,
                      "dependencies": predecessors,
                      "tensor_refs": list(refs), "execution": "not-performed"})
        return identity

    def ref(role: str) -> dict[str, Any]:
        return ground.tensor_ref(layer, role)

    node("hidden", "state.read_hidden", [])
    node("attn_norm", "nn.rms_norm", ["hidden"], [ref("input_layernorm.weight")])
    for name in ("q", "k", "v"):
        node(name, "nn.linear", ["attn_norm"], [ref(f"{name}_proj.weight")])
    node("q_norm", "nn.head_rms_norm", ["q"], [ref("q_norm.weight")])
    node("k_norm", "nn.head_rms_norm", ["k"], [ref("k_norm.weight")])
    node("q_rotary", "nn.rope", ["q_norm"])
    node("k_rotary", "nn.rope", ["k_norm"])
    node("kv_append", "state.kv_append", ["k_rotary", "v"])
    node("attention", "nn.causal_gqa", ["q_rotary", "kv_append"])
    node("attn_out", "nn.linear", ["attention"], [ref("o_proj.weight")])
    node("attn_residual", "nn.add", ["hidden", "attn_out"])
    node("mlp_norm", "nn.rms_norm", ["attn_residual"], [ref("post_attention_layernorm.weight")])
    node("router", "nn.topk_router", ["mlp_norm"], [ref("router.weight")])
    names: list[str] = []
    for assignment in event["routes"]:
        expert = assignment["expert"]
        name = f"expert_{expert}"
        node(name, "nn.expert_swiglu", ["mlp_norm", "router"],
             [ground.tensor_ref(layer, role, expert=expert) for role in
              ("gate_proj.weight", "up_proj.weight", "down_proj.weight")])
        names.append(name)
    node("sparse_reduce", "nn.router_weighted_sum", ["router", *names])
    node("layer_output", "nn.add", ["attn_residual", "sparse_reduce"])
    frontier = {"schema": "ecpu-vp.moe-frontier.v1", "model_ground": ground.identifier,
                "layer": layer, "token_position": token, "router_receipt_digest": report["digest"],
                "tasks": tasks, "tensor_payloads_loaded": False,
                "neural_inference_performed": False,
                "scope": "typed dependency frontier; no weights, attention or expert math executed"}
    return {"frontier": frontier, "digest": digest(frontier)}


def verify_layer_frontier(ground: ModelGround, route: Mapping[str, Any],
                          frontier: Mapping[str, Any]) -> None:
    if not isinstance(frontier, Mapping) or set(frontier) != {"frontier", "digest"}:
        raise MoERefusal("frontier envelope must have exact fields")
    if digest(frontier["frontier"]) != frontier["digest"]:
        raise MoERefusal("frontier digest mismatch")
    if canonical(frontier) != canonical(compile_layer_frontier(ground, route)):
        raise MoERefusal("frontier dependency or tensor binding differs from Ground")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-CPU VP-002 large-MoE structure and routing reference")
    parser.add_argument("command", choices=("describe", "route", "verify", "frontier"))
    parser.add_argument("--model", choices=tuple(CONFIGURATIONS), default="qwen3-30b-a3b")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--logits", type=Path, help="JSON array of one externally supplied router logit per expert")
    parser.add_argument("--receipt", type=Path, help="JSON route report to verify")
    parser.add_argument("--weight-manifest-digest", help="Digest of a separately verified weight manifest")
    args = parser.parse_args(argv)
    g = ModelGround.published(args.model)
    if args.command == "describe":
        print(json.dumps(g.description(), sort_keys=True, indent=2))
    elif args.command == "route":
        if args.logits is None:
            parser.error("route requires --logits containing genuine or synthetic, explicitly labeled scores")
        try:
            vals = json.loads(args.logits.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MoERefusal("cannot parse logits JSON") from exc
        event = route_experts(g, layer=args.layer, token=args.token, logits=vals,
                              weight_manifest_digest=args.weight_manifest_digest)
        print(json.dumps({"receipt": event.snapshot(), "digest": event.receipt_digest}, sort_keys=True, indent=2))
    else:
        if args.receipt is None:
            parser.error(f"{args.command} requires --receipt")
        try:
            report = json.loads(args.receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MoERefusal("cannot parse receipt") from exc
        verify_route(g, report)
        if args.command == "frontier":
            print(json.dumps(compile_layer_frontier(g, report), sort_keys=True, indent=2))
        else:
            print(json.dumps({"verified": True, "receipt_digest": report["digest"], "model_ground": g.identifier}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
