"""VP-005: algebraic expert navigation over a huge, precisely declared model Ground.

This is a SYNTHETIC procedural family, not a pretrained MoE checkpoint. It
reuses MailCPU's lcg_navigate and VirtualLayerRuntime when hosted natively;
this module supplies only the domain law, selected expert arithmetic, and
independent receipt verification, not a new scheduler or model authority.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class NavigableExpertRefused(ValueError):
    """A request, model Ground, route, or observation has insufficient evidence."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NavigableExpertRefused("the value is not finite canonical JSON") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise NavigableExpertRefused(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def jump_affine_independent(a: int, b: int, modulus: int, origin: int, coordinate: int) -> int:
    """Independent binary affine-map composition, O(log coordinate), not token iteration.

    Main native path must additionally agree with MailCPU lcg_navigate.
    """
    mul, add = 1, 0
    base_mul, base_add = a % modulus, b % modulus
    t = coordinate
    while t:
        if t & 1:
            mul, add = (base_mul * mul) % modulus, (base_mul * add + base_add) % modulus
        base_mul, base_add = (base_mul * base_mul) % modulus, (base_mul * base_add + base_add) % modulus
        t >>= 1
    return (mul * origin + add) % modulus


@dataclass(frozen=True)
class ExpertGround:
    """Compact Ground for N procedural experts, not a trained weight snapshot."""

    expert_count: int
    top_k: int
    hidden_dim: int
    intermediate_dim: int
    multiplier: int = 5
    increment: int = 1
    initial_state: int = 7

    def __post_init__(self) -> None:
        n = strict_integer(self.expert_count, "expert_count", 8, 1 << 60)
        if n & (n - 1):
            raise NavigableExpertRefused("expert_count must be a power of two for the declared full-period orbit")
        strict_integer(self.top_k, "top_k", 1, min(16, n))
        strict_integer(self.hidden_dim, "hidden_dim", 1, 16)
        strict_integer(self.intermediate_dim, "intermediate_dim", 1, 16)
        a = strict_integer(self.multiplier, "multiplier", 1, n - 1)
        b = strict_integer(self.increment, "increment", 1, n - 1)
        strict_integer(self.initial_state, "initial_state", 0, n - 1)
        if a % 4 != 1 or b % 2 != 1:
            raise NavigableExpertRefused("affine coordinate law must have full period for the power-of-two Ground")

    def to_dict(self) -> dict[str, int]:
        return {"expert_count": self.expert_count, "top_k": self.top_k,
                "hidden_dim": self.hidden_dim, "intermediate_dim": self.intermediate_dim,
                "multiplier": self.multiplier, "increment": self.increment,
                "initial_state": self.initial_state}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertGround":
        if not isinstance(value, Mapping) or set(value) != set(cls(8, 1, 1, 1).to_dict()):
            raise NavigableExpertRefused("expert Ground manifest fields do not match")
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        return digest({"schema": "vp005.generative-expert-ground.v1",
                       "procedural_weights": "integer-gated-relu-v1", **self.to_dict()})

    @property
    def logical_parameters(self) -> int:
        # Gate/up: 2*(w*d). Down: d*w, all scalar parameters.
        return 3 * self.expert_count * self.hidden_dim * self.intermediate_dim

    def center(self, coordinate: int, *, require_ecpu: bool = False) -> int:
        strict_integer(coordinate, "token_coordinate", 0, (1 << 63) - 1)
        independent = jump_affine_independent(self.multiplier, self.increment,
                                             self.expert_count, self.initial_state, coordinate)
        if require_ecpu:
            from mailcpu.law_certifier import lcg_navigate
            native = lcg_navigate(self.multiplier, self.increment,
                                  self.expert_count, self.initial_state, coordinate)
            if independent != native:
                raise NavigableExpertRefused("E-CPU coordinate law disagrees with independent affine composition")
        return independent

    def route(self, coordinate: int, *, require_ecpu: bool = False) -> dict[str, Any]:
        """Select the provable top-k of score(j)=-(j-center)^2 in O(k)."""
        c = self.center(coordinate, require_ecpu=require_ecpu)
        left, right = c, c + 1
        selected: list[int] = []
        while len(selected) < self.top_k:
            if left < 0:
                chosen, right = right, right + 1
            elif right >= self.expert_count or c - left <= right - c:
                chosen, left = left, left - 1
            else:
                chosen, right = right, right + 1
            selected.append(chosen)
        dists = [abs(i-c) for i in selected]
        # Exact rational scores from inverse distance, not Qwen softmax routing.
        raw = [Fraction(1, 1 + d) for d in dists]
        total = sum(raw, Fraction(0))
        weights = [w / total for w in raw]
        report = {"schema": "vp005.closed-form-route.v1", "ground": self.fingerprint,
                  "coordinate": coordinate, "center": c, "selected_experts": selected,
                  "scores": [-(d*d) for d in dists],
                  "weights": [rational_text(x) for x in weights],
                  "selection_method": "provable-nearest-center-no-scan"}
        return {**report, "digest": digest(report)}


def rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_rational(value: str) -> Fraction:
    if type(value) is not str or len(value) > 100 or '/' not in value:
        raise NavigableExpertRefused("expected a canonical rational text")
    try:
        result = Fraction(value)
    except (TypeError, ZeroDivisionError, ValueError) as exc:
        raise NavigableExpertRefused("invalid rational") from exc
    if rational_text(result) != value:
        raise NavigableExpertRefused("rational is not canonical")
    return result


def verify_route(ground: ExpertGround, route: Mapping[str, Any], *, require_ecpu: bool = False) -> None:
    if not isinstance(route, Mapping) or type(route.get("coordinate")) is not int:
        raise NavigableExpertRefused("malformed route")
    expected = ground.route(route["coordinate"], require_ecpu=require_ecpu)
    if canonical(dict(route)) != canonical(expected):
        raise NavigableExpertRefused("route disagrees with receiver-owned mathematical Ground")
    # Independent exhaustive top-k oracle for bounded model sizes only.
    if ground.expert_count <= 256:
        center = ground.center(route["coordinate"])
        oracle = sorted(range(ground.expert_count), key=lambda j: (abs(j-center), j))[:ground.top_k]
        if list(route["selected_experts"]) != oracle:
            raise NavigableExpertRefused("finite exhaustive router oracle disagrees")


def _coefficient(expert: int, role: str, row: int, col: int) -> int:
    """Strictly declared law for synthetic signed integer neural weight coordinates."""
    tags = {"gate": 11, "up": 17, "down": 23}
    if role not in tags:
        raise NavigableExpertRefused("unrecognized parameter role")
    return ((expert % 251 * 13 + tags[role] * 7 + row * 11 + col * 19) % 17) - 8


def _hidden(ground: ExpertGround, values: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != ground.hidden_dim:
        raise NavigableExpertRefused("hidden vector shape mismatch")
    return tuple(strict_integer(x, "hidden scalar", -1000, 1000) for x in values)


def _expert_forward(ground: ExpertGround, expert: int, hidden: tuple[int, ...]) -> tuple[int, ...]:
    activations = []
    for row in range(ground.intermediate_dim):
        gate = sum(_coefficient(expert, "gate", row, col) * x for col, x in enumerate(hidden))
        up = sum(_coefficient(expert, "up", row, col) * x for col, x in enumerate(hidden))
        activations.append(max(0, gate) * up)
    return tuple(sum(_coefficient(expert, "down", row, col) * val
                     for col, val in enumerate(activations)) for row in range(ground.hidden_dim))


def _independent_forward(ground: ExpertGround, expert: int, hidden: tuple[int, ...]) -> tuple[int, ...]:
    """Independent scalar test oracle with explicit nested products, no matrix routines."""
    result = [0] * ground.hidden_dim
    for j in range(ground.intermediate_dim):
        g, u = 0, 0
        for k in range(ground.hidden_dim):
            e = expert % 251
            g += (((e*13 + 77 + j*11 + k*19) % 17) - 8) * hidden[k]
            u += (((e*13 + 119 + j*11 + k*19) % 17) - 8) * hidden[k]
        act = (g if g > 0 else 0) * u
        for out in range(ground.hidden_dim):
            weight = ((e*13 + 161 + out*11 + j*19) % 17) - 8
            result[out] += weight * act
    return tuple(result)


def observe(ground: ExpertGround, route: Mapping[str, Any], hidden: Sequence[int], *,
            require_ecpu: bool = False) -> dict[str, Any]:
    verify_route(ground, route, require_ecpu=require_ecpu)
    x = _hidden(ground, hidden)
    contributions = [_expert_forward(ground, i, x) for i in route["selected_experts"]]
    weights = [parse_rational(t) for t in route["weights"]]
    total = [sum((w * vector[j] for w, vector in zip(weights, contributions)), Fraction(0))
             for j in range(ground.hidden_dim)]
    report = {"schema": "vp005.exact-expert-observation.v1", "ground": ground.fingerprint,
              "route_digest": route["digest"], "hidden": list(x),
              "result": [rational_text(y) for y in total],
              "materialized_experts": len(contributions),
              "logical_parameters": ground.logical_parameters,
              "actual_expert_math": True, "source_class": "synthetic-law-not-trained-weights"}
    return {**report, "digest": digest(report)}


def verify_observation(ground: ExpertGround, route: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    verify_route(ground, route)
    if not isinstance(observation, Mapping) or type(observation.get("hidden")) is not list:
        raise NavigableExpertRefused("malformed observation")
    x = _hidden(ground, observation["hidden"])
    weights = [parse_rational(t) for t in route["weights"]]
    actual = [_independent_forward(ground, j, x) for j in route["selected_experts"]]
    out = [rational_text(sum((w * val[i] for w, val in zip(weights, actual)), Fraction(0)))
           for i in range(ground.hidden_dim)]
    expected = {"schema": "vp005.exact-expert-observation.v1", "ground": ground.fingerprint,
                "route_digest": route["digest"], "hidden": list(x), "result": out,
                "materialized_experts": ground.top_k, "logical_parameters": ground.logical_parameters,
                "actual_expert_math": True, "source_class": "synthetic-law-not-trained-weights"}
    if canonical(dict(observation)) != canonical({**expected, "digest": digest(expected)}):
        raise NavigableExpertRefused("observation disagrees with independent neural arithmetic")


def build_plan(ground: ExpertGround, *, coordinate: int, hidden: Sequence[int],
               request_id: str, require_ecpu: bool = False) -> dict[str, Any]:
    if type(request_id) is not str or not 1 <= len(request_id) <= 96 or request_id.strip() != request_id:
        raise NavigableExpertRefused("invalid request id")
    x = _hidden(ground, hidden)
    route = ground.route(coordinate, require_ecpu=require_ecpu)
    document = {"schema": "vp005.neural-intention.v1", "ground": ground.to_dict(),
                "model_ground": ground.fingerprint, "request_id": request_id,
                "hidden": list(x), "route": route}
    return {**document, "digest": digest(document)}


def verify_plan(ground: ExpertGround, plan: Mapping[str, Any], *, require_ecpu: bool = False) -> None:
    if not isinstance(plan, Mapping):
        raise NavigableExpertRefused("plan must be a canonical object")
    request_id = plan.get("request_id")
    if type(request_id) is not str:
        raise NavigableExpertRefused("invalid request id")
    # Receiver's own Ground object MUST be the authority, never plan's purported identity.
    expected = build_plan(ground, coordinate=plan.get("route", {}).get("coordinate"),
                          hidden=plan.get("hidden"), request_id=request_id, require_ecpu=require_ecpu)
    if canonical(dict(plan)) != canonical(expected):
        raise NavigableExpertRefused("plan violates receiver-owned Ground/source policy")


class NativeExpertSession:
    """Receiver-local binding around the EXISTING E-CPU Hypervisor operator.

    One session retains the native semantic-result ledger across exact request
    retransmissions; this is deliberately not a second persistence authority.
    """

    def __init__(self, ground: ExpertGround, *, allow_compute: bool = True,
                 node_id: str = "neural-receiver") -> None:
        from .native_inference import NativeInferenceHost
        self.ground = ground
        self.host = NativeInferenceHost(
            self._handle, node_id=node_id, model_ground=ground.fingerprint,
            operator_id="vp.procedural.expert-observe", allow_compute=allow_compute,
        )

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        verify_plan(self.ground, request["plan"], require_ecpu=True)
        observation = observe(self.ground, request["plan"]["route"],
                              request["plan"]["hidden"], require_ecpu=True)
        verify_observation(self.ground, request["plan"]["route"], observation)
        return observation

    def query(self, plan: Mapping[str, Any], *, sequence: int = 1) -> dict[str, Any]:
        verify_plan(self.ground, plan, require_ecpu=True)
        command = {"request_id": plan["request_id"],
                   "model_ground": self.ground.fingerprint, "plan": dict(plan)}
        return self.host.execute(command, sequence=sequence)


def run_native(ground: ExpertGround, plan: Mapping[str, Any], *,
               allow_compute: bool = True, node_id: str = "neural-receiver") -> dict[str, Any]:
    """Execute once through native E-CPU; keep NativeExpertSession for replays."""
    return NativeExpertSession(ground, allow_compute=allow_compute,
                               node_id=node_id).query(plan)


def verify_native_report(ground: ExpertGround, report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping) or set(report) != {"schema", "plan", "native", "digest"}:
        raise NavigableExpertRefused("native report schema invalid")
    plan, native = report["plan"], report["native"]
    verify_plan(ground, plan)
    if not isinstance(native, Mapping) or (native.get("accepted") is not True
                                         or native.get("hypervisor_action") != "allow"
                                         or native.get("model_ground") != ground.fingerprint
                                         or type(native.get("operator_contract_digest")) is not str
                                         or len(native["operator_contract_digest"]) != 64):
        raise NavigableExpertRefused("receiver did not report a valid authorized E-CPU execution")
    verify_observation(ground, plan["route"], native.get("result"))
    expected = {"schema": "vp005.native-observer-report.v1", "plan": plan, "native": native}
    if canonical(dict(report)) != canonical({**expected, "digest": digest(expected)}):
        raise NavigableExpertRefused("native report digest mismatch")


def _load(path: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) > 262144:
        raise NavigableExpertRefused("observation file exceeds 256 KiB")
    document = json.loads(raw)
    if type(document) is not dict:
        raise NavigableExpertRefused("expected a JSON object")
    return document


def _save(path: str, document: Mapping[str, Any]) -> None:
    Path(path).write_bytes(canonical(document) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VP-005 independent observer-host neural Ground")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--expert-count", type=int, default=1 << 48)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=3)
    p.add_argument("--intermediate-dim", type=int, default=4)
    p.add_argument("--token-coordinate", type=int, default=(1 << 56) + 73)
    p.add_argument("--hidden", type=int, nargs="+", default=[3, -2, 5])
    p.add_argument("--request-id", default="vp005-first-query")
    e = sub.add_parser("execute")
    e.add_argument("--plan", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--expected-ground", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--report", required=True)
    v.add_argument("--expected-ground", required=True)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        g = ExpertGround(args.expert_count, args.top_k, args.hidden_dim, args.intermediate_dim)
        document = build_plan(g, coordinate=args.token_coordinate, hidden=args.hidden,
                              request_id=args.request_id)
        _save(args.out, document)
        print(canonical({"ground": g.fingerprint, "logical_parameters": g.logical_parameters,
                         "selected": document["route"]["selected_experts"],
                         "plan_sha256": document["digest"]}).decode())
    elif args.mode == "execute":
        plan = _load(args.plan)
        ground = ExpertGround.from_dict(plan.get("ground"))
        if ground.fingerprint != args.expected_ground:
            raise NavigableExpertRefused("receiver-owned expected model Ground mismatch")
        native = run_native(ground, plan)
        if native.get("accepted") is not True:
            raise NavigableExpertRefused("receiver Hypervisor denied model observation")
        body = {"schema": "vp005.native-observer-report.v1", "plan": plan, "native": native}
        report = {**body, "digest": digest(body)}
        verify_native_report(ground, report)
        _save(args.out, report)
        print(canonical({"native_execution": True, "report_digest": report["digest"]}).decode())
    else:
        report = _load(args.report)
        ground = ExpertGround.from_dict(report.get("plan", {}).get("ground"))
        if ground.fingerprint != args.expected_ground:
            raise NavigableExpertRefused("auditor-owned expected model Ground mismatch")
        verify_native_report(ground, report)
        print(canonical({"independent_neural_oracle": True, "native_receipt_hash_valid": True,
                         "model_ground": ground.fingerprint, "report_digest": report["digest"]}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
