"""A process-independent, evidence-bearing E-CPU virtual processor cursor.

Only the approved affine computational Ground is admitted in VP-001. The
computation is performed by E-CPU's real native orbit-navigation methods;
there is deliberately no local fallback that could masquerade as E-CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVENT_SCHEMA = "ecpu.vp.event.v1"
REALIZER = "mailcpu.law_certifier.lcg_navigate+mailcpu.rrns.affine_orbit_jump"
ECPU_REVISION = "6fd416facdb2aefd6fc7d66a06b014e3d31b5a52"
MAX_COORDINATE = (1 << 63) - 1
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 100000
DEFAULT_GROUND = {"family": "affine-modular-v1", "a": 1664525,
                  "c": 1013904223, "modulus": 4294967296, "x0": 73}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_ZERO = "0" * 64


class ProcessorRefused(ValueError):
    """Invalid Ground, history, intention, or unauthorized state transition."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _keys(value: Any, expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ProcessorRefused(f"{label}: unexpected or missing fields")


def _number(value: Any, label: str, *, low: int = 0, high: int = MAX_COORDINATE) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ProcessorRefused(f"{label}: integer outside admitted range [{low}, {high}]")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ProcessorRefused(f"{label}: invalid identifier")
    return value


def _hash(value: Any, label: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProcessorRefused(f"{label}: invalid digest")
    return value


def ground_check(ground: Any) -> dict[str, Any]:
    _keys(ground, {"family", "a", "c", "modulus", "x0"}, "ground")
    if ground["family"] != "affine-modular-v1":
        raise ProcessorRefused("unadmitted ground family")
    mod = _number(ground["modulus"], "modulus", low=2, high=MAX_COORDINATE)
    for field in ("a", "c", "x0"):
        _number(ground[field], field, high=mod - 1)
    return ground


def native_observe(ground: dict[str, Any], coordinate: int) -> int:
    """Use two distinct actual E-CPU native algorithms; refuse disagreement."""
    ground_check(ground)
    _number(coordinate, "coordinate")
    try:
        from mailcpu.law_certifier import lcg_navigate
        from mailcpu.rrns import affine_orbit_jump
    except ImportError as exc:
        raise ProcessorRefused("native E-CPU mailcpu is required; no fallback") from exc
    a, c, m, x0 = (ground[k] for k in ("a", "c", "modulus", "x0"))
    primary = lcg_navigate(a, c, m, x0, coordinate)
    aa, bb = affine_orbit_jump(a, c, m, coordinate)
    independent = (aa * x0 + bb) % m
    if type(primary) is not int or primary != independent or not 0 <= primary < m:
        raise ProcessorRefused("native E-CPU realizers disagree")
    return primary


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "digest": digest(payload)}


def make_genesis(processor_id: str = "ecpu-vp-primary", ground: dict[str, Any] | None = None) -> dict[str, Any]:
    ground = dict(DEFAULT_GROUND if ground is None else ground)
    ground_check(ground)
    _identifier(processor_id, "processor_id")
    return _seal({"schema": EVENT_SCHEMA, "kind": "genesis", "sequence": 0,
                  "previous": _ZERO, "processor_id": processor_id,
                  "ground": ground, "coordinate": 0,
                  "value": native_observe(ground, 0), "realizer": REALIZER,
                  "ecpu_revision": ECPU_REVISION})


@dataclass(frozen=True)
class Snapshot:
    processor_id: str
    ground: dict[str, Any]
    coordinate: int
    value: int
    sequence: int
    head_digest: str
    requests: dict[str, dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {"processor_id": self.processor_id, "ground": self.ground,
                "coordinate": self.coordinate, "value": self.value,
                "sequence": self.sequence, "head_digest": self.head_digest,
                "realizer": REALIZER, "ecpu_revision": ECPU_REVISION}


def _pairs_no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ProcessorRefused("duplicate JSON member")
        out[key] = value
    return out


def _verify_seal(event: Any) -> None:
    if type(event) is not dict or "digest" not in event:
        raise ProcessorRefused("malformed event")
    _hash(event["digest"], "event digest")
    if digest({k: v for k, v in event.items() if k != "digest"}) != event["digest"]:
        raise ProcessorRefused("event digest mismatch")


def _validate_intent(intent: Any) -> dict[str, Any]:
    if type(intent) is not dict:
        raise ProcessorRefused("intent must be an object")
    op = intent.get("operation")
    required = {"request_id", "actor", "operation"}
    if op in {"peek", "navigate"}:
        required.add("target")
    elif op == "advance":
        required.add("distance")
    elif op != "observe":
        raise ProcessorRefused("unsupported intent operation")
    _keys(intent, required, "intent")
    _identifier(intent["request_id"], "request_id")
    _identifier(intent["actor"], "actor")
    if "target" in intent:
        _number(intent["target"], "target")
    if "distance" in intent:
        _number(intent["distance"], "distance", low=1)
    return intent


def _realize(snapshot: Snapshot, intent: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    op = intent["operation"]
    if op == "navigate":
        target = intent["target"]
        after_coord = target
    elif op == "advance":
        after_coord = _number(snapshot.coordinate + intent["distance"], "advanced coordinate")
        target = after_coord
    elif op == "peek":
        after_coord = snapshot.coordinate
        target = intent["target"]
    else:
        after_coord = snapshot.coordinate
        target = snapshot.coordinate
    observed = {"coordinate": target, "value": native_observe(snapshot.ground, target)}
    after = {"coordinate": after_coord,
             "value": observed["value"] if after_coord == target else snapshot.value}
    return after, observed


def _genesis(event: Any) -> Snapshot:
    _keys(event, {"schema", "kind", "sequence", "previous", "processor_id",
                  "ground", "coordinate", "value", "realizer", "ecpu_revision", "digest"}, "genesis")
    if (event["schema"] != EVENT_SCHEMA or event["kind"] != "genesis" or
        type(event["sequence"]) is not int or event["sequence"] != 0 or event["previous"] != _ZERO or
        type(event["coordinate"]) is not int or event["coordinate"] != 0 or event["realizer"] != REALIZER or
        event["ecpu_revision"] != ECPU_REVISION):
        raise ProcessorRefused("invalid genesis contract")
    _identifier(event["processor_id"], "processor_id")
    ground_check(event["ground"])
    if type(event["value"]) is not int or event["value"] != native_observe(event["ground"], 0):
        raise ProcessorRefused("invalid genesis value")
    return Snapshot(event["processor_id"], event["ground"], 0, event["value"],
                    0, event["digest"], {})


def _verify_transition(event: Any, previous: Snapshot) -> Snapshot:
    _keys(event, {"schema", "kind", "sequence", "previous", "processor_id", "intent",
                  "before", "after", "observation", "realizer", "ecpu_revision", "digest"}, "transition")
    if (event["schema"] != EVENT_SCHEMA or event["kind"] != "intent" or
        type(event["sequence"]) is not int or event["sequence"] != previous.sequence + 1 or
        event["previous"] != previous.head_digest or
        event["processor_id"] != previous.processor_id or
        event["realizer"] != REALIZER or event["ecpu_revision"] != ECPU_REVISION):
        raise ProcessorRefused("broken transition lineage")
    intent = _validate_intent(event["intent"])
    if intent["request_id"] in previous.requests:
        raise ProcessorRefused("request ID duplicated in journal")
    for key in ("before", "after", "observation"):
        _keys(event[key], {"coordinate", "value"}, key)
        _number(event[key]["coordinate"], f"{key}.coordinate")
        _number(event[key]["value"], f"{key}.value", high=previous.ground["modulus"]-1)
    if event["before"] != {"coordinate": previous.coordinate, "value": previous.value}:
        raise ProcessorRefused("before state disagrees with prior event")
    expected_after, expected_observation = _realize(previous, intent)
    if event["after"] != expected_after or event["observation"] != expected_observation:
        raise ProcessorRefused("native computation or observation mismatch")
    # A verified journal is replayed in order. Reuse its ephemeral request index
    # rather than cloning an ever-growing mapping for every record (O(n^2)).
    # This mutation happens only after the full transition has been checked.
    previous.requests[intent["request_id"]] = event
    return Snapshot(previous.processor_id, previous.ground,
                    expected_after["coordinate"], expected_after["value"],
                    event["sequence"], event["digest"], previous.requests)


def parse_journal(encoded: str) -> tuple[list[dict[str, Any]], Snapshot]:
    if len(encoded.encode("utf-8")) > MAX_JOURNAL_BYTES:
        raise ProcessorRefused("journal exceeds configured verification quota")
    lines = encoded.splitlines()
    if not lines or len(lines) > MAX_EVENTS or not encoded.endswith("\n"):
        raise ProcessorRefused("journal missing, incomplete, or over quota")
    events = []
    snapshot = None
    for i, line in enumerate(lines):
        try:
            event = json.loads(line, object_pairs_hook=_pairs_no_duplicate,
                               parse_constant=lambda _: (_ for _ in ()).throw(ProcessorRefused("non-finite JSON")))
        except (ValueError, TypeError) as exc:
            raise ProcessorRefused("invalid journal JSON") from exc
        if canonical(event) != line:
            raise ProcessorRefused("noncanonical journal serialization")
        _verify_seal(event)
        snapshot = _genesis(event) if i == 0 else _verify_transition(event, snapshot)
        events.append(event)
    return events, snapshot


def serialize(events: list[dict[str, Any]]) -> str:
    return "".join(canonical(event) + "\n" for event in events)


def load(path: Path) -> tuple[list[dict[str, Any]], Snapshot]:
    return parse_journal(path.read_text(encoding="utf-8"))


def save_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".vp-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def bootstrap(path: Path, *, processor_id: str = "ecpu-vp-primary") -> Snapshot:
    if path.exists():
        _, state = load(path)
        if state.processor_id != processor_id:
            raise ProcessorRefused("a different processor identity already occupies this journal")
        return state
    genesis = make_genesis(processor_id)
    save_atomic(path, serialize([genesis]))
    return _genesis(genesis)


def run(path: Path, intent: dict[str, Any]) -> dict[str, Any]:
    intent = _validate_intent(intent)
    events, state = load(path)
    old = state.requests.get(intent["request_id"])
    if old is not None:
        if old["intent"] != intent:
            raise ProcessorRefused("same request ID supplied with a different intent")
        return {"replayed": True, "receipt": old, "current": state.public()}
    after, observation = _realize(state, intent)
    event = _seal({"schema": EVENT_SCHEMA, "kind": "intent", "sequence": state.sequence + 1,
                   "previous": state.head_digest, "processor_id": state.processor_id,
                   "intent": dict(intent), "before": {"coordinate": state.coordinate, "value": state.value},
                   "after": after, "observation": observation, "realizer": REALIZER,
                   "ecpu_revision": ECPU_REVISION})
    verified = _verify_transition(event, state)
    save_atomic(path, serialize([*events, event]))
    return {"replayed": False, "receipt": event, "current": verified.public()}
