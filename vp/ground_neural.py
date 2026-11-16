"""VP-004: one real E-CPU Ground observation feeding one native neural operation.

Uses MailCPU's EXISTING StructuralMemoryRegistry, VerifiedGroundRealizer,
VirtualLayerRuntime, Hypervisor and NavigableComputationDAG. No new executor or
admission authority. The Ground is a declared token-coordinate law, NOT proof
that an LLM's hidden state is derivable from its token index.

A supplied hidden activation is numerically evaluated using receiver-resident,
verified MoE parameter bytes. This is a real neural sublayer and NOT an entire
language-model forward pass, prefill/decode, or a physical path-native kernel.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .native_inference import NativeInferenceHost
from .neural_inference import SparseWeightIndex, sparse_forward, verify_sparse_receipt


class GroundNeuralRefused(ValueError):
    """Refuse a mismatched Ground, coordinate, numerical input, or forged result."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def validate_neural_intention(index: SparseWeightIndex, request: Mapping[str, Any]) -> dict[str, Any]:
    """Strict, non-eval input contract. Model identity and parameters are receiver-owned."""
    if not isinstance(request, Mapping) or set(request) != {
        "request_id", "model_ground", "token_coordinate", "hidden"
    }:
        raise GroundNeuralRefused("neural Ground query requires exactly four typed fields")
    rid = request["request_id"]
    if type(rid) is not str or not 1 <= len(rid) <= 96 or rid.strip() != rid:
        raise GroundNeuralRefused("invalid request identity")
    if request["model_ground"] != index.record["model_ground"]:
        raise GroundNeuralRefused("model Ground differs from admitted weight index")
    token = request["token_coordinate"]
    if type(token) is not int or not 0 <= token <= 2**63 - 1:
        raise GroundNeuralRefused("invalid logical token coordinate")
    values = request["hidden"]
    if (not isinstance(values, list) or len(values) != index.record["hidden_size"]
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values)):
        raise GroundNeuralRefused("activation does not match admitted hidden width or finiteness")
    return {"request_id": rid, "model_ground": request["model_ground"],
            "token_coordinate": token, "hidden": list(values)}


class GroundNeuralProcessor:
    """Queryable native receiver with a single Hypervisor for Ground and MoE math.

    The token-coordinate Ground is a trusted declared arithmetic coordinate map.
    It is not an empirical discovery of a model or tokenization semantics. It
    makes the ground-coordinate read an actual, authorized predecessor of the
    inference operator; neither the caller nor a floating model policy can
    substitute a result or enlarge a capability.
    """

    def __init__(self, index: SparseWeightIndex, *, allow_read: bool = True,
                 allow_compute: bool = True) -> None:
        from mailcpu import (Capability, ContextualOrbitAddress, ObserverContract,
                             SchedulerReading, VirtualStructuralMemoryRef,
                             contextual_orbit_context, structural_memory_spec_representation_ref)
        from mailcpu.law_certifier import Law
        from mailcpu.semantic_translation import (ObservationContract, RepresentationRef,
                                                  canonical_value_digest)
        self.index = index
        self._issued: dict[str, tuple[str, dict[str, Any]]] = {}
        self._queries: dict[str, tuple[str, dict[str, Any]]] = {}
        self.host = NativeInferenceHost(
            self._infer_verified, node_id="vp004-native-neural-" + index.digest[:12],
            model_ground=index.record["model_ground"], operator_id="vp.nn.ground_sparse_swiglu",
            allow_compute=allow_compute,
            structural_memory_admission_policy=lambda proposal: proposal.spec.domain == "vp-neural-token-orbit",
        )
        from mailcpu.classifier import HypervisorPolicy
        # Use the same explicitly configured local policy as E-CPU's
        # observed-orbit authorization integration tests.
        self.host.system.hypervisor.policy = HypervisorPolicy(
            review_confidence=0.0, automatic_confidence=0.0,
            destructive_confidence=0.0, danger_review_probability=1.0,
            embedding_review_confidence=1.0,
        )
        self.registry = self.host.system.structural_memory
        self.description = RepresentationRef.from_payload(
            "ecpu-vp.model-neural-ground", 1,
            {"model_ground": index.record["model_ground"], "weight_index": index.digest,
             "layer": index.record["layer"], "meaning": "token-coordinate neural operation"},
        )
        self.extensions = {"model_ground": index.record["model_ground"],
                           "weight_index": index.digest, "layer": index.record["layer"]}
        self.scheduler_id = "vp-neural-token-scheduler"
        self.context = contextual_orbit_context(
            self.host.identity, self.host.ground, self.description,
            scheduler_id=self.scheduler_id, extensions=self.extensions,
        )
        # A DECLARED position law. These synthetic witnesses test the local law
        # but do not certify an LLM, learned weights, or externally observed tokens.
        self.position_law = Law("int", {"x0": 0, "step": 1, "mod": 2**63}, 1, "GROUND")
        proposal = self.registry.propose(
            "vp-position-" + index.digest[:24], self.position_law,
            {i: i for i in range(16)}, domain="vp-neural-token-orbit", context=self.context,
        )
        self.version = self.registry.admit(proposal.proposal_id, expected_revision=0)
        self.observer = ObserverContract(
            observer_id="vp004-token-observer", revision=1,
            subject=structural_memory_spec_representation_ref(self.version.spec),
            interpretation=self.description,
            observation=ObservationContract.from_definition(
                "ecpu-vp.token-coordinate.observation", 1,
                {"fields": ["coordinate", "value", "value_digest"]},
            ),
            context_digest=canonical_value_digest(self.context),
        )
        self._binding = VirtualStructuralMemoryRef.from_version(
            "vp004-position-generator", self.version, 0,
        )
        self._address = ContextualOrbitAddress(
            system_identity=self.host.identity, base_ground=self.host.ground,
            computational_ground=self.description,
            coordinate=SchedulerReading(self.scheduler_id, self.host.ground.coordinate),
            structural_memory=self._binding, context_extensions=self.extensions,
        )
        from mailcpu.observed_orbit_pipeline import virtual_ground_read_resource
        resource = virtual_ground_read_resource(
            self.version.spec.memory_id, revision=self.version.spec.revision,
            context_digest=self._address.context_digest,
        )
        self.read_pid = self.host.system.environment.kernel.spawn(
            "vp004-orbit-observer",
            capabilities=(Capability(resource, frozenset({"read"})) if allow_read
                          else Capability("/unrelated", frozenset({"read"})),),
            parent_pid=self.host.system.environment.init_pid,
        )

    def _infer_verified(self, request: dict[str, Any]) -> dict[str, Any]:
        """Receiver checks the locally issued observation before numerical work."""
        ident = request.get("request_id")
        issued = self._issued.get(ident)
        if issued is None:
            raise GroundNeuralRefused("native operator requires locally realized predecessor Ground")
        record = issued[1]
        if (request.get("token_coordinate") != record["token_coordinate"]
                or request.get("observed_position") != record["observed_position"]
                or request.get("ground_receipt_digest") != record["ground_receipt_digest"]
                or request.get("model_ground") != self.index.record["model_ground"]
                or _digest(request.get("hidden")) != record["hidden_digest"]):
            raise GroundNeuralRefused("native neural input is not bound to observed Ground predecessor")
        return sparse_forward(self.index, request["hidden"], request_id=ident)

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Navigate the exact token coordinate, then execute verified sparse math.

        The two operations share the SAME HeadlessOS Hypervisor. An E-CPU DAG
        commits the ground read before the neural result can become ready.
        """
        from mailcpu import SchedulerReading
        from mailcpu.computation_dag import ComputationNode, NavigableComputationDAG
        from mailcpu.observed_orbit_pipeline import (NavigationPlan, VerifiedGroundRealizer,
                                                     interpret_intention)
        req = validate_neural_intention(self.index, request)
        request_digest = _digest(req)
        previous = self._queries.get(req["request_id"])
        if previous is not None:
            if previous[0] != request_digest:
                raise GroundNeuralRefused("request ID reused with different mathematical intention")
            return json.loads(_canonical(previous[1]))
        coordinate = req["token_coordinate"]
        address = replace(
            self._address,
            coordinate=SchedulerReading(self.scheduler_id, self.host.ground.coordinate + coordinate),
            structural_memory=replace(self._binding, coordinate=coordinate),
        )
        intent = interpret_intention({
            "operation": "observe token coordinate before native sparse neural realization",
            "model_ground": req["model_ground"], "token_coordinate": coordinate,
            "resource": address.digest,
        })
        realized = VerifiedGroundRealizer(self.registry).realize(
            self.host.system, self.read_pid, address, self.observer,
            NavigationPlan.direct(address.coordinate.tick, intent_digest=intent.digest),
            host_actor_id="vp004-ground-observer", intention=intent,
        )
        if not realized.verified or realized.navigation.projection.value != coordinate:
            raise GroundNeuralRefused("admitted token-coordinate Ground disagrees with request")
        receipt = realized.receipt_digest
        ground = {
            "token_coordinate": coordinate,
            "observed_position": realized.navigation.projection.value,
            "ground_receipt_digest": receipt,
            "observation_digest": realized.navigation.observation.digest,
            "projection_digest": realized.navigation.projection.projection_digest,
            "ground_revision": self.version.spec.revision,
            "model_ground": req["model_ground"], "weight_index_digest": self.index.digest,
            "navigator": realized.navigation.generator_family,
            "expanded_states": realized.navigation.expanded_states,
        }
        ctx = _digest({"index": self.index.digest, "request": request_digest,
                       "address": address.digest})
        graph = NavigableComputationDAG((
            ComputationNode("ground-read", 1, "mailcpu.ground.observe", "structural-memory",
                            ctx, (), {"address_digest": address.digest,
                                       "intention_digest": intent.digest}),
            ComputationNode("neural-forward", 1, "vp.nn.ground_sparse_swiglu",
                            "virtual-layer", ctx, ("ground-read",),
                            {"model_ground": req["model_ground"],
                             "index_digest": self.index.digest,
                             "input_digest": _digest(req["hidden"])}),
        ))
        if [node.node_id for node in graph.ready_nodes()] != ["ground-read"]:
            raise GroundNeuralRefused("Ground node must be first ready frontier")
        graph.commit("ground-read", ground)
        if [node.node_id for node in graph.ready_nodes()] != ["neural-forward"]:
            raise GroundNeuralRefused("neural node was not opened by Ground observation")
        native_request = {**req, "observed_position": coordinate,
                          "ground_receipt_digest": receipt}
        self._issued[req["request_id"]] = (request_digest, {
            **ground, "hidden_digest": _digest(req["hidden"]),
        })
        try:
            native = self.host.execute(native_request, sequence=len(self._queries) + 1)
        finally:
            self._issued.pop(req["request_id"], None)
        if not native["accepted"]:
            # Denial is a result, not a completed neural operation.
            return {"status": "denied", "model_ground": req["model_ground"],
                    "ground": ground, "native": native,
                    "dag_complete": False}
        verify_sparse_receipt(self.index, req["hidden"], native["result"])
        graph.commit("neural-forward", {
            "operator_contract_digest": native["operator_contract_digest"],
            "native_fingerprint": native["native_fingerprint"],
            "numerical_digest": native["result"]["digest"],
            "predecessor_ground_receipt": receipt,
        })
        snap = graph.snapshot()
        restored = NavigableComputationDAG.from_snapshot(snap)
        if not restored.is_complete or restored.digest != graph.digest:
            raise GroundNeuralRefused("E-CPU DAG snapshot failed to reconstruct")
        result = {"schema": "ecpu-vp.ground-neural-query.v1",
                  "status": "realized", "request_id": req["request_id"],
                  "model_ground": req["model_ground"], "weight_index_digest": self.index.digest,
                  "request_digest": request_digest, "ground": ground, "native": native,
                  "dag_digest": graph.digest,
                  "dag_snapshot_digest": _digest(snap),
                  "dag_complete": True,
                  "neural_math_executed": True, "full_model_forward_executed": False}
        result["digest"] = _digest(result)
        self._queries[req["request_id"]] = (request_digest, result)
        return json.loads(_canonical(result))


def main(argv: Sequence[str] | None = None) -> int:
    """Use a provided tensor index: no implicit download or model substitution."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="E-CPU Ground-to-real-MoE native neural query")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--token-coordinate", type=int, default=0)
    parser.add_argument("--deny-read", action="store_true")
    parser.add_argument("--deny-compute", action="store_true")
    args = parser.parse_args(argv)
    index = SparseWeightIndex.open(args.index)
    hidden = json.loads(args.hidden.read_text(encoding="utf-8"))
    processor = GroundNeuralProcessor(index, allow_read=not args.deny_read,
                                     allow_compute=not args.deny_compute)
    result = processor.query({"request_id": args.request_id,
                              "model_ground": index.record["model_ground"],
                              "token_coordinate": args.token_coordinate,
                              "hidden": hidden})
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "realized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
