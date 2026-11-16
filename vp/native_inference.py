"""E-CPU receiver-governed neural execution using the existing Virtual Layer.

Adapts the established IFS TCR-003 NativeCut pattern; E-CPU owns admission,
capability, operator contract, pure-result ledger and replay. It intentionally
never grants authority based on caller-supplied model names or file paths.
"""
from __future__ import annotations

from typing import Any, Callable


class NativeInferenceRefused(ValueError):
    pass


class NativeInferenceHost:
    """Host one locally-provisioned, pure inference handler under E-CPU rights.

    A host is one authorized execution locus. Only immutable model bindings are
    captured by handler; the request payload cannot define a Python callable.
    """

    def __init__(self, handler: Callable[[dict],dict], *, node_id: str,
                 model_ground: str, operator_id: str, allow_compute: bool = True,
                 max_input_bytes: int = 1_000_000, max_output_bytes: int = 1_000_000):
        from mailcpu import (Capability, HeadlessOS, OperatorRegistry, OperatorSpec,
                             SystemIdentity, SystemSeed, SystemSeedEnvelope,
                             VirtualGroundRef, VirtualLayerRuntime)
        from mailcpu.classifier import HypervisorPolicy, LexicalDecisionModel
        if not callable(handler) or len(model_ground) != 64 or len(node_id) > 128:
            raise NativeInferenceRefused("invalid native host binding")
        self.handler = handler
        self.calls = 0
        self.node_id = node_id
        self.model_ground = model_ground
        self.operator_id = operator_id
        self.system = HeadlessOS()
        self.system.hypervisor.classifier = LexicalDecisionModel()
        self.system.hypervisor.policy = HypervisorPolicy(embedding_review_confidence=1.0)
        self.identity = SystemIdentity("vp-model-" + node_id, 1)
        seed = SystemSeed.from_system(self.system)
        envelope = SystemSeedEnvelope(self.identity, "vp-model", "vp003", seed)
        if not self.system.seed_admission.admit(envelope).accepted:
            raise NativeInferenceRefused("model Ground was refused by receiver")
        self.ground = VirtualGroundRef.from_envelope(envelope)
        self.spec = OperatorSpec(operator_id=operator_id,version=1,
             description="Receiver-locked neural inference",input_kind="object",
             output_kind="object",substrates=("cpu",),
             required_rights=frozenset({"compute"}),
             max_input_bytes=max_input_bytes,max_output_bytes=max_output_bytes,
             base_energy=2,energy_per_byte=1,deterministic=True,effect_mode="pure")
        registry = OperatorRegistry()
        registry.register(self.spec, self._handler)
        self.runtime = VirtualLayerRuntime(registry, node_id, frozenset({"cpu"}))
        cap = (Capability("/virtual-layer/operators",frozenset({"compute"})) if allow_compute
               else Capability("/unrelated",frozenset({"read"})))
        self.pid = self.system.environment.kernel.spawn(node_id,capabilities=(cap,),
                                 parent_pid=self.system.environment.init_pid)

    def _handler(self, request: Any, workload: Any) -> dict:
        if not isinstance(request,dict) or request.get("model_ground") != self.model_ground:
            raise NativeInferenceRefused("request is not bound to resident model Ground")
        self.calls += 1
        result = self.handler(request)
        if not isinstance(result,dict):
            raise NativeInferenceRefused("model handler returned invalid result")
        return result

    def execute(self, request: dict, *, sequence: int = 1) -> dict:
        from mailcpu import VirtualBudget, VirtualLayerWorkload, VirtualOperatorRef
        from mailcpu.classifier import HypervisorAction
        if not isinstance(request,dict) or request.get("model_ground") != self.model_ground:
            raise NativeInferenceRefused("wrong model binding before admission")
        operation = VirtualLayerWorkload(
            workload_id="vp003-" + str(request.get("request_id", "invalid")),
            system=self.identity, ground=self.ground,
            intention={"goal":"neural infer","source":self.model_ground},
            operators=(VirtualOperatorRef(self.operator_id,1,"cpu",self.node_id),),
            input_value=request,
            budget=VirtualBudget(4_000_000,1_000_000,1_000_000),
            logical_time=17,eligible_nodes=(self.node_id,),max_hops=1)
        native = self.runtime.execute_operator(self.system,self.pid,"vp-requester",
                           "vp003.inference",sequence,17,operation,self.operator_id)
        if native.action is not HypervisorAction.ALLOW or not native.executed:
            return {"accepted":False,"hypervisor_action":native.action.value,
                    "executed":bool(native.executed),"handler_calls":self.calls}
        effect = native.effect
        if not isinstance(effect,dict) or effect.get("kind")!="virtual_layer.operator_result":
            raise NativeInferenceRefused("missing native operator result")
        if not self.system.environment.kernel.journal.verify():
            raise NativeInferenceRefused("native kernel journal corrupt")
        return {"accepted":True,"hypervisor_action":native.action.value,
                "native_replayed":bool(native.replayed),"handler_calls":self.calls,
                "operator_contract_digest":self.spec.digest,
                "model_ground":self.model_ground,
                "native_fingerprint":native.fingerprint,"result":effect["result"]}
