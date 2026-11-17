"""VP-002: deterministic sparse routing and large architectural scale gates."""
import json
import math
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from vp.moe import ModelGround, MoERefusal, route_experts, verify_route


class ModelStructureTests(unittest.TestCase):
    def test_large_profiles_admitted_without_weight_allocation(self):
        expected = {"qwen3-30b-a3b": (48, 6144, 384, 2048),
                    "qwen3-235b-a22b": (94, 12032, 752, 4096)}
        for name, (layers, experts, active, hidden) in expected.items():
            with self.subTest(name=name):
                model = ModelGround.published(name)
                desc = model.description()
                self.assertEqual((desc["layers"], desc["expert_instances"],
                                  desc["active_expert_invocations_per_token"],
                                  desc["hidden_size"]),
                                 (layers, experts, active, hidden))
                self.assertFalse(desc["weights_loaded"])
                self.assertFalse(desc["neural_inference_performed"])
                self.assertGreater(desc["parameter_inventory"]["declared_elements"], 25_000_000_000)

    def test_per_expert_tensor_shape(self):
        m = ModelGround.published("qwen3-235b-a22b")
        self.assertEqual(m.tensor_shape(93, "gate_proj.weight", expert=127), (1536, 4096))
        self.assertEqual(m.tensor_shape(93, "down_proj.weight", expert=127), (4096, 1536))
        self.assertEqual(m.tensor_shape(0, "q_proj.weight"), (8192, 4096))
        self.assertEqual(m.tensor_shape(0, "k_proj.weight"), (512, 4096))
        self.assertEqual(m.tensor_shape(0, "router.weight"), (128, 4096))
        self.assertEqual(m.tensor_ref(0, "router.weight")["tensor"], "model.layers.0.mlp.gate.weight")
        self.assertEqual(m.tensor_ref(1, "gate_proj.weight", expert=2)["tensor"],
                         "model.layers.1.mlp.experts.2.gate_proj.weight")

    def test_address_refusals(self):
        m = ModelGround.published("qwen3-30b-a3b")
        for args in ((48, "q_proj.weight", None), (0, "down_proj.weight", None),
                     (0, "up_proj.weight", 128), (0, "q_proj.weight", 0),
                     (0, "rogue.weight", None)):
            with self.subTest(args=args), self.assertRaises(MoERefusal):
                m.tensor_shape(args[0], args[1], expert=args[2])

    def test_kv_cache_scale_exact(self):
        m = ModelGround.published("qwen3-235b-a22b")
        self.assertEqual(m.kv_budget(1)["kv_bytes_per_token"], 94 * 4 * 128 * 2 * 2)
        self.assertEqual(m.kv_budget(40960)["kv_bytes_total"], 94 * 4 * 128 * 2 * 2 * 40960)
        with self.assertRaises(MoERefusal):
            m.kv_budget(40961)

    def test_source_configuration_mutation_refused(self):
        m = ModelGround.published("qwen3-30b-a3b")
        with self.assertRaises(TypeError):
            m.parameters["num_experts_per_tok"] = 7
        m.validate()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.ground = ModelGround.published("qwen3-235b-a22b")

    def report(self, scores):
        r = route_experts(self.ground, layer=93, token=32000, logits=scores)
        return {"receipt": r.snapshot(), "digest": r.receipt_digest}

    def test_sparse_top_k_and_receipt(self):
        scores = [float(i % 11) for i in range(128)]
        report = self.report(scores)
        verify_route(self.ground, report)
        routes = report["receipt"]["routes"]
        self.assertEqual(len(routes), 8)
        self.assertEqual(len({r["expert"] for r in routes}), 8)
        self.assertAlmostEqual(sum(r["weight"] for r in routes), 1, places=14)
        self.assertEqual(len(routes[0]["tensor_addresses"]), 3)
        self.assertFalse(report["receipt"]["neural_inference_performed"])
        self.assertIsNone(report["receipt"]["weight_manifest_digest"])

    def test_tie_break_by_expert_index(self):
        r = self.report([0.0] * 128)
        self.assertEqual([v["expert"] for v in r["receipt"]["routes"]], list(range(8)))
        for v in r["receipt"]["routes"]:
            self.assertAlmostEqual(v["weight"], 0.125)

    def test_extreme_softmax_logits(self):
        scores = [-10000.0] * 128
        scores[127] = 10000.0
        r = self.report(scores)
        self.assertEqual(r["receipt"]["routes"][0]["expert"], 127)
        self.assertAlmostEqual(r["receipt"]["routes"][0]["weight"], 1.0)
        verify_route(self.ground, r)

    def test_tampering_even_if_rehashed_is_refused(self):
        report = self.report([float(i) for i in range(128)])
        report["receipt"]["routes"][0]["expert"] = 0
        from vp.moe import digest
        report["digest"] = digest(report["receipt"])
        with self.assertRaises(MoERefusal):
            verify_route(self.ground, report)

    def test_cross_model_ground_refused(self):
        report = self.report([float(i) for i in range(128)])
        with self.assertRaises(MoERefusal):
            verify_route(ModelGround.published("qwen3-30b-a3b"), report)

    def test_bad_logits_refused(self):
        for vec in ([1] * 127, [float('nan')] * 128, [float('inf')] * 128, ["a"] * 128):
            with self.subTest(bad=vec[:2]), self.assertRaises(MoERefusal):
                self.report(vec)
        with self.assertRaises(MoERefusal):
            route_experts(self.ground, layer=94, token=0, logits=[0] * 128)
        with self.assertRaises(MoERefusal):
            route_experts(self.ground, layer=0, token=40960, logits=[0] * 128)

    def test_optional_weight_digest_contract(self):
        zeros = [0.0] * 128
        with self.assertRaises(MoERefusal):
            route_experts(self.ground, layer=0, token=0, logits=zeros,
                          weight_manifest_digest="not-a-digest")
        r = route_experts(self.ground, layer=0, token=0, logits=zeros,
                          weight_manifest_digest="a" * 64)
        self.assertEqual(r.snapshot()["weight_manifest_digest"], "a" * 64)
        self.assertFalse(r.snapshot()["tensor_payloads_loaded"])

    def test_cli_describe_and_route_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            logits_path = root / "router.json"
            report_path = root / "report.json"
            logits_path.write_text(json.dumps([float(i % 13) for i in range(128)]))
            proc = subprocess.run([sys.executable, "-m", "vp.moe", "route", "--model", "qwen3-30b-a3b",
                                   "--logits", str(logits_path), "--layer", "4", "--token", "10"],
                                  capture_output=True, text=True, check=True)
            report = json.loads(proc.stdout)
            report_path.write_text(json.dumps(report))
            verify = subprocess.run([sys.executable, "-m", "vp.moe", "verify", "--model", "qwen3-30b-a3b",
                                     "--receipt", str(report_path)], capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(verify.stdout)["verified"])


if __name__ == "__main__":
    unittest.main()


class FrontierTests(unittest.TestCase):
    def test_whole_235b_layer_frontier_is_bounded(self):
        from vp.moe import compile_layer_frontier, verify_layer_frontier
        ground = ModelGround.published("qwen3-235b-a22b")
        event = route_experts(ground, layer=93, token=40959,
                              logits=[float(i % 17) for i in range(128)])
        report = {"receipt": event.snapshot(), "digest": event.receipt_digest}
        plan = compile_layer_frontier(ground, report)
        verify_layer_frontier(ground, report, plan)
        nodes = plan["frontier"]["tasks"]
        self.assertEqual(len(nodes), 25)
        self.assertEqual(sum(n["operator"] == "nn.expert_swiglu" for n in nodes), 8)
        self.assertTrue(all(n["execution"] == "not-performed" for n in nodes))
        self.assertFalse(plan["frontier"]["neural_inference_performed"])
        self.assertEqual(nodes[-1]["operator"], "nn.add")
        self.assertEqual(nodes[-1]["dependencies"], ["l93:t40959:attn_residual", "l93:t40959:sparse_reduce"])
        self.assertLess(len(nodes), 256)

    def test_frontier_refuses_tampered_dependencies(self):
        from vp.moe import compile_layer_frontier, verify_layer_frontier, digest
        m = ModelGround.published("qwen3-30b-a3b")
        r = route_experts(m, layer=0, token=0, logits=[1.0] * 128)
        receipt = {"receipt": r.snapshot(), "digest": r.receipt_digest}
        plan = compile_layer_frontier(m, receipt)
        plan["frontier"]["tasks"][-1]["dependencies"] = []
        plan["digest"] = digest(plan["frontier"])
        with self.assertRaises(MoERefusal):
            verify_layer_frontier(m, receipt, plan)


class ECPUIntegrationTests(unittest.TestCase):
    def test_native_dag_ready_and_snapshot_reconstruction(self):
        try:
            from mailcpu.computation_dag import NavigableComputationDAG
        except ImportError:
            self.skipTest("native E-CPU is not installed in this local Python runtime")
        from vp.moe import compile_ecpu_dag
        m = ModelGround.published("qwen3-235b-a22b")
        r = route_experts(m, layer=93, token=40959, logits=[float(i) for i in range(128)])
        receipt = {"receipt": r.snapshot(), "digest": r.receipt_digest}
        dag = compile_ecpu_dag(m, receipt)
        self.assertEqual(len(dag.nodes), 25)
        self.assertEqual([node.node_id for node in dag.ready_nodes()], ["l93:t40959:hidden"])
        snap = dag.snapshot()
        restored = NavigableComputationDAG.from_snapshot(snap)
        self.assertEqual(restored.snapshot()["digest"], snap["digest"])
        self.assertEqual([node.node_id for node in restored.ready_nodes()], ["l93:t40959:hidden"])
