"""VP-005 exact navigable expert Ground, independent neural oracle and refusals."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
import unittest

from vp.navigable_experts import (
    ExpertGround, NavigableExpertRefused, build_plan, canonical, digest,
    jump_affine_independent, observe, verify_observation, verify_plan,
    verify_route, run_native, NativeExpertSession, verify_native_report,
)


class NavigableExpertTests(unittest.TestCase):
    def test_exact_small_topk_exhaustive_across_full_orbit(self):
        for n in (8, 16, 64, 128):
            ground = ExpertGround(n, min(n, 8), 3, 2)
            centers = set()
            for token in range(n):
                route = ground.route(token)
                verify_route(ground, route)
                centers.add(route["center"])
                exhaustive = sorted(range(n), key=lambda j: (abs(j-route["center"]), j))[:ground.top_k]
                self.assertEqual(route["selected_experts"], exhaustive)
                self.assertEqual(sum(Fraction(w) for w in route["weights"]), 1)
            self.assertEqual(len(centers), n)

    def test_independent_affine_covers_extreme_token_coordinates_without_traversal(self):
        ground = ExpertGround(1 << 48, 8, 3, 4)
        for t in (0, 1, 63, (1 << 40), (1 << 56)+73, (1 << 63)-1):
            expected = ground.center(t)
            self.assertEqual(expected, jump_affine_independent(5, 1, 1 << 48, 7, t))
            self.assertEqual(len(ground.route(t)["selected_experts"]), 8)
            self.assertTrue(all(0 <= j < 1 << 48 for j in ground.route(t)["selected_experts"]))
        self.assertGreater(ground.logical_parameters, 10**15)
        self.assertLess(len(canonical(ground.to_dict())), 256)

    def test_procedural_realizations_match_independent_neural_oracle(self):
        for n in (8, 128, 1 << 48):
            ground = ExpertGround(n, 5, 3, 4)
            for t in (0, 11, 312):
                route = ground.route(t)
                for hidden in ((1, 2, 3), (0, -5, 7), (-2, -3, -4)):
                    result = observe(ground, route, hidden)
                    verify_observation(ground, route, result)
                    self.assertEqual(result["materialized_experts"], 5)
                    self.assertEqual(len(result["result"]), 3)

    def test_refuses_forged_route_even_with_rehashed_digest(self):
        ground = ExpertGround(128, 8, 3, 4)
        route = ground.route(157)
        forged = {**route, "selected_experts": [0] * 8}
        body = {k: v for k, v in forged.items() if k != "digest"}
        forged["digest"] = digest(body)
        with self.assertRaisesRegex(NavigableExpertRefused, "route disagrees"):
            verify_route(ground, forged)
        with self.assertRaises(NavigableExpertRefused):
            verify_route(ground, {**route, "weights": ["2/2"]*8})

    def test_refuses_forged_neural_math_even_with_rehash(self):
        ground = ExpertGround(64, 4, 3, 4)
        route = ground.route(7)
        result = observe(ground, route, (4, 2, 1))
        forged = {**result, "result": ["0/1"] * 3}
        forged["digest"] = digest({k: v for k, v in forged.items() if k != "digest"})
        with self.assertRaisesRegex(NavigableExpertRefused, "independent neural"):
            verify_observation(ground, route, forged)

    def test_receiver_owned_ground_and_intention_identity(self):
        ground = ExpertGround(128, 8, 3, 4)
        plan = build_plan(ground, coordinate=1 << 40, hidden=(1, 2, 3), request_id="q1")
        verify_plan(ground, plan)
        mismatch = ExpertGround(256, 8, 3, 4)
        with self.assertRaises(NavigableExpertRefused):
            verify_plan(mismatch, plan)
        with self.assertRaises(NavigableExpertRefused):
            verify_plan(ground, {**plan, "request_id": "q2"})
        with self.assertRaises(NavigableExpertRefused):
            verify_plan(ground, {**plan, "hidden": [9, 2, 3]})

    def test_rejects_unsupported_family_and_invalid_coordinate(self):
        with self.assertRaisesRegex(NavigableExpertRefused, "power of two"):
            ExpertGround(100, 8, 3, 4)
        with self.assertRaisesRegex(NavigableExpertRefused, "full period"):
            ExpertGround(128, 8, 3, 4, multiplier=2)
        with self.assertRaises(NavigableExpertRefused):
            ExpertGround(True, 8, 3, 4)
        g = ExpertGround(128, 8, 3, 4)
        with self.assertRaises(NavigableExpertRefused):
            g.route(-1)
        with self.assertRaises(NavigableExpertRefused):
            g.route(1 << 63)

    @unittest.skipUnless(importlib.util.find_spec("mailcpu") is not None, "native E-CPU unavailable")
    def test_native_ecpu_law_is_same_for_distant_coordinates(self):
        g = ExpertGround(1 << 48, 8, 3, 4)
        for t in (0, 7, 1000, (1 << 56) + 73, (1 << 63) - 1):
            self.assertEqual(g.center(t, require_ecpu=True), g.center(t))

    @unittest.skipUnless(importlib.util.find_spec("mailcpu") is not None, "native E-CPU unavailable")
    def test_native_hypervisor_executes_neural_result_with_exact_replay(self):
        g = ExpertGround(1 << 40, 8, 3, 4)
        plan = build_plan(g, coordinate=(1 << 56) + 73, hidden=(3, -2, 5), request_id="native-001")
        receiver = NativeExpertSession(g)
        report = receiver.query(plan, sequence=1)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["handler_calls"], 1)
        self.assertEqual(report["hypervisor_action"], "allow")
        self.assertFalse(report["native_replayed"])
        verify_observation(g, plan["route"], report["result"])
        replay = receiver.query(plan, sequence=1)
        self.assertTrue(replay["native_replayed"])
        self.assertEqual(replay["handler_calls"], 1)
        self.assertEqual(replay["result"], report["result"])
        with self.assertRaises(NavigableExpertRefused):
            receiver.query({**plan, "hidden": [8, -2, 5]}, sequence=1)
        refused = run_native(g, plan, allow_compute=False)
        self.assertFalse(refused["accepted"])
        self.assertEqual(refused["handler_calls"], 0)


if __name__ == "__main__":
    unittest.main()
