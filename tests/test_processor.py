"""Protocol and recovery tests; native E-CPU test runs separately in hosted CI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vp import core


def reference_affine(ground, coordinate):
    a, c, m, value = (ground[k] for k in ("a", "c", "modulus", "x0"))
    scale, offset = 1, 0
    while coordinate:
        if coordinate & 1:
            scale, offset = (a * scale) % m, (a * offset + c) % m
        a, c = (a * a) % m, (a * c + c) % m
        coordinate >>= 1
    return (scale * value + offset) % m


class ProcessorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "env" / "processor.jsonl"
        self.patcher = patch.object(core, "native_observe", side_effect=reference_affine)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def intent(self, request_id, operation, **args):
        return {"request_id": request_id, "actor": "human", "operation": operation, **args}

    def test_bootstrap_and_independent_reconstruction(self):
        before = core.bootstrap(self.path)
        self.assertEqual(before.sequence, 0)
        self.assertEqual(before.value, 73)
        self.assertEqual(core.bootstrap(self.path).head_digest, before.head_digest)
        a = core.run(self.path, self.intent("r1", "advance", distance=1234567890123))
        self.assertEqual(a["current"]["coordinate"], 1234567890123)
        self.assertEqual(a["receipt"]["observation"]["value"], reference_affine(core.DEFAULT_GROUND, 1234567890123))
        b = core.run(self.path, self.intent("r2", "peek", target=1))
        self.assertEqual(b["receipt"]["observation"]["coordinate"], 1)
        self.assertEqual(b["current"]["coordinate"], 1234567890123)
        c = core.run(self.path, self.intent("r3", "navigate", target=0))
        self.assertEqual(c["current"]["coordinate"], 0)
        self.assertEqual(c["current"]["value"], 73)
        restored = core.load(self.path)[1]
        self.assertEqual(restored.sequence, 3)
        self.assertEqual(restored.head_digest, c["receipt"]["digest"])
        self.assertEqual(core.run(self.path, self.intent("r4", "observe"))["receipt"]["observation"]["value"], 73)

    def test_duplicate_is_idempotent_but_conflicting_request_refused(self):
        core.bootstrap(self.path)
        intent = self.intent("same", "advance", distance=5)
        first = core.run(self.path, intent)
        contents = self.path.read_bytes()
        second = core.run(self.path, intent)
        self.assertTrue(second["replayed"])
        self.assertEqual(first["receipt"], second["receipt"])
        self.assertEqual(self.path.read_bytes(), contents)
        with self.assertRaisesRegex(core.ProcessorRefused, "same request ID"):
            core.run(self.path, self.intent("same", "advance", distance=6))

    def test_modified_valid_digest_still_refused_by_semantic_reconstruction(self):
        core.bootstrap(self.path)
        core.run(self.path, self.intent("r1", "advance", distance=12))
        events, _ = core.load(self.path)
        events[1]["observation"]["value"] ^= 1
        events[1]["after"]["value"] ^= 1
        original = {k: v for k, v in events[1].items() if k != "digest"}
        events[1]["digest"] = core.digest(original)
        core.save_atomic(self.path, core.serialize(events))
        with self.assertRaisesRegex(core.ProcessorRefused, "native computation or observation"):
            core.load(self.path)

    def test_truncation_broken_link_and_noncanonical_json_refused(self):
        core.bootstrap(self.path)
        core.run(self.path, self.intent("r1", "navigate", target=8))
        original = self.path.read_text()
        self.path.write_text(original[:-1])
        with self.assertRaisesRegex(core.ProcessorRefused, "incomplete"):
            core.load(self.path)
        self.path.write_text(original)
        events, _ = core.load(self.path)
        events[1]["previous"] = "0" * 64
        events[1]["digest"] = core.digest({k: v for k, v in events[1].items() if k != "digest"})
        core.save_atomic(self.path, core.serialize(events))
        with self.assertRaisesRegex(core.ProcessorRefused, "lineage"):
            core.load(self.path)
        self.path.write_text(original.replace('"kind":"genesis"', '"kind": "genesis"'))
        with self.assertRaisesRegex(core.ProcessorRefused, "noncanonical"):
            core.load(self.path)

    def test_different_ground_identity_refused(self):
        core.bootstrap(self.path, processor_id="one")
        with self.assertRaisesRegex(core.ProcessorRefused, "different processor identity"):
            core.bootstrap(self.path, processor_id="two")

    def test_illegal_and_malformed_intents_refused(self):
        core.bootstrap(self.path)
        cases = [
            self.intent("x", "advance", distance=0),
            self.intent("x", "navigate", target=True),
            self.intent("x", "navigate", target=-1),
            self.intent("x", "navigate", target=2 ** 64),
            self.intent("x", "peek", target=1, extra="hidden"),
            self.intent("x", "run_shell"),
            self.intent("bad user", "observe"),
        ]
        for intent in cases:
            with self.subTest(intent=intent), self.assertRaises(core.ProcessorRefused):
                core.run(self.path, intent)
        self.assertEqual(core.load(self.path)[1].sequence, 0)

    def test_bootstrap_refuses_bogus_genesis_even_with_rehashed_digest(self):
        core.bootstrap(self.path)
        events, _ = core.load(self.path)
        events[0]["value"] = 9
        events[0]["digest"] = core.digest({k:v for k,v in events[0].items() if k!="digest"})
        self.path.write_text(core.serialize(events))
        with self.assertRaisesRegex(core.ProcessorRefused, "genesis value"):
            core.bootstrap(self.path)

    def test_no_code_execution_or_unsanctioned_ground_fields(self):
        core.bootstrap(self.path)
        with self.assertRaises(core.ProcessorRefused):
            core.run(self.path, {"request_id":"id", "actor":"human", "operation":"observe", "code":"__import__('os')"})
        ground = {**core.DEFAULT_GROUND, "untrusted":"python"}
        with self.assertRaises(core.ProcessorRefused):
            core.make_genesis("another", ground)


class NativeMailCPUIntegrationTests(unittest.TestCase):
    def test_live_native_orbit_methods_agree_at_large_coordinates(self):
        try:
            from mailcpu.law_certifier import lcg_navigate
            from mailcpu.rrns import affine_orbit_jump
        except ImportError:
            self.skipTest("native E-CPU not installed; GitHub CI installs pinned E-CPU")
        for coordinate in (0, 1, 11, 2**31, (1 << 63)-1):
            with self.subTest(coordinate=coordinate):
                actual = core.native_observe(core.DEFAULT_GROUND, coordinate)
                self.assertEqual(actual, reference_affine(core.DEFAULT_GROUND, coordinate))
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory)/"state.jsonl"
            core.bootstrap(journal)
            core.run(journal, {"actor":"ci", "request_id":"native", "operation":"navigate", "target":2**62})
            self.assertEqual(core.load(journal)[1].coordinate, 2**62)


if __name__ == "__main__":
    unittest.main()
