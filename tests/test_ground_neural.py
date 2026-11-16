"""VP-004: a navigated Ground is the predecessor of actual sparse neural math."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from vp.ground_neural import (GroundNeuralProcessor, GroundNeuralRefused,
                              validate_neural_intention)
from vp.neural_inference import SparseWeightIndex
from test_neural_inference import write_index

HAS_ECPU = importlib.util.find_spec("mailcpu") is not None


class ContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_index(self.root)
        self.index = SparseWeightIndex.open(self.root / "weights.json")
        self.request = {
            "request_id": "vp004-req", "model_ground": self.index.record["model_ground"],
            "token_coordinate": 10**12 + 13, "hidden": [1., 2., 3., 4., 5., 6.],
        }

    def test_accepts_distant_token_and_validated_activation(self):
        self.assertEqual(validate_neural_intention(self.index, self.request), self.request)

    def test_fails_closed_on_model_mismatch(self):
        with self.assertRaises(GroundNeuralRefused):
            validate_neural_intention(self.index, {**self.request, "model_ground": "f" * 64})

    def test_fails_closed_on_extra_fields(self):
        with self.assertRaises(GroundNeuralRefused):
            validate_neural_intention(self.index, {**self.request, "unsafe_callback": "exec"})

    def test_refuses_bool_negative_or_oversized_coordinates(self):
        for bad in (True, -1, 2**64, 1.2):
            with self.subTest(bad=bad), self.assertRaises(GroundNeuralRefused):
                validate_neural_intention(self.index, {**self.request, "token_coordinate": bad})

    def test_refuses_nonfinite_wrong_width_or_nontyped_hidden(self):
        for bad in ([1, 2], [float("nan")] * 6, [1] * 5 + ["2"], tuple([1.] * 6)):
            with self.subTest(bad=str(bad)), self.assertRaises(GroundNeuralRefused):
                validate_neural_intention(self.index, {**self.request, "hidden": bad})

    def test_refuses_unbounded_identity(self):
        with self.assertRaises(GroundNeuralRefused):
            validate_neural_intention(self.index, {**self.request, "request_id": "x" * 128})


@unittest.skipUnless(HAS_ECPU, "Requires actual E-CPU (CI installs a pinned version)")
class NativeGroundNeuralTests(ContractTests):
    def test_one_receiver_navigates_then_executes_real_tensor_math_and_replays(self):
        processor = GroundNeuralProcessor(self.index)
        outcome = processor.query(self.request)
        self.assertEqual(outcome["status"], "realized")
        self.assertTrue(outcome["dag_complete"])
        self.assertTrue(outcome["neural_math_executed"])
        self.assertFalse(outcome["full_model_forward_executed"])
        self.assertEqual(outcome["ground"]["observed_position"], self.request["token_coordinate"])
        self.assertEqual(outcome["ground"]["expanded_states"], 0)
        self.assertTrue(outcome["native"]["accepted"])
        self.assertEqual(outcome["native"]["handler_calls"], 1)
        self.assertTrue(outcome["native"]["result"]["receipt"]["actual_neural_math_executed"])
        self.assertEqual(len(outcome["native"]["result"]["receipt"]["output"]), 6)
        self.assertEqual(processor.query(self.request), outcome)
        self.assertEqual(processor.host.calls, 1)
        with self.assertRaises(GroundNeuralRefused):
            processor.query({**self.request, "hidden": [2.] * 6})

    def test_without_ground_read_capability_there_is_no_neural_execution(self):
        from mailcpu.observed_orbit_pipeline import GroundRealizationRefused
        processor = GroundNeuralProcessor(self.index, allow_read=False)
        with self.assertRaises(GroundRealizationRefused):
            processor.query(self.request)
        self.assertEqual(processor.host.calls, 0)

    def test_without_compute_capability_it_does_not_execute_numerics(self):
        processor = GroundNeuralProcessor(self.index, allow_compute=False)
        outcome = processor.query(self.request)
        self.assertEqual(outcome["status"], "denied")
        self.assertFalse(outcome["dag_complete"])
        self.assertFalse(outcome["native"]["accepted"])
        self.assertEqual(processor.host.calls, 0)

    def test_second_fresh_token_coordinate_can_execute_without_repeating_ground_history(self):
        processor = GroundNeuralProcessor(self.index)
        first = processor.query(self.request)
        second = processor.query({**self.request, "request_id": "vp004-second",
                                  "token_coordinate": self.request["token_coordinate"] + 1})
        self.assertEqual(processor.host.calls, 2)
        self.assertNotEqual(first["ground"]["projection_digest"],
                            second["ground"]["projection_digest"])
        self.assertNotEqual(first["dag_digest"], second["dag_digest"])
        self.assertTrue(processor.host.system.environment.kernel.journal.verify())


if __name__ == "__main__":
    unittest.main()
