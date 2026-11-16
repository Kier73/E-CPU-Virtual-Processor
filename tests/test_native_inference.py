"""Live E-CPU VirtualLayer/Hypervisor integration, required in hosted CI."""
from __future__ import annotations
import importlib.util
import tempfile
from pathlib import Path
import unittest

from vp.neural_inference import SparseWeightIndex,sparse_forward,verify_sparse_receipt
from vp.native_inference import NativeInferenceHost,NativeInferenceRefused

ECPU=importlib.util.find_spec("mailcpu") is not None


@unittest.skipUnless(ECPU,"requires actual E-CPU runtime; hosted CI installs pinned revision")
class NativeInferenceTests(unittest.TestCase):
    def setUp(self):
        from test_neural_inference import write_index
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        write_index(root)
        self.index=SparseWeightIndex.open(root/'weights.json')
        self.request={"request_id":"native-one","model_ground":self.index.record["model_ground"],
                      "hidden":[1.,2.,3.,4.,5.,6.]}

    def host(self,grant=True):
        return NativeInferenceHost(lambda req:sparse_forward(self.index,req['hidden'],request_id=req['request_id']),
               node_id='vp-native-test',model_ground=self.index.record['model_ground'],
               operator_id='vp.nn.sparse_swiglu',allow_compute=grant)

    def test_native_real_neural_math_and_replay(self):
        host=self.host()
        first=host.execute(self.request,sequence=1)
        self.assertTrue(first['accepted'],first)
        self.assertEqual(first['handler_calls'],1)
        self.assertEqual(first['hypervisor_action'],'allow')
        self.assertFalse(first['native_replayed'])
        verify_sparse_receipt(self.index,self.request['hidden'],first['result'])
        second=host.execute(self.request,sequence=1)
        self.assertTrue(second['native_replayed'],second)
        self.assertEqual(second['handler_calls'],1)

    def test_native_capability_denial_zero_inference(self):
        host=self.host(False)
        outcome=host.execute(self.request,sequence=1)
        self.assertFalse(outcome['accepted'],outcome)
        self.assertEqual(outcome['handler_calls'],0)
        self.assertEqual(outcome['hypervisor_action'],'deny')

    def test_query_service_binds_prompt_to_native_operator_and_replays(self):
        from vp.query import QueryService,QueryRefused
        class StubLocalModel:
            identity='a'*64
            def infer(self,request):
                # Independent deterministic stub tests authority and query
                # lifecycle only; neural arithmetic is tested separately.
                return {'full_neural_forward_performed':False,
                        'generated_text':request['prompt'].upper()}
        provider=StubLocalModel()
        service=QueryService(provider)
        first=service.ask('hi',request_id='same',max_new_tokens=2)
        self.assertTrue(first['accepted'],first)
        self.assertEqual(first['result']['generated_text'],'HI')
        self.assertEqual(service.ask('hi',request_id='same',max_new_tokens=2),first)
        self.assertEqual(service.host.calls,1)
        with self.assertRaises(QueryRefused):
            service.ask('changed',request_id='same',max_new_tokens=2)

    def test_wrong_model_receipt_fails_before_execution(self):
        host=self.host()
        with self.assertRaises(NativeInferenceRefused):
            host.execute({**self.request,'model_ground':'b'*64},sequence=1)
        self.assertEqual(host.calls,0)


if __name__=='__main__':unittest.main()
