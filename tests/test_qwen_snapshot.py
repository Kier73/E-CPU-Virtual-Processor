"""Test two-phase real Safetensors source discovery without remote downloads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file

from vp.qwen_snapshot import prepare_qwen_index
from vp.neural_inference import SparseWeightIndex, sparse_forward, InferenceRefused


class MiniQwenGround:
    identifier='a'*64
    source_model='local-test-miniqwen'
    revision='f'*40
    c={'num_hidden_layers':2,'hidden_size':6,'num_experts':5,'num_experts_per_tok':2}
    def validate(self):pass
    def tensor_shape(self,layer,role,*,expert=None):
        if role=='router.weight':return (5,6)
        if role in ('gate_proj.weight','up_proj.weight'):return (4,6)
        if role=='down_proj.weight':return (6,4)
        raise ValueError(role)


class QwenSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.hidden=[1.0]*6
        self.ground=MiniQwenGround()
        rng=np.random.default_rng(123)
        arrays={}
        router=np.zeros((5,6),dtype=np.float32)
        router[:,0]=[1.,0.5,0.1,0.0,-0.1]
        arrays['model.layers.0.mlp.gate.weight']=torch.from_numpy(router)
        for e in (0,1):
            for role,shape in [('gate_proj',(4,6)),('up_proj',(4,6)),('down_proj',(6,4))]:
                name=f'model.layers.0.mlp.experts.{e}.{role}.weight'
                arrays[name]=torch.tensor(rng.normal(scale=0.1,size=shape),dtype=torch.float32)
        save_file(arrays,str(self.root/'model-00001-of-00001.safetensors'))
        data={'weight_map':{name:'model-00001-of-00001.safetensors' for name in arrays}}
        (self.root/'model.safetensors.index.json').write_text(json.dumps(data))

    def test_real_router_selects_and_loads_pretrained_style_shards(self):
        out=prepare_qwen_index(self.root,self.ground,layer=0,hidden=self.hidden,
                      output=self.root/'vp003-layer-0.json')
        self.assertEqual(out['selected_experts'],[0,1])
        self.assertEqual(out['physical_shards_hashed'],1)
        self.assertFalse(out['tensor_payloads_executed'])
        ix=SparseWeightIndex.open(self.root/'vp003-layer-0.json',expected_sha256=out['index_digest'])
        result=sparse_forward(ix,self.hidden,request_id='real-safetensors')
        self.assertTrue(result['receipt']['actual_neural_math_executed'])
        self.assertEqual(len(result['receipt']['loaded_tensor_digests']),7)
        self.assertTrue(all(np.isfinite(result['receipt']['output'])))

    def test_source_manifest_anchor_refuses_substitution(self):
        with self.assertRaises(InferenceRefused):
            prepare_qwen_index(self.root,self.ground,layer=0,hidden=self.hidden,
                     output=self.root/'vp003-layer-0.json',expected_weight_map_sha256='0'*64)

    def test_pretrained_snapshot_missing_selected_shard_refused(self):
        (self.root/'model-00001-of-00001.safetensors').unlink()
        with self.assertRaises(InferenceRefused):
            prepare_qwen_index(self.root,self.ground,layer=0,hidden=self.hidden,
                     output=self.root/'vp003-layer-0.json')


if __name__=='__main__':unittest.main()
