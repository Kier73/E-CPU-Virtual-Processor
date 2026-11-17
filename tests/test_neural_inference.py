"""Real parameter-bearing sparse forward, tampering and architecture scale."""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np

from vp.neural_inference import SparseWeightIndex, InferenceRefused, sparse_forward, verify_sparse_receipt
from vp.query import validate_query, QueryRefused
from vp.moe import ModelGround


def write_index(root:Path,*,hidden=6,experts=7,top_k=2,scale=0.035):
    rng=np.random.default_rng(42)
    manifest={"schema":"ecpu-vp.sparse-weights.v1", "model_ground":'a'*64,
        "layer":0,"hidden_size":hidden,"num_experts":experts,"top_k":top_k,
        "provenance":{"kind":"synthetic","source":"independent seeded tensors"},"tensors":{}}
    values={}
    def tensor(name,arr):
        fname=f"w{len(manifest['tensors']):03d}.npy"
        path=root/fname
        np.save(path,arr)
        manifest['tensors'][name]={"file":fname,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "shape":list(arr.shape),"dtype":arr.dtype.name}
        values[name]=arr
    router=np.zeros((experts,hidden),dtype=np.float64)
    for i in range(experts):router[i,0]=(experts-i)*0.025
    tensor("router",router)
    for e in range(top_k):
        tensor(f"experts/{e}/gate",rng.normal(scale=scale,size=(5,hidden)).astype('float64'))
        tensor(f"experts/{e}/up",rng.normal(scale=scale,size=(5,hidden)).astype('float64'))
        tensor(f"experts/{e}/down",rng.normal(scale=scale,size=(hidden,5)).astype('float64'))
    (root/"weights.json").write_text(json.dumps(manifest,sort_keys=True))
    return manifest,values


class SparseInferenceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.manifest,self.tensors=write_index(self.root)
        self.index=SparseWeightIndex.open(self.root/"weights.json")
        self.hidden=[1.,2.,3.,4.,5.,6.]

    def test_forward_is_actual_tensor_math_and_matches_independent_torch(self):
        import torch
        r=sparse_forward(self.index,self.hidden,request_id="neural-1")
        result=r["receipt"]
        self.assertTrue(result["actual_neural_math_executed"])
        self.assertFalse(result["full_llm_inference_performed"])
        self.assertEqual(result["selected_experts"],[0,1])
        x=torch.tensor(self.hidden,dtype=torch.float64)
        router=torch.tensor(self.tensors["router"])
        logits=router @ x
        probs=torch.softmax(logits[result["selected_experts"]],dim=0)
        ref=torch.zeros_like(x)
        for slot,e in enumerate(result["selected_experts"]):
            gate=torch.tensor(self.tensors[f"experts/{e}/gate"])
            up=torch.tensor(self.tensors[f"experts/{e}/up"])
            down=torch.tensor(self.tensors[f"experts/{e}/down"])
            ref+=probs[slot]*(down @ (torch.nn.functional.silu(gate @ x)*(up @ x)))
        np.testing.assert_allclose(result["output"],ref.tolist(),rtol=1e-12,atol=1e-12)
        verify_sparse_receipt(self.index,self.hidden,r)
        self.assertEqual(len(result["loaded_tensor_digests"]),7)

    def test_unselected_experts_need_not_be_materialized(self):
        self.assertEqual(self.index.record["num_experts"],7)
        self.assertEqual(len(self.index.record["tensors"]),7)
        self.assertEqual(len(sparse_forward(self.index,self.hidden,request_id="q")["receipt"]["output"]),6)

    def test_tampered_payload_refused(self):
        p=self.root/self.manifest["tensors"]["experts/0/gate"]["file"]
        with p.open('ab') as output:output.write(b'altered')
        with self.assertRaises(InferenceRefused):sparse_forward(self.index,self.hidden,request_id="q")

    def test_tampered_index_refused_by_external_anchor(self):
        original=self.index.digest
        self.manifest["top_k"]=1
        (self.root/"weights.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(InferenceRefused):
            SparseWeightIndex.open(self.root/"weights.json",expected_sha256=original)

    def test_forged_receipt_rejected_even_after_rehash(self):
        r=sparse_forward(self.index,self.hidden,request_id="q")
        r["receipt"]["output"][0]+=1
        r["digest"]=hashlib.sha256(json.dumps(r["receipt"],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.assertRaises(InferenceRefused):verify_sparse_receipt(self.index,self.hidden,r)

    def test_wrong_shapes_or_missing_selected_weights_refused(self):
        self.manifest["tensors"].pop("experts/1/up")
        (self.root/"weights.json").write_text(json.dumps(self.manifest))
        index=SparseWeightIndex.open(self.root/"weights.json")
        with self.assertRaises(InferenceRefused):sparse_forward(index,self.hidden,request_id="q")

    def test_actual_safetensors_bf16_tensor_is_loaded_and_verified(self):
        import torch
        from safetensors.torch import save_file
        current=self.manifest['tensors']['experts/0/gate']
        old_path=self.root/current['file']
        new_path=self.root/'weights.safetensors'
        input_tensor=torch.tensor(self.tensors['experts/0/gate'],dtype=torch.bfloat16)
        save_file({'model.layers.0.mlp.experts.0.gate_proj.weight':input_tensor},str(new_path))
        current['file']=new_path.name
        current['tensor_key']='model.layers.0.mlp.experts.0.gate_proj.weight'
        current['sha256']=hashlib.sha256(new_path.read_bytes()).hexdigest()
        current['dtype']='bfloat16'
        (self.root/'weights.json').write_text(json.dumps(self.manifest))
        index=SparseWeightIndex.open(self.root/'weights.json')
        output=index.load('experts/0/gate')
        self.assertEqual(output.shape,(5,6))
        self.assertTrue(np.array_equal(output,input_tensor.float().numpy()))
        observation=sparse_forward(index,self.hidden,request_id='bf16-neural')
        self.assertEqual(observation['receipt']['source_kind'],'synthetic')
        self.assertTrue(observation['receipt']['actual_neural_math_executed'])
        old_path.unlink()

    def test_claimed_235b_ground_rejects_small_fake_weights(self):
        self.manifest['model_ground']=ModelGround.published('qwen3-235b-a22b').identifier
        (self.root/'weights.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(InferenceRefused):SparseWeightIndex.open(self.root/'weights.json')

    def test_wrong_input_fails(self):
        with self.assertRaises(InferenceRefused):sparse_forward(self.index,[1,2,3],request_id="q")
        with self.assertRaises(InferenceRefused):sparse_forward(self.index,[float('nan')]*6,request_id="q")

    def test_128_experts_no_dense_materialization(self):
        root=self.root/"large";root.mkdir()
        write_index(root,hidden=12,experts=128,top_k=8)
        idx=SparseWeightIndex.open(root/"weights.json")
        self.assertEqual(len(idx.record["tensors"]),25)
        out=sparse_forward(idx,[1.]*12,request_id="128-experts")
        self.assertEqual(out["receipt"]["selected_experts"],list(range(8)))
        self.assertEqual(len(out["receipt"]["loaded_tensor_digests"]),25)

    def test_235b_model_can_be_described_without_allocating_weights(self):
        model=ModelGround.published("qwen3-235b-a22b")
        self.assertEqual(model.c["num_hidden_layers"],94)
        self.assertEqual(model.c["num_experts"],128)
        self.assertEqual(model.c["moe_intermediate_size"],1536)
        self.assertEqual(model.tensor_shape(93,"gate_proj.weight",expert=127),(1536,4096))

    def test_text_query_schema_rejects_bad_model_or_unbounded_generation(self):
        sample={"request_id":"req","model_ground":"a"*64,"prompt":"hello","max_new_tokens":2}
        self.assertEqual(validate_query(sample,"a"*64),sample)
        with self.assertRaises(QueryRefused):validate_query(sample,"b"*64)
        with self.assertRaises(QueryRefused):validate_query({**sample,"max_new_tokens":1_000_000},"a"*64)
        with self.assertRaises(QueryRefused):validate_query({**sample,"extra":True},"a"*64)


if __name__=="__main__":unittest.main()
