"""Queryable model gateway with explicit native E-CPU authority and backend labels.

The actual text-generation backend is Hugging Face Transformers. The model is
not secretly replaced with a small reference or hosted API. E-CPU admits the
requested operation, controls the receiver's capability, and retains the
operator receipt. This is hosted numerical inference; E-CPU tensor-native MoE
partial inference is provided separately by neural_inference.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .native_inference import NativeInferenceHost


class QueryRefused(ValueError):
    pass


def _digest(v:Any)->str:
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_query(request:dict,expected_model_ground:str)->dict:
    if not isinstance(request,dict) or set(request)!={"request_id","model_ground","prompt","max_new_tokens"}:
        raise QueryRefused("unknown query fields")
    if request["model_ground"]!=expected_model_ground:
        raise QueryRefused("unadmitted model identity")
    if type(request["request_id"]) is not str or not 1<=len(request["request_id"])<=128:
        raise QueryRefused("invalid request ID")
    if type(request["prompt"]) is not str or not 1<=len(request["prompt"].encode('utf8'))<=65536:
        raise QueryRefused("prompt must be nonempty and bounded")
    if type(request["max_new_tokens"]) is not int or not 1<=request["max_new_tokens"]<=1024:
        raise QueryRefused("token budget must be between 1 and 1024")
    return request


class TransformersProvider:
    """Pretrained full-model forward adapter; available models depend on resources.

    Model revision is explicit, remote implementation code never trusted. A
    caller can opt into downloading a published checkpoint; default offline
    is safer for a persistent service with preprovisioned model weights.
    """
    def __init__(self, model_id:str, revision:str, *, allow_download:bool=False,
                 device_map:str="auto", cache_dir:str|None=None):
        if not isinstance(model_id,str) or not model_id or len(model_id)>250 or '..' in model_id:
            raise QueryRefused("invalid model identity")
        if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
            raise QueryRefused("pin remote models to an exact 40-character revision")
        if device_map not in ("auto","cpu"):
            raise QueryRefused("unsupported device map")
        self.model_id=model_id
        self.revision=revision
        self.identity=_digest({"kind":"ecpu-vp.transformers-model.v1","model":model_id,"revision":revision})
        try:
            import torch
            from transformers import AutoModelForCausalLM,AutoTokenizer
        except ImportError as exc:
            raise QueryRefused("install torch, transformers and accelerate for full-model inference") from exc
        kwargs={"revision":revision,"trust_remote_code":False,
                "local_files_only":not allow_download}
        if cache_dir is not None:
            kwargs["cache_dir"]=cache_dir
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,**kwargs)
        self.model=AutoModelForCausalLM.from_pretrained(model_id,
                    device_map=device_map,torch_dtype="auto",**kwargs).eval()
        self.torch=torch

    def infer(self,request:dict)->dict:
        validate_query(request,self.identity)
        torch=self.torch
        tokens=self.tokenizer(request["prompt"],return_tensors="pt")
        input_ids=tokens["input_ids"]
        input_len=input_ids.shape[-1]
        # With distributed device_map, place input on the embedding module's
        # execution device rather than assuming a single-device model.
        try:
            device=self.model.get_input_embeddings().weight.device
        except AttributeError:
            device=self.model.device
        tokens={k:v.to(device) for k,v in tokens.items()}
        with torch.inference_mode():
            output=self.model.generate(**tokens,do_sample=False,
                        max_new_tokens=request["max_new_tokens"],use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id)
        new_ids=output[0,input_len:].tolist()
        answer=self.tokenizer.decode(new_ids,skip_special_tokens=True)
        return {"kind":"ecpu-vp.query-result.v1","request_id":request["request_id"],
                "model_ground":self.identity,"provider":"transformers.native-pytorch",
                "prompt_sha256":hashlib.sha256(request["prompt"].encode()).hexdigest(),
                "generated_token_ids":new_ids,"generated_text":answer,
                "full_neural_forward_performed":True,
                "ecpu_tensor_native":False}


class QueryService:
    def __init__(self,provider:TransformersProvider, *, allow_compute:bool=True):
        self.provider=provider
        self.host=NativeInferenceHost(provider.infer,node_id="vp-query-host",
                   model_ground=provider.identity,operator_id="vp.nn.generate",
                   allow_compute=allow_compute,max_input_bytes=1_000_000,
                   max_output_bytes=1_000_000)
        self._sequence=0
        # Idempotency within one live receiver. Cross-restart replay requires the
        # independent E-CPU/IFS durable checkpoint authority (future VP milestone).
        self._seen: dict[str,tuple[str,dict]]={}

    def ask(self,prompt:str,*,request_id:str,max_new_tokens:int=32)->dict:
        request={"request_id":request_id,"model_ground":self.provider.identity,
                 "prompt":prompt,"max_new_tokens":max_new_tokens}
        validate_query(request,self.provider.identity)
        fingerprint=_digest(request)
        existing=self._seen.get(request_id)
        if existing is not None:
            if existing[0]!=fingerprint:
                raise QueryRefused("request ID reused with changed intention")
            return existing[1]
        self._sequence+=1
        result=self.host.execute(request,sequence=self._sequence)
        if result.get("accepted"):
            self._seen[request_id]=(fingerprint,result)
        return result


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Query a pinned, locally provisioned model through E-CPU")
    p.add_argument("--model-id",required=True)
    p.add_argument("--revision",required=True,help="Exact pretrained model revision")
    p.add_argument("--prompt",required=True)
    p.add_argument("--request-id",required=True)
    p.add_argument("--max-new-tokens",type=int,default=32)
    p.add_argument("--allow-download",action="store_true",help="Opt into a real remote model download")
    p.add_argument("--device-map",choices=["auto","cpu"],default="auto")
    p.add_argument("--cache-dir")
    args=p.parse_args(argv)
    provider=TransformersProvider(args.model_id,args.revision,
             allow_download=args.allow_download,device_map=args.device_map,cache_dir=args.cache_dir)
    result=QueryService(provider).ask(args.prompt,request_id=args.request_id,
                                    max_new_tokens=args.max_new_tokens)
    print(json.dumps(result,sort_keys=True,indent=2))
    if not result.get("accepted"):
        raise QueryRefused("native E-CPU inference admission refused")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
