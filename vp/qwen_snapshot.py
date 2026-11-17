"""Qwen3 MoE: bind real local Safetensors to VP-002 Ground and selected experts.

The adapter reads a pinned *local* model snapshot only. It never executes
remote model code, invents missing tensor bytes or downloads a checkpoint.
It reads router values, but only the metadata (shape/dtype) of expert tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .moe import ModelGround
from .neural_inference import InferenceRefused, _canonical, _sha


_DTYPES={"BF16":"bfloat16", "F16":"float16", "F32":"float32", "F64":"float64"}


def _tensor_name(layer:int,role:str,expert:int|None=None)->str:
    prefix=f"model.layers.{layer}.mlp."
    if role=="router":return prefix+"gate.weight"
    if expert is None or role not in ("gate","up","down"):
        raise InferenceRefused("invalid Qwen expert tensor address")
    return prefix+f"experts.{expert}.{role}_proj.weight"


def _read_index(root:Path)->dict[str,str]:
    path=root/"model.safetensors.index.json"
    if not path.is_file():
        raise InferenceRefused("local pretrained snapshot requires model.safetensors.index.json")
    try:
        index=json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        raise InferenceRefused("unreadable original Safetensors index") from exc
    weight_map=index.get("weight_map") if isinstance(index,dict) else None
    if not isinstance(weight_map,dict) or not weight_map:
        raise InferenceRefused("missing original Safetensors weight_map")
    return weight_map


def prepare_qwen_index(snapshot:Path,ground:ModelGround,*,layer:int,hidden:Sequence[float],
                       output:Path,expected_weight_map_sha256:str|None=None)->dict[str,Any]:
    """Select real expert tensors using a query's real router forward.

    The generated manifest has no fiction about a fully resident 235B model:
    it binds only the router and eight selected expert triples for one layer.
    E-CPU can then execute that sparse numerical operation.
    """
    import numpy as np
    from safetensors import safe_open
    snapshot=Path(snapshot).resolve(strict=True)
    output=Path(output).absolute()
    if output.parent.resolve()!=snapshot or output.suffix!='.json':
        raise InferenceRefused("derived weight index must be stored inside its snapshot directory")
    ground.validate()
    if type(layer) is not int or not 0<=layer<ground.c['num_hidden_layers']:
        raise InferenceRefused("layer outside the pinned Ground")
    if not isinstance(hidden,(list,tuple)) or len(hidden)!=ground.c['hidden_size'] or any(type(x) not in (int,float) or not math.isfinite(x) for x in hidden):
        raise InferenceRefused("hidden vector does not match the published model")
    original=snapshot/'model.safetensors.index.json'
    original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    if expected_weight_map_sha256 is not None and original_sha!=expected_weight_map_sha256:
        raise InferenceRefused("pretrained source index differs from externally pinned bytes")
    weight_map=_read_index(snapshot)
    cache={}
    def locate(name):
        file=weight_map.get(name)
        if type(file) is not str or '/' in file or '\\' in file or Path(file).name!=file or not file.endswith('.safetensors'):
            raise InferenceRefused('tensor absent or uses unsafe source filename: '+name)
        path=snapshot/file
        if not path.is_file():
            raise InferenceRefused('required local pretrained shard missing: '+file)
        return file,path
    def header(name,shape):
        file,path=locate(name)
        with safe_open(str(path),framework='pt',device='cpu') as reader:
            if name not in reader.keys():
                raise InferenceRefused('source weight_map does not match real shard tensor')
            entry=reader.get_slice(name)
            actual=list(entry.get_shape())
            dtype=entry.get_dtype()
            if actual!=list(shape) or dtype not in _DTYPES:
                raise InferenceRefused('source tensor shape/dtype contradicts Ground: '+name)
        if file not in cache:
            hasher=hashlib.sha256()
            with path.open('rb') as source:
                while part:=source.read(1024*1024):hasher.update(part)
            cache[file]=hasher.hexdigest()
        return {'file':file,'sha256':cache[file], 'shape':actual,
                'dtype':_DTYPES[dtype],'tensor_key':name}
    router_name=_tensor_name(layer,'router')
    router_ref=header(router_name,ground.tensor_shape(layer,'router.weight'))
    with safe_open(str(snapshot/router_ref['file']),framework='pt',device='cpu') as reader:
        table=reader.get_tensor(router_name).float().numpy().astype(np.float64)
    x=np.asarray(hidden,dtype=np.float64)
    logits=table @ x
    if not np.isfinite(logits).all():
        raise InferenceRefused("source router produced nonfinite scores")
    chosen=sorted(range(len(logits)),key=lambda i:(-float(logits[i]),i))[:ground.c['num_experts_per_tok']]
    tensors={'router':router_ref}
    for expert in chosen:
        for role in ('gate','up','down'):
            key=f'experts/{expert}/{role}'
            tensors[key]=header(_tensor_name(layer,role,expert),
                                ground.tensor_shape(layer,role+'_proj.weight',expert=expert))
    manifest={'schema':'ecpu-vp.sparse-weights.v1','model_ground':ground.identifier,
              'layer':layer,'hidden_size':ground.c['hidden_size'],
              'num_experts':ground.c['num_experts'],
              'top_k':ground.c['num_experts_per_tok'],
              'provenance':{'kind':'measured','source_model':ground.source_model,
                 'revision':ground.revision,'source_index_sha256':original_sha,
                 'hidden_sha256':hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest(),
                 'trusted_source_manifest':expected_weight_map_sha256 is not None},
              'tensors':tensors}
    # A derived index is a materialization recipe, not an authenticated source
    # identity unless its original model index digest was independently pinned.
    output.write_bytes(_canonical(manifest)+b'\n')
    return {'model_ground':ground.identifier,'index':str(output),
            'index_digest':_sha(_canonical(manifest)),'selected_experts':chosen,
            'physical_shards_hashed':len(cache),'tensor_payloads_executed':False}


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Bind local Qwen3 pretrained Safetensors to a real sparse E-CPU query')
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--model',choices=('qwen3-30b-a3b','qwen3-235b-a22b'),required=True)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--hidden',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-weight-map-sha256',help='Externally trusted source index digest')
    args=p.parse_args(argv)
    ground=ModelGround.published(args.model)
    vector=json.loads(args.hidden.read_text())
    output=prepare_qwen_index(args.snapshot,ground,layer=args.layer,hidden=vector,
               output=args.output,expected_weight_map_sha256=args.expected_weight_map_sha256)
    print(json.dumps(output,sort_keys=True,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
