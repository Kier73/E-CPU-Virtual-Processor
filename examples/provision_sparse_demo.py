"""Create explicit synthetic 128-expert weights for a native E-CPU query demo.

Never claim these randomly initialized tensors are a pretrained LLM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('local-moe-demo'))
    p.add_argument('--hidden-size',type=int,default=16)
    p.add_argument('--num-experts',type=int,default=128)
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--intermediate-size',type=int,default=32)
    args=p.parse_args()
    if not 0<args.top_k<=args.num_experts or not 0<args.hidden_size or not 0<args.intermediate_size:
        raise ValueError('positive dimensions and 1 <= top_k <= experts are required')
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261115)
    model_ground=hashlib.sha256(json.dumps({'kind':'procedural-demo','hidden':args.hidden_size,
             'experts':args.num_experts,'top_k':args.top_k,'intermediate':args.intermediate_size},
             sort_keys=True).encode()).hexdigest()
    m={'schema':'ecpu-vp.sparse-weights.v1','model_ground':model_ground,
       'layer':0,'hidden_size':args.hidden_size,'num_experts':args.num_experts,
       'top_k':args.top_k,'provenance':{'kind':'synthetic','seed':20261115},'tensors':{}}
    def put(name,values):
        path=f'tensor-{len(m["tensors"]):03d}.npy'
        np.save(root/path,values.astype('float32'))
        b=(root/path).read_bytes()
        m['tensors'][name]={'file':path,'sha256':hashlib.sha256(b).hexdigest(),
                            'dtype':'float32','shape':list(values.shape)}
    router=np.zeros((args.num_experts,args.hidden_size),dtype=np.float32)
    router[:,0]=np.arange(args.num_experts,0,-1,dtype=np.float32)/100
    put('router',router)
    for i in range(args.top_k):
        put(f'experts/{i}/gate',rng.standard_normal((args.intermediate_size,args.hidden_size))*.02)
        put(f'experts/{i}/up',rng.standard_normal((args.intermediate_size,args.hidden_size))*.02)
        put(f'experts/{i}/down',rng.standard_normal((args.hidden_size,args.intermediate_size))*.02)
    (root/'weights.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    (root/'hidden.json').write_text(json.dumps([1.]*args.hidden_size)+'\n')
    digest=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    print(json.dumps({'kind':'synthetic; not a pretrained LLM','model_ground':model_ground,
                      'weight_index_digest':digest,'index':str(root/'weights.json'),
                      'hidden':str(root/'hidden.json')},indent=2))

if __name__=='__main__':main()
