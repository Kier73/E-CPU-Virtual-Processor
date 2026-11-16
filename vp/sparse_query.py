"""Queryable sparse expert operation through the real E-CPU Hypervisor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .neural_inference import SparseWeightIndex, sparse_forward
from .native_inference import NativeInferenceHost


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Run actual parameter-bearing sparse expert inference under E-CPU")
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--expected-index-digest",help="Trust anchor of the receiver's weight descriptor")
    p.add_argument("--hidden",type=Path,required=True,help="JSON array of floating point hidden state")
    p.add_argument("--request-id",required=True)
    args=p.parse_args(argv)
    index=SparseWeightIndex.open(args.index,expected_sha256=args.expected_index_digest)
    hidden=json.loads(args.hidden.read_text())
    host=NativeInferenceHost(lambda q: sparse_forward(index,q["hidden"],request_id=q["request_id"]),
         node_id="vp-sparse-expert",model_ground=index.record["model_ground"],
         operator_id="vp.nn.sparse_swiglu")
    answer=host.execute({"request_id":args.request_id,"model_ground":index.record["model_ground"],
                         "hidden":hidden})
    print(json.dumps(answer,indent=2,sort_keys=True))
    if not answer.get("accepted"):
        raise RuntimeError("E-CPU refused sparse expert operation")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
