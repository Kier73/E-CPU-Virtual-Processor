"""Small, explicit HTTP query surface for a provisioned E-CPU model host.

Run behind a TLS/auth reverse proxy for deployment, or localhost for testing.
The server does not accept code, model identities, file paths, or arbitrary
operator names from clients. It serializes requests against one E-CPU host.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import hmac
import json
import os

from .query import QueryService,QueryRefused,TransformersProvider


def serve(service:QueryService,*,host:str="127.0.0.1",port:int=8765,token:str|None=None)->None:
    if not token and host not in ("127.0.0.1","::1","localhost"):
        raise QueryRefused("external bind requires an API token and an external TLS reverse proxy")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):
            # Do not log prompts or authorization headers.
            pass
        def _send(self,status:int,body:dict):
            data=json.dumps(body,sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.end_headers();self.wfile.write(data)
        def _auth(self):
            return not token or hmac.compare_digest(self.headers.get("Authorization",""),"Bearer "+token)
        def do_GET(self):
            if not self._auth():return self._send(401,{"error":"unauthorized"})
            if self.path!="/health":return self._send(404,{"error":"unknown path"})
            return self._send(200,{"status":"ready","model_ground":service.provider.identity})
        def do_POST(self):
            if not self._auth():return self._send(401,{"error":"unauthorized"})
            if self.path!="/query":return self._send(404,{"error":"unknown path"})
            try:
                size=int(self.headers.get("Content-Length", "0"))
                if size<=0 or size>100_000:raise QueryRefused("request body size refused")
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict) or set(payload)!={"prompt","request_id","max_new_tokens"}:
                    raise QueryRefused("unexpected query schema")
                out=service.ask(payload["prompt"],request_id=payload["request_id"],
                                max_new_tokens=payload["max_new_tokens"])
                return self._send(200 if out.get("accepted") else 403,out)
            except (QueryRefused,ValueError,TypeError,KeyError) as exc:
                return self._send(400,{"error":str(exc)})
            except Exception:
                # Never expose model internals, file paths or user prompt in errors.
                return self._send(500,{"error":"model execution failed"})
    HTTPServer((host,port),Handler).serve_forever()


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description="Serve E-CPU model queries on a trusted interface")
    p.add_argument("--model-id",required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--token-env",default="ECPU_VP_API_TOKEN")
    p.add_argument("--cache-dir")
    p.add_argument("--allow-download",action="store_true")
    args=p.parse_args(argv)
    token=os.getenv(args.token_env)
    if args.host not in ("127.0.0.1","::1","localhost") and (not token or len(token)<32):
        raise QueryRefused("nonlocal service requires a strong API token")
    model=TransformersProvider(args.model_id,args.revision,allow_download=args.allow_download,cache_dir=args.cache_dir)
    serve(QueryService(model),host=args.host,port=args.port,token=token)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
