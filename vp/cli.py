"""Restricted, non-eval CLI for a persistent E-CPU observer-host."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import ProcessorRefused, bootstrap, load, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecpu-vp", description="Persist and navigate an E-CPU virtual processor")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("bootstrap", help="Create genesis, or verify the existing identity")
    start.add_argument("--journal", type=Path, required=True)
    start.add_argument("--processor-id", default="ecpu-vp-primary")
    status = commands.add_parser("status", help="Verify all receipts and show current Ground coordinate")
    status.add_argument("--journal", type=Path, required=True)
    verify = commands.add_parser("verify", help="Independently reconstruct full journal")
    verify.add_argument("--journal", type=Path, required=True)
    operation = commands.add_parser("run", help="Submit one idempotent observation or navigation")
    operation.add_argument("--journal", type=Path, required=True)
    operation.add_argument("--request-id", required=True)
    operation.add_argument("--actor", required=True)
    operation.add_argument("--operation", choices=["observe", "peek", "navigate", "advance"], required=True)
    operation.add_argument("--coordinate", type=int)
    operation.add_argument("--distance", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            result = {"bootstrapped_or_verified": True, "current": bootstrap(args.journal, processor_id=args.processor_id).public()}
        elif args.command in {"status", "verify"}:
            events, state = load(args.journal)
            result = {"verified": True, "event_count": len(events), "current": state.public()}
        else:
            intent = {"request_id": args.request_id, "actor": args.actor, "operation": args.operation}
            if args.operation in ("peek", "navigate"):
                if args.coordinate is None or args.distance is not None:
                    raise ProcessorRefused("peek/navigate requires --coordinate and prohibits --distance")
                intent["target"] = args.coordinate
            elif args.operation == "advance":
                if args.coordinate is not None or args.distance is None:
                    raise ProcessorRefused("advance requires --distance and prohibits --coordinate")
                intent["distance"] = args.distance
            elif args.coordinate is not None or args.distance is not None:
                raise ProcessorRefused("observe accepts no coordinate or distance")
            result = run(args.journal, intent)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ProcessorRefused, FileNotFoundError, OSError, UnicodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
