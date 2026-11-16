# E-CPU Virtual Processor

**An externally taskable virtual processor with a stable logical identity, an admitted computational Ground, verifiable observations, and persistent history across disposable hosts.**

The processor is not the GitHub runner. The runner is one available observer-host that can restore the processor, execute a Ground-native operation through the actual [E-CPU](https://github.com/Kier73/E-CPU) implementation, independently check its output, retain a receipt, and disappear. A later observer can recover the same logical processor.

> **VP-001 is operationally narrow by design.** It is a real, runnable E-CPU affine-orbit navigator with durable, hash-chained interaction history, **not** a general-purpose hosted VM, a perpetually running process, a physical-path compute benchmark, or a quorum-backed production authority. Expand the realization methods after establishing independently verifiable continuity.

## Interact now via GitHub Actions

1. Open **[Actions → Interact with E-CPU Virtual Processor](https://github.com/Kier73/E-CPU-Virtual-Processor/actions/workflows/processor.yml)**.
2. Select **Run workflow**, choose `observe`, `peek`, `navigate`, or `advance`, and enter an integer only when that operation requests one.
3. Open the resulting run's **Summary**. The observation and exact transition receipt are posted there and retained in a downloadable artifact.
4. Inspect the persisted [processor-state branch](https://github.com/Kier73/E-CPU-Virtual-Processor/tree/processor-state/environments) to examine its portable journal. A later Actions run reconstructs it from that branch before executing a new intention.

Available operations:

| Operation | Input | Result | Persistent cursor |
|---|---|---|---|
| `observe` | none | Observe the current Ground coordinate | Unchanged |
| `peek` | `coordinate` | Observe any admissible coordinate | Unchanged |
| `navigate` | `coordinate` | Resolve the target coordinate via native E-CPU | Moved to target |
| `advance` | `distance` | Navigate forward without replaying intermediate states | Advanced |

A request ID (the originating GitHub run ID) is stable across retries. Replaying an identical request returns its original receipt without repeating the transition. Reusing a request ID with a changed intention is refused. All commands, including reads, append an observation receipt to the journal.

The admitted Ground is an exact synthetic affine recurrence over integers mod 2³². It is *not* an inferred physical law. A coordinate can be as large as 2⁶³−1; native E-CPU `lcg_navigate` and `affine_orbit_jump` independently derive the result using logarithmic-time orbit jumps. They must agree or the operation is refused.

## Local executable example

Requires Python 3.11+ and the **actual E-CPU package** at the pinned revision:

```sh
python -m pip install 'git+https://github.com/Kier73/E-CPU.git@6fd416facdb2aefd6fc7d66a06b014e3d31b5a52'
python -m vp.cli bootstrap --journal local-state/primary.jsonl
python -m vp.cli run --journal local-state/primary.jsonl --request-id example-1 --actor operator --operation navigate --coordinate 1000000000000
python -m vp.cli run --journal local-state/primary.jsonl --request-id example-2 --actor operator --operation observe
python -m vp.cli verify --journal local-state/primary.jsonl
```

Delete the process, begin a fresh process, and run the final `verify` command again. No running executor is needed to retain the Ground or navigate its stored history. The local CLI is an *adapter*, not an authentication boundary: use a trusted caller or protected deployment.

Run the tests:

```sh
python -m unittest discover -s tests -v
```

The protocol tests can be executed without E-CPU by injecting an explicitly labeled test-only reference realizer. The native integration test only runs if the actual `mailcpu` package is installed; the hosted verification workflow installs the pinned version. The live hosted result must be inspected before claiming cross-platform or CI success.

## Architecture and evidence

```text
Authorized GitHub Actions workflow_dispatch
                  |
                  v
         Admitted intention
                  |
                  v
       Restore verified journal <-------- Git processor-state branch
                  |
                  v
      E-CPU native orbit navigation
        |                     |
   lcg_navigate       affine_orbit_jump
        \_____________________/
                  |
       exact value agreement
                  |
          durable receipt
                  |
    atomic local write + Git push
                  |
         verified new head
```

E-CPU retains the native computation. This repository owns the **addressable processor interface, admissible commands, evidence and persistence adapter**. It does not reproduce E-CPU's Hypervisor, StructuralMemoryRegistry, Virtual Layer or recovery-lineage quorum. Those existing systems are the expansion path.

The journal stores canonical JSON events linked by SHA-256. Every read rechecks the entire chain, its strict schema, and **the actual native E-CPU results**. A process does not trust a stored digest as proof of mathematical correctness. A complete rehash by a malicious actor can still rewrite unauthenticated history: Git access controls protect the branch, and later deployments should anchor history with E-CPU recovery-lineage authorities in independent fault domains. The Git fast-forward push and Actions concurrency group serialize the current single-owner implementation; they do not constitute multi-host distributed consensus.

[VP-001 specification and advancement plan](docs/VP-001.md)

## Vision

We seek a processor whose identity is constituted by a coherent, independently verifiable computational state-space—not by an always-running operating-system process. Observation can produce a computational result, and a Ground can make distant coordinates directly navigable without enumerating intermediate states. An appropriate host realizes an admitted transformation, verifies it, and returns the processor to its addressable, recoverable state.

This first system demonstrates that **a computation can remain addressable across the destruction and recreation of its execution host**. Later milestones can connect authenticated E-CPU model hosting, the Virtual Layer, structural-memory Grounds, multi-hop paths, independent recovery authorities and the [Virtual-Ground-Interpreter](https://github.com/Kier73/Virtual-Ground-Interpreter)—without resetting the processor's identity every time its physical implementation changes.
