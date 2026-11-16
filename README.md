# E-CPU Virtual Processor

**An externally taskable virtual processor with a stable logical identity, an admitted computational Ground, verifiable observations, and persistent history across disposable hosts.**

The processor is not the GitHub runner. The runner is one available observer-host that can restore the processor, execute a Ground-native operation through the actual [E-CPU](https://github.com/Kier73/E-CPU) implementation, independently check its output, retain a receipt, and disappear. A later observer can recover the same logical processor.

> **VP-001 is operationally narrow by design.** It is a real, runnable E-CPU affine-orbit navigator with durable, hash-chained interaction history, **not** a general-purpose hosted VM, a perpetually running process, a physical-path compute benchmark, or a quorum-backed production authority. Expand the realization methods after establishing independently verifiable continuity.

## Computational structures

In modern computing, a **computational structure** is an organized system of representations, states, relationships, and transformation rules that defines possible computational behavior. Examples include state machines, data structures, algorithms, circuits, memory hierarchies, computational graphs, and neural networks. These structures can be mathematical descriptions, executable arrangements, or physical implementations—and often combine several of these forms.

Three distinctions are essential:

- **Structure and data:** a structure organizes information; particular values or parameters instantiate it, and may themselves have structure.
- **Structure and execution:** a rule can define possible transitions without requiring every state to be stored, visited, or executed in sequence.
- **Structure and realization:** equivalent computations may be realized through different processors, software interpreters, or other qualified mechanisms.

A computational structure describes **what is representable and transformable**; an execution or observation obtains a particular result. The virtual processor starts with one exact affine-modular Ground, then aims to admit and navigate additional verified computational structures through E-CPU's existing capabilities—not to confuse the structure with any one host that realizes it.

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

## VP-002: large Mixture-of-Experts computational structures (in development)

The next research target is **sparse, multi-billion-parameter model computation**, not another GPT-2 graph example. VP-002 introduces pinned Qwen3 MoE architectural Grounds for **30B-A3B** (48 layers, 6,144 expert instances) and **235B-A22B** (94 layers, 12,032 expert instances). The module can lazily address their tensors, compute bounded memory budgets, reconstruct **top-8 expert routing** from supplied logits, and produce a **25-operation, verified layer-local execution frontier** without materializing all experts.

This is an **architectural control-plane milestone**, not a claim that a pretrained model has generated tokens. No model weights are loaded, no model router logits are computed, and no neural numerical operators have yet executed. E-CPU's existing tensor operators, typed graph, Virtual Layer and authority/recovery systems are the intended execution substrate.

Run `python -m vp.moe describe --model qwen3-235b-a22b` or use the [large MoE reference Action](.github/workflows/moe-plan.yml). Read the [VP-002 specification, verification contract and exact reproduction commands](docs/VP-002.md).

## Vision

We seek a processor whose identity is constituted by a coherent, independently verifiable computational state-space—not by an always-running operating-system process. Observation can produce a computational result, and a Ground can make distant coordinates directly navigable without enumerating intermediate states. An appropriate host realizes an admitted transformation, verifies it, and returns the processor to its addressable, recoverable state.

This first system demonstrates that **a computation can remain addressable across the destruction and recreation of its execution host**. Later milestones can connect authenticated E-CPU model hosting, the Virtual Layer, structural-memory Grounds, multi-hop paths, independent recovery authorities and the [Virtual-Ground-Interpreter](https://github.com/Kier73/Virtual-Ground-Interpreter)—without resetting the processor's identity every time its physical implementation changes.
