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

## VP-003: Queryable neural inference (native and hosted)

The virtual processor now has a **query API**, beyond the VP-001 affine cursor and VP-002 large-MoE topology. It supports two explicitly distinguished realizations:

- **Native sparse neural math:** execute a receiver-owned, parameter-bearing SwiGLU expert sublayer through E-CPU's existing Virtual Layer and Hypervisor. Verified local NumPy/Safetensors weights, router matrix multiplication, selected-expert aggregation, and an independent scalar numerical oracle produce operation receipts. Missing/altered tensors and forged outputs are refused.
- **Full text generation through an E-CPU-governed host:** submit a prompt to a provisioned Transformers/PyTorch language model via an E-CPU operator. The model's actual tokenizer and generation backend perform autoregressive inference. It is *not* a claim that those entire computations have been translated to E-CPU's native SIMT tensor algebra.

Run a **local synthetic 128-expert** numerical experiment with `python examples/provision_sparse_demo.py`, then `python -m vp.sparse_query --index local-moe-demo/weights.json --hidden local-moe-demo/hidden.json --request-id first` after provisioning E-CPU. Query a **real pretrained text model** through `python -m vp.query --model-id <model> --revision <40-char-sha> --prompt 'Hello' --request-id first` with separately provisioned weights and the optional Transformers/Accelerate dependencies. For a warm HTTP endpoint, run `python -m vp.query_server` with explicit model/revision and an API token. See [VP-003 executable usage and evidence contract](docs/VP-003.md).

To bind an actual **Qwen3-30B/235B Safetensors snapshot** to E-CPU's sparse operator, run `python -m vp.qwen_snapshot` with the published model profile, a real hidden-vector observation, and a locally provisioned source snapshot. It reads the genuine router, selects only the required experts and verifies their weight-shard identities before execution. See the [VP-003 Qwen3 source adapter](docs/VP-003.md#use-an-actual-qwen3-local-checkpoint-without-materializing-all-experts).

The interfaces do **not** hard-code a total model parameter count. Resource feasibility, accessible weights, numerical operator support, and authenticated tensor placement still determine whether a particular model can actually execute. **235B full inference, persistent KV recovery, and distributed E-CPU-native Transformer execution remain unverified development objectives.**

## VP-004: Ground-navigated native neural queries (draft)

Rather than introducing another computation interpreter, VP-004 **connects existing E-CPU components in one receiver**: a declared token-coordinate Ground is admitted to E-CPU Structural Memory, observed at a requested logical coordinate through `VerifiedGroundRealizer` and the receiver Hypervisor, and committed as the prerequisite of a real, parameter-bearing sparse neural operator in an E-CPU `NavigableComputationDAG`. That operator runs through the **same receiver's** `VirtualLayerRuntime` and Hypervisor, and its numerical result is independently checked against the admitted tensor bytes before the DAG completes.

```sh
python examples/provision_sparse_demo.py --output vp004-demo
python -m vp.ground_neural --index vp004-demo/weights.json \
  --hidden vp004-demo/hidden.json --token-coordinate 1000000000013 \
  --request-id my-neural-observation
```

The example uses **synthetic expert weights**. The numerical execution is real, but it is a sparse MoE **sublayer**, not yet an entire Qwen3 Transformer or an autoregressive model. The token-position Ground is a declared, exact arithmetic law and does not manufacture pretrained hidden activations. For real model weights, use VP-003's pinned Qwen3 Safetensors source adapter and supply the correct preceding-layer hidden activation. Read [VP-004](docs/VP-004.md) for verification, permission-refusal tests, and the roadmap to full native attention and recoverable generation. This PR is stacked on the pending VP-002/003 drafts; native hosted CI has not yet completed.

## VP-005: navigable expert Grounds across disposable observer-hosts (draft)

VP-005 tests **Ground-native selection instead of expert-count iteration** for a *deliberately synthetic, law-defined neural model*. An admitted affine orbit addresses the routing center at distant logical token coordinates, and a proved nearest-center relation selects exactly the eight needed experts without reading or evaluating the remaining expert space. A compact Ground can describe `2^48` expert identities (over `10^16` logical weights for the chosen dimensions) while a query performs only eight actual gated-neural expert realizations. An independent arithmetic oracle checks every output coordinate. This is **not** an actual 235B trained checkpoint or a replacement for Qwen3 routing semantics.

The [VP-005 three-observer GitHub workflow](.github/workflows/vp005-navigable-experts.yml) separates typed intention, an actual receiver-local E-CPU Hypervisor/Virtual Layer realization, and an independent verification job. GitHub artifact transfer is a real communication path; no passive channel or GitHub storage service is falsely credited with performing the neural arithmetic. The receiver recomputes the source-defined Ground rather than trusting authority claimed in the incoming request. See [VP-005's model and evidence contracts](docs/VP-005.md) and `vp/navigable_experts.py`. This work is stacked on VP-004 and remains a draft until native hosted tests pass.

## Vision

We seek a processor whose identity is constituted by a coherent, independently verifiable computational state-space—not by an always-running operating-system process. Observation can produce a computational result, and a Ground can make distant coordinates directly navigable without enumerating intermediate states. An appropriate host realizes an admitted transformation, verifies it, and returns the processor to its addressable, recoverable state.

This first system demonstrates that **a computation can remain addressable across the destruction and recreation of its execution host**. Later milestones can connect authenticated E-CPU model hosting, the Virtual Layer, structural-memory Grounds, multi-hop paths, independent recovery authorities and the [Virtual-Ground-Interpreter](https://github.com/Kier73/Virtual-Ground-Interpreter)—without resetting the processor's identity every time its physical implementation changes.
