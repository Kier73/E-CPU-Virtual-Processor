# VP-005 local navigation scale profile

**Collected 2026-11-16, local Python 3.13; native E-CPU not installed in this local test.** The same Ground selector (`top_k=8`, `hidden_dim=3`, `intermediate_dim=4`) was observed at logical token coordinate `72,057,594,037,928,009` for four declared expert counts. Each case performed 200 sequential plan → selected-expert exact evaluation → independently recomputed scalar-verifier operations, measured with `time.perf_counter_ns`. These samples are **not** a standardized cross-platform performance benchmark or a network/physical-medium compute claim.

| Declared experts | Logical parameter coordinates | Plan bytes | Observation bytes | Median verified local path |
|---:|---:|---:|---:|---:|
| 128 | 4,608 | 819 | 476 | 447.17 µs |
| 1,048,576 | 37,748,736 | 859 | 480 | 402.44 µs |
| 1,099,511,627,776 | 39,582,418,599,936 | 919 | 486 | 420.55 µs |
| 281,474,976,710,656 | 10,133,099,161,583,104 | 948 | 490 | 420.23 µs |

The final query's exact output was `["197642/101", "120044/101", "-187258/101"]`, independently verified from eight selected experts. For this **special synthetic law**, the number of expert arithmetic realizations is fixed at eight; the selector doesn't scan the complete expert list. Description and receipt lengths grow slowly with the number of digits needed to encode the model and selected expert indices.

This establishes an exact locally executed, finitely observed and independently checked property of a particular **procedural Ground**. Its declared 10.13-quadrillion parameter positions are **not 10.13-quadrillion individually trained or stored weight values**. There is no evidence here that an arbitrary pretrained Qwen checkpoint has the same generative parameter or routing structure, or that the GitHub transport path performed the neural multiplications. Such claims would require separate source and path-observation evidence. The workflow in `.github/workflows/vp005-navigable-experts.yml` is the separate native receiver and multi-job observer-host gate.

Reproduction:

```sh
python -m unittest discover -s tests -p 'test_navigable_experts.py' -v
python -m vp.navigable_experts plan --out plan.json
```

Local tests at creation: **7 passed, 2 skipped**; skipped tests explicitly require the E-CPU package. A successful GitHub workflow is required to claim native Hypervisor execution and independent inter-job observation.
