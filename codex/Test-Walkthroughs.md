# DeepEP Test Walkthroughs

## Contents

- [Conventions](#conventions)
- [Intranode (NVLink) Walkthrough](#intranode-nvlink-walkthrough)
- [Internode (RDMA + NVLink) Walkthrough](#internode-rdma--nvlink-walkthrough)
- [Low‑Latency (IBGDA‑only) Walkthrough](#lowlatency-ibgdaonly-walkthrough)
- [End‑to‑End Call Stack Examples](#endtoend-call-stack-examples)
- [Notes on Correctness Checks](#notes-on-correctness-checks)

This guide traces the full call paths used by the tests, from Python APIs through bindings to C++ and CUDA kernels, with annotated links to the relevant code.


## Conventions

- File links point to repository‑relative paths with a `#Lline` anchor for VS Code navigation.
- Python API surfaces are in `deep_ep/buffer.py`; C++ runtime and bindings are in `csrc/deep_ep.cpp`; kernels live under `csrc/kernels/`.


## Intranode (NVLink) Walkthrough

Entry points:

- Test driver: [tests/test_intranode.py:297](tests/test_intranode.py#L297)
- Main test flow: [tests/test_intranode.py:15](tests/test_intranode.py#L15)
- Process group init: [tests/utils.py:14](tests/utils.py#L14)

High‑level flow:

1) Setup and Buffer construction
   - Initialize NCCL group and CUDA device — [tests/utils.py:14](tests/utils.py#L14)
   - Construct `deep_ep.Buffer(group, num_nvl_bytes=..., num_rdma_bytes=0, ...)` — [tests/test_intranode.py:281](tests/test_intranode.py#L281)
     - Python constructor sets env knobs, gathers device IDs + CUDA IPC handles, optionally gets NVSHMEM IDs, and calls C++ `Buffer::sync(...)` — [deep_ep/buffer.py:32](deep_ep/buffer.py#L32) → [csrc/deep_ep.cpp:211](csrc/deep_ep.cpp#L211)

2) Layout computation
   - Compute `topk_idx` and expert metadata in Python — [tests/test_intranode.py:26](tests/test_intranode.py#L26)
   - Call `num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)` — [tests/test_intranode.py:62](tests/test_intranode.py#L62)
     - Python → C++ `Buffer::get_dispatch_layout(...)` — [deep_ep/buffer.py:291](deep_ep/buffer.py#L291) → [csrc/deep_ep.cpp:278](csrc/deep_ep.cpp#L278)
     - CUDA kernel builds per‑rank/expert counts and mask — [csrc/kernels/layout.cu:8](csrc/kernels/layout.cu#L8)

3) Dispatch
   - API: `recv_x, recv_topk_idx, recv_topk_weights, counts, handle, ev = buffer.dispatch(...)` — [tests/test_intranode.py:112](tests/test_intranode.py#L112)
   - Intranode selected (no RDMA ranks): [deep_ep/buffer.py:320](deep_ep/buffer.py#L320) → C++ intranode path
   - Notify/prefix: [csrc/kernels/intranode.cu:11](csrc/kernels/intranode.cu#L11) prepares per‑rank + per‑channel prefix matrices in receiver‑owned buffers
   - Data movement: [csrc/kernels/intranode.cu:197](csrc/kernels/intranode.cu#L197) copies BF16/FP8 shards into per‑channel ring buffers; receivers advance heads and copy into the final output tensor
   - A reusable `handle` containing prefix matrices and queue heads is returned for cached dispatches — [deep_ep/buffer.py:320](deep_ep/buffer.py#L320)

4) Combine
   - API: `combined_x, combined_topk_weights, ev = buffer.combine(x=recv_x, handle=handle, ...)` — [tests/test_intranode.py:173](tests/test_intranode.py#L173)
   - Intranode `combine(...)` reduces shards with optional weights/bias — [deep_ep/buffer.py:404](deep_ep/buffer.py#L404) → [csrc/kernels/api.cuh:186](csrc/kernels/api.cuh#L186) (impl in `intranode.cu`)
   - Correctness is validated by comparing to references — [tests/test_intranode.py:142](tests/test_intranode.py#L142)

5) Cached dispatch/combine and tuning
   - Cached dispatch using the prior `handle` — [tests/test_intranode.py:161](tests/test_intranode.py#L161)
   - Config tuning over SM/channel depths — [tests/test_intranode.py:208](tests/test_intranode.py#L208) and [tests/test_intranode.py:249](tests/test_intranode.py#L249)


## Internode (RDMA + NVLink) Walkthrough

Entry points:

- Test driver: [tests/test_internode.py:353](tests/test_internode.py#L353)
- Main test flow: [tests/test_internode.py:16](tests/test_internode.py#L16)
- Process group init: [tests/utils.py:14](tests/utils.py#L14)

High‑level flow:

1) Setup and Buffer construction
   - Multi‑node world (`WORLD_SIZE>1`); construct `deep_ep.Buffer(group, num_nvl_bytes=..., num_rdma_bytes=..., ...)` — [tests/test_internode.py:307](tests/test_internode.py#L307)
   - C++ `Buffer::sync(...)` initializes NVSHMEM, allocates RDMA buffers, and creates per‑GPU‑index teams if low‑latency is enabled — [csrc/deep_ep.cpp:211](csrc/deep_ep.cpp#L211), [csrc/kernels/runtime.cu:49](csrc/kernels/runtime.cu#L49)

2) Layout computation
   - Grouped top‑k selection to limit inter‑node traffic — [tests/test_internode.py:26](tests/test_internode.py#L26)
   - `buffer.get_dispatch_layout(...)` also returns `num_tokens_per_rdma_rank` — [tests/test_internode.py:64](tests/test_internode.py#L64) → [csrc/deep_ep.cpp:278](csrc/deep_ep.cpp#L278)

3) Dispatch
   - API: `recv_x, recv_topk_idx, recv_topk_weights, counts, handle, ev = buffer.dispatch(...)` — [tests/test_internode.py:108](tests/test_internode.py#L108)
   - Internode selected: [deep_ep/buffer.py:320](deep_ep/buffer.py#L320) routes to `internode_dispatch` — [deep_ep/buffer.py:460](deep_ep/buffer.py#L460) → [csrc/deep_ep.cpp:813](csrc/deep_ep.cpp#L813)
   - C++ path releases the GIL to avoid blocking Python while GPU/CPU wait — [csrc/deep_ep.cpp:836](csrc/deep_ep.cpp#L836)
   - Notify + clean + prefix (RDMA + NVL tiers): [csrc/kernels/internode.cu:49](csrc/kernels/internode.cu#L49) and [csrc/kernels/api.cuh:233](csrc/kernels/api.cuh#L233)
   - Data movement: RDMA tier enqueues NVSHMEM IBGDA puts into decoupled per‑channel buffers, then NVL tier fans in to destination ranks — [csrc/kernels/api.cuh:206](csrc/kernels/api.cuh#L206), [csrc/kernels/internode.cu](csrc/kernels/internode.cu)

4) Combine
   - API: `combined_x, combined_topk_weights, ev = buffer.combine(x=recv_x, handle=handle, bias=(b0,b1), ...)` — [tests/test_internode.py:143](tests/test_internode.py#L143)
   - Cached notify + reduce across both tiers — [csrc/kernels/api.cuh:294](csrc/kernels/api.cuh#L294), [csrc/kernels/api.cuh:316](csrc/kernels/api.cuh#L316)
   - Correctness checked against references — [tests/test_internode.py:161](tests/test_internode.py#L161)

5) Tuning and profiling
   - Dispatch tuning over NVL/RDMA chunk sizes with Kineto timing — [tests/test_internode.py:190](tests/test_internode.py#L190)
   - Combine tuning — [tests/test_internode.py:260](tests/test_internode.py#L260)


## Low‑Latency (IBGDA‑only) Walkthrough

Entry points:

- Test driver: [tests/test_low_latency.py:313](tests/test_low_latency.py#L313)
- Main test flow: [tests/test_low_latency.py:37](tests/test_low_latency.py#L37)

High‑level flow:

1) Setup and Buffer construction
   - Compute RDMA size hint and construct `deep_ep.Buffer(..., low_latency_mode=True, num_qps_per_rank=num_local_experts, ...)` — [tests/test_low_latency.py:267](tests/test_low_latency.py#L267)
   - C++ allocates a symmetric odd/even buffer layout (`LowLatencyLayout`) in NVSHMEM memory — [csrc/config.hpp:127](csrc/config.hpp#L127), [csrc/deep_ep.cpp](csrc/deep_ep.cpp) (low‑latency methods)

2) Dispatch
   - `packed_recv_x, recv_count, handle, ev, hook = buffer.low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, ...)` — [tests/test_low_latency.py:98](tests/test_low_latency.py#L98)
   - Optionally casts to FP8 (per‑token/channel amax) and returns a “recv hook” to integrate with compute pipelines — [deep_ep/buffer.py:547](deep_ep/buffer.py#L547)
   - Device barrier and masked rank handling embedded in kernels — [csrc/kernels/internode_ll.cu:22](csrc/kernels/internode_ll.cu#L22)

3) Combine
   - Option A (normal): `combined_x, ev, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle, ...)` — [tests/test_low_latency.py:156](tests/test_low_latency.py#L156)
   - Option B (zero‑copy): fill `buffer.get_next_low_latency_combine_buffer(handle)` then call `low_latency_combine(..., zero_copy=True)` — [tests/test_low_latency.py:154](tests/test_low_latency.py#L154)
   - Optional LogFMT‑10 receive format reduces bandwidth — [deep_ep/buffer.py:616](deep_ep/buffer.py#L616), [csrc/kernels/internode_ll.cu](csrc/kernels/internode_ll.cu) (combine)

4) Shrink/failure simulation
   - Tests mark ranks as failed on first API use and ensure mask buffer reflects failures — [tests/test_low_latency.py:12](tests/test_low_latency.py#L12), [tests/test_low_latency.py:184](tests/test_low_latency.py#L184)
   - Query/clean mask buffer APIs — [deep_ep/buffer.py:673](deep_ep/buffer.py#L673), [deep_ep/buffer.py:683](deep_ep/buffer.py#L683); C++ binds to internode_ll helpers — [csrc/deep_ep.cpp:1702](csrc/deep_ep.cpp#L1702)


## End‑to‑End Call Stack Examples

Intranode dispatch:

```
[tests/test_intranode.py:112](tests/test_intranode.py#L112)
  → deep_ep.Buffer.dispatch(...)               ([deep_ep/buffer.py:320](deep_ep/buffer.py#L320))
    → deep_ep_cpp.Buffer.intranode_dispatch    ([csrc/deep_ep.cpp:355](csrc/deep_ep.cpp#L355))
      → intranode::notify_dispatch             ([csrc/kernels/intranode.cu:11](csrc/kernels/intranode.cu#L11))
      → intranode::dispatch                    ([csrc/kernels/intranode.cu:197](csrc/kernels/intranode.cu#L197))
```

Internode combine:

```
[tests/test_internode.py:143](tests/test_internode.py#L143)
  → deep_ep.Buffer.combine(...)                ([deep_ep/buffer.py:404](deep_ep/buffer.py#L404))
    → deep_ep_cpp.Buffer.internode_combine     ([csrc/deep_ep.cpp:1187](csrc/deep_ep.cpp#L1187))
      → internode::cached_notify               ([csrc/kernels/api.cuh:294](csrc/kernels/api.cuh#L294))
      → internode::combine                     ([csrc/kernels/api.cuh:316](csrc/kernels/api.cuh#L316))
```

Low‑latency dispatch + combine:

```
[tests/test_low_latency.py:98](tests/test_low_latency.py#L98)
  → deep_ep.Buffer.low_latency_dispatch        ([deep_ep/buffer.py:547](deep_ep/buffer.py#L547))
    → internode_ll::dispatch                   ([csrc/kernels/internode_ll.cu:129](csrc/kernels/internode_ll.cu#L129))

[tests/test_low_latency.py:156](tests/test_low_latency.py#L156)
  → deep_ep.Buffer.low_latency_combine         ([deep_ep/buffer.py:616](deep_ep/buffer.py#L616))
    → internode_ll::combine                    ([csrc/kernels/internode_ll.cu](csrc/kernels/internode_ll.cu), after dispatch)
```


## Notes on Correctness Checks

- Tests compute reference results on input tensors (`x`, `topk_idx`, `topk_weights`) and compare reduced outputs element‑wise with tolerances depending on format (BF16 vs FP8/LogFMT) — e.g., [tests/test_internode.py:154](tests/test_internode.py#L154).
- Layout correctness is verified by recomputing counts in Python and comparing against `get_dispatch_layout` outputs — [tests/test_intranode.py:62](tests/test_intranode.py#L62) and [tests/test_internode.py:64](tests/test_internode.py#L64).
