# DeepEP Architecture

## Contents

- [Map of the Codebase](#map-of-the-codebase)
- [High‑Level Data Flow](#highlevel-data-flow)
- [Python API Layer](#python-api-layer)
- [C++ Runtime and Initialization](#c-runtime-and-initialization)
- [Configuration and Buffer Sizing](#configuration-and-buffer-sizing)
- [Dispatch Layout Computation](#dispatch-layout-computation)
- [Intranode All‑to‑All (NVLink)](#intranode-alltoall-nvlink)
- [Internode All‑to‑All (RDMA + NVLink via NVSHMEM)](#internode-alltoall-rdma--nvlink-via-nvshmem)
- [Low‑Latency Path (IBGDA‑only)](#lowlatency-path-ibgdaonly)
- [Synchronization Primitives](#synchronization-primitives)
- [Performance Notes](#performance-notes)
- [Background Primer](#background-primer)
- [Where To Look Next](#where-to-look-next)

DeepEP is an expert-parallel (EP) communication library optimized for Mixture‑of‑Experts (MoE) workloads. It provides:

- High‑throughput intranode all‑to‑all over NVLink
- High‑throughput internode all‑to‑all over RDMA (NVSHMEM + NVLink)
- Low‑latency all‑to‑all using IBGDA RDMA

The public Python API lives in `deep_ep`, backed by a C++/CUDA extension `deep_ep_cpp` (pybind11), which in turn invokes CUDA kernels and (optionally) NVSHMEM device operations.


## Map of the Codebase

- Python package: `deep_ep/`
  - [deep_ep/buffer.py](deep_ep/buffer.py) — user‑facing Buffer and API orchestration
  - [deep_ep/utils.py](deep_ep/utils.py) — event wrapper + NVLink checks
  - [deep_ep/__init__.py](deep_ep/__init__.py) — re‑exports and dtype bindings
- C++/CUDA extension: `csrc/`
  - [csrc/deep_ep.cpp](csrc/deep_ep.cpp), [csrc/deep_ep.hpp](csrc/deep_ep.hpp) — runtime, pybind11 bindings, host orchestration
  - [csrc/config.hpp](csrc/config.hpp), [csrc/event.hpp](csrc/event.hpp) — configuration helpers, CUDA event utils
  - Kernels in `csrc/kernels/`: [layout.cu](csrc/kernels/layout.cu), [intranode.cu](csrc/kernels/intranode.cu), [internode.cu](csrc/kernels/internode.cu), [internode_ll.cu](csrc/kernels/internode_ll.cu), [runtime.cu](csrc/kernels/runtime.cu) (+ helpers)
- Build: `setup.py` (Torch CUDAExtension) selects sources and NVSHMEM options
- Tests: `tests/` cover intra‑, inter‑node, and low‑latency paths


## High‑Level Data Flow

User code calls Python APIs on `deep_ep.Buffer` which forward into the C++ runtime (`deep_ep_cpp.Buffer`), which schedules compute and communication on a dedicated CUDA stream and launches CUDA/NVSHMEM kernels.

```
Python (user)
  └─ deep_ep.Buffer (Python)
       └─ deep_ep_cpp.Buffer (C++ runtime via pybind11)
            ├─ layout::get_dispatch_layout (CUDA)
            ├─ intranode::* (NVLink CUDA)
            ├─ internode::* (NVSHMEM + NVLink + CUDA)
            └─ internode_ll::* (IBGDA low‑latency)
```

Key bindings (pybind11): [csrc/deep_ep.cpp:1723](csrc/deep_ep.cpp#L1723) exposes `Config`, `EventHandle`, and `Buffer` with all methods.


## Python API Layer

- `deep_ep/__init__.py` re‑exports core types: [deep_ep/__init__.py:1](deep_ep/__init__.py#L1)
- `EventOverlap` is a small helper wrapping C++ `EventHandle` to coordinate stream waits and record tensors for stream semantics: [deep_ep/utils.py:10](deep_ep/utils.py#L10).
- `Buffer` is the main user surface:
  - Definition and docstring: [deep_ep/buffer.py:13](deep_ep/buffer.py#L13)
  - Constructor coordinates NVLink/NVSHMEM setup (details below): [deep_ep/buffer.py:32](deep_ep/buffer.py#L32)
  - Layout computation: `get_dispatch_layout(...)` → C++: [deep_ep/buffer.py:291](deep_ep/buffer.py#L291)
  - All‑to‑all data movement:
    - `dispatch(...)` picks intra‑ vs. internode by RDMA availability: [deep_ep/buffer.py:320](deep_ep/buffer.py#L320)
    - `combine(...)` mirrors dispatch and reduces incoming shards: [deep_ep/buffer.py:404](deep_ep/buffer.py#L404)
  - Low‑latency APIs (`low_latency_*`) for IBGDA: [deep_ep/buffer.py:547](deep_ep/buffer.py#L547)


## C++ Runtime and Initialization

The C++ runtime owns device resources and drives kernels. Its lifetime is managed by Python’s `Buffer`.

- Constructor: allocates per‑GPU IPC/NVLink buffers, workspace, and host‑mapped counters; computes rank topology and validates sizes — see [csrc/deep_ep.cpp:17](csrc/deep_ep.cpp#L17).
- `sync(...)` completes peer setup:
  - Opens CUDA IPC handles for NVLink peer buffers and copies pointer tables to device
  - Initializes NVSHMEM using a root unique ID, allocates RDMA buffer(s), and optional shrink/mask buffers
  - Final device/cluster barriers
  - [csrc/deep_ep.cpp:211](csrc/deep_ep.cpp#L211)
- Destruction frees NVLink/NVSHMEM buffers, synchronizes, and tears down NVSHMEM if used — [csrc/deep_ep.cpp:164](csrc/deep_ep.cpp#L164).
- A dedicated comm stream is used for all communication: `get_comm_stream()` — [csrc/deep_ep.cpp:160](csrc/deep_ep.cpp#L160).

Environment knobs for RDMA/IBGDA are set in Python (`Buffer.__init__`), e.g. `NVSHMEM_IB_ENABLE_IBGDA`, QP depth, etc. DeepEP also supports “shrink” mode (masking failing ranks) implemented via NVSHMEM‑allocated mask/sync buffers.


## Configuration and Buffer Sizing

`Config` encapsulates kernel tiling and channelization parameters such as number of SMs and per‑channel send/recv depths, with buffer size estimators for NVL and RDMA layouts:

- `struct Config` and invariants — [csrc/config.hpp:23](csrc/config.hpp#L23)
- NVLink buffer size hint — [csrc/config.hpp:52](csrc/config.hpp#L52)
- RDMA buffer size hint — [csrc/config.hpp:76](csrc/config.hpp#L76)

Python convenience pickers `Buffer.get_dispatch_config(...)`/`get_combine_config(...)` provide tuned defaults indexed by world size — [deep_ep/buffer.py:231](deep_ep/buffer.py#L231) and [deep_ep/buffer.py:261](deep_ep/buffer.py#L261).


## Dispatch Layout Computation

`Buffer.get_dispatch_layout(topk_idx, num_experts, ...)` computes:

- `num_tokens_per_rank`: tokens per destination rank
- `num_tokens_per_rdma_rank`: tokens per destination RDMA rank (internode only)
- `num_tokens_per_expert`: tokens per expert
- `is_token_in_rank`: boolean matrix [num_tokens, num_ranks]

It records/waits events for optional async overlap and can allocate outputs on the comm stream. Python entry: [deep_ep/buffer.py:291](deep_ep/buffer.py#L291) → C++: [csrc/deep_ep.cpp:278](csrc/deep_ep.cpp#L278) → CUDA kernel: [csrc/kernels/layout.cu:8](csrc/kernels/layout.cu#L8).

The kernel parallelizes over experts and ranks per‑SM and builds the above statistics and mask in a single pass, writing results directly to device buffers.


## Intranode All‑to‑All (NVLink)

When `get_num_rdma_ranks() == 1`, DeepEP uses the NVLink path.

Phases (dispatch):

1) Notify and prefix computation: [csrc/kernels/intranode.cu:11](csrc/kernels/intranode.cu#L11) (`notify_dispatch`) builds per‑rank and per‑channel prefix matrices in remote NVLink‑visible buffers, using a GPU‑level barrier over a ring of IPC‑shared “barrier signals.” The host‑side wrapper launches via [csrc/kernels/intranode.cu:129](csrc/kernels/intranode.cu#L129).
2) Data movement: [csrc/kernels/intranode.cu:197](csrc/kernels/intranode.cu#L197) (`dispatch`) sends per‑token payloads into receiver‑owned ring buffers with per‑channel head/tail indices stored on the receiver side. SM90 TMA stores are used when available; otherwise vectorized global loads/stores.

Phases (combine):

1) Cached notify (`cached_notify_combine`) readies per‑channel queues (clean/heads) on receivers.
2) `combine(...)` reduces received shards with optional top‑k weights and bias, writing the final fused tensor for the local rank — see [csrc/kernels/api.cuh:186](csrc/kernels/api.cuh#L186) and implementation in [csrc/kernels/intranode.cu](csrc/kernels/intranode.cu).

Python orchestration selects the path and collects a reusable “handle” composed of the prefix matrices and queue heads — [deep_ep/buffer.py:320](deep_ep/buffer.py#L320) and [deep_ep/buffer.py:404](deep_ep/buffer.py#L404). The handle enables cached dispatches without recomputing layout/channel metadata.


## Internode All‑to‑All (RDMA + NVLink via NVSHMEM)

When `get_num_rdma_ranks() > 1`, DeepEP uses NVSHMEM for internode rendezvous and transfers, while retaining NVLink for intra‑node fan‑in/out.

Initialization and teams:

- Unique IDs and NVSHMEM init — [csrc/kernels/runtime.cu:41](csrc/kernels/runtime.cu#L41) and [csrc/kernels/runtime.cu:49](csrc/kernels/runtime.cu#L49)
- If `low_latency_mode` and world size > 8, a per‑GPU‑index RDMA team is split for synchronization — [csrc/kernels/runtime.cu:56](csrc/kernels/runtime.cu#L56)

Phases (dispatch):

1) Notify + reduce sizes: [csrc/kernels/internode.cu:49](csrc/kernels/internode.cu#L49) (`notify_dispatch`) exchanges per‑rank/per‑expert counts via NVSHMEM puts, computes RDMA/NVL prefix sums, and cleans per‑channel queues. Global and per‑team barriers synchronize epochs.
2) Data movement: [csrc/kernels/internode.cu](csrc/kernels/internode.cu) (`dispatch`) transmits payloads in two tiers:
   - RDMA tier uses `nvshmemi_ibgda_put_nbi_warp` to send contiguous slices to the correct RDMA peer into decoupled send/recv buffers (`SymBuffer`) and updates per‑RDMA‑rank heads
   - NVLink tier fans in from RDMA receivers to final NVL destinations, pushing into receiver‑owned per‑channel ring buffers — see [csrc/kernels/api.cuh:206](csrc/kernels/api.cuh#L206), [csrc/kernels/internode.cu](csrc/kernels/internode.cu)

Phases (combine):

1) Cached notify prepares per‑channel queues and writes combined heads for both NVL/RDMA tiers — [csrc/kernels/api.cuh:294](csrc/kernels/api.cuh#L294)
2) `combine(...)` reduces shards and optional top‑k weights + bias to produce the final tensor — [csrc/kernels/api.cuh:316](csrc/kernels/api.cuh#L316)

The Python `dispatch(...)`/`combine(...)` mirror the intranode API but return a richer handle that includes RDMA/NVL prefix matrices and source metadata — [deep_ep/buffer.py:460](deep_ep/buffer.py#L460).

GIL handling: the C++ internode dispatch releases the Python GIL during potentially long GPU/CPU waits to avoid blocking other Python work — [csrc/deep_ep.cpp:836](csrc/deep_ep.cpp#L836).


## Low‑Latency Path (IBGDA‑only)

Low‑latency dispatch/combine uses a symmetric RDMA ring with two alternating buffers (odd/even) and compact message formats. Buffer layout is computed from `LowLatencyLayout` — [csrc/config.hpp:127](csrc/config.hpp#L127).

Clean and barrier:

- Symmetric clean with NVSHMEM block‑barriers or masked per‑rank IBGDA “rendezvous” barrier — [csrc/kernels/internode_ll.cu:72](csrc/kernels/internode_ll.cu#L72) and [csrc/kernels/internode_ll.cu:104](csrc/kernels/internode_ll.cu#L104)

Dispatch and combine:

- `low_latency_dispatch(...)` copies (and optionally casts to FP8) selected tokens into RDMA send buffers, issues requests, and can return a “recv hook” for deferred synchronization — Python: [deep_ep/buffer.py:547](deep_ep/buffer.py#L547); C++: [csrc/deep_ep.cpp](csrc/deep_ep.cpp) (low‑latency methods); Kernels: [csrc/kernels/internode_ll.cu:129](csrc/kernels/internode_ll.cu#L129)
- `low_latency_combine(...)` optionally consumes prefilled RDMA buffers (zero‑copy), reduces shards (BF16 or LogFMT‑10), and returns the fused tensor — Python: [deep_ep/buffer.py:616](deep_ep/buffer.py#L616); Kernels: [csrc/kernels/internode_ll.cu](csrc/kernels/internode_ll.cu) (combine path starts after dispatch)

Shrink/failure masking:

- Ranks can be masked/unmasked dynamically via NVSHMEM‑allocated `mask_buffer` with device‑side checks inside barriers and send/recv paths — C++ API: [csrc/deep_ep.cpp:1702](csrc/deep_ep.cpp#L1702) and queries/clean at [csrc/deep_ep.cpp:1708](csrc/deep_ep.cpp#L1708), [csrc/deep_ep.cpp:1716](csrc/deep_ep.cpp#L1716); device checks: [csrc/kernels/internode_ll.cu:10](csrc/kernels/internode_ll.cu#L10)


## Synchronization Primitives

- Intranode barrier: cooperative block barrier over IPC “barrier signals” with timeouts — [csrc/kernels/runtime.cu:18](csrc/kernels/runtime.cu#L18) and [csrc/kernels/utils.cuh:232](csrc/kernels/utils.cuh#L232)
- NVSHMEM barriers: global or per‑team (`cpu_rdma_team`) — [csrc/kernels/runtime.cu:83](csrc/kernels/runtime.cu#L83)
- Event sequencing between compute/comm streams (`EventHandle`, `stream_wait`) — [csrc/event.hpp:22](csrc/event.hpp#L22) and [csrc/event.hpp:33](csrc/event.hpp#L33)


## Performance Notes

- Channelization: SMs are split into send/recv channels (even/odd blocks), maximizing overlap — see [csrc/kernels/intranode.cu:210](csrc/kernels/intranode.cu#L210) and `launch.cuh` setup
- SM90 features: TMA + cluster launches are used when available; fallbacks are enabled with `DISABLE_SM90_FEATURES` — [csrc/kernels/launch.cuh:6](csrc/kernels/launch.cuh#L6) and [csrc/kernels/utils.cuh](csrc/kernels/utils.cuh) (TMA helpers)
- Aggressive LD/ST cache hints are guarded by `DISABLE_AGGRESSIVE_PTX_INSTRS` — [csrc/kernels/utils.cuh:94](csrc/kernels/utils.cuh#L94)


## Background Primer

- NVLink: NVIDIA GPU‑GPU interconnect with high bandwidth and low latency used for intranode transfers. DeepEP uses CUDA IPC to share NVLink‑visible buffers and implements per‑channel ring queues for all‑to‑all.
- RDMA + NVSHMEM: NVSHMEM provides GPU‑initiated one‑sided operations across nodes; with IBGDA, the NIC doorbells can be rung directly from the GPU for low‑overhead puts/gets. DeepEP forms per‑GPU‑index teams for scalable synchronization.
- IBGDA: Infiniband GPUDirect Async enables GPUs to enqueue RDMA work directly to NIC QPs. DeepEP batches and quiets QPs between epochs to maintain ordering and avoid buffer reuse hazards.
- LogFMT/FP8: For low‑latency dispatch/combine, DeepEP supports FP8 E4M3 casting with optional power‑of‑two rounding and a compact LogFMT‑10 receive/combine format to reduce bandwidth while maintaining fidelity.


## Where To Look Next

- Public bindings and dtypes: [csrc/deep_ep.cpp:1723](csrc/deep_ep.cpp#L1723)
- Python entry points: [deep_ep/buffer.py:13](deep_ep/buffer.py#L13), [deep_ep/utils.py:10](deep_ep/utils.py#L10)
- NVLink intranode path: [csrc/kernels/intranode.cu](csrc/kernels/intranode.cu)
- NVSHMEM internode path: [csrc/kernels/internode.cu](csrc/kernels/internode.cu), [csrc/kernels/runtime.cu](csrc/kernels/runtime.cu)
- Low‑latency IBGDA path: [csrc/kernels/internode_ll.cu](csrc/kernels/internode_ll.cu), [csrc/config.hpp:127](csrc/config.hpp#L127)
