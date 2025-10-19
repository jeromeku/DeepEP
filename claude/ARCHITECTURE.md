# DeepEP Architecture

## Table of Contents

1. [Overview](#overview)
2. [Background: CUDA Networking Technologies](#background-cuda-networking-technologies)
3. [High-Level Architecture](#high-level-architecture)
4. [Component Deep Dive](#component-deep-dive)
   - [Python API Layer](#python-api-layer)
   - [C++ Core and PyBind11 Bindings](#c-core-and-pybind11-bindings)
   - [Buffer Management](#buffer-management)
   - [Kernel Layer](#kernel-layer)
5. [Data Flow Walkthrough](#data-flow-walkthrough)
6. [Multi-Device Synchronization](#multi-device-synchronization)
7. [Build System](#build-system)

---

## Overview

**DeepEP** is a high-performance communication library designed for **Mixture-of-Experts (MoE)** models using **Expert Parallelism (EP)**. It provides optimized all-to-all GPU communication kernels (dispatch and combine operations) with support for both high-throughput and low-latency scenarios.

### Key Features

- **High-throughput intranode communication** using NVLink
- **High-throughput internode communication** using RDMA (InfiniBand) + NVLink
- **Low-latency internode communication** using pure RDMA for inference decoding
- **FP8 and BF16 precision support**
- **Hook-based communication-computation overlapping** without SM occupation
- **SM control** for throughput optimization

### Target Use Cases

1. **Training and Inference Prefilling**: Normal kernels with NVLink + RDMA forwarding
2. **Inference Decoding**: Low-latency kernels with pure RDMA

---

## Background: CUDA Networking Technologies

Before diving into the architecture, let's understand the key networking technologies DeepEP leverages.

### NVLink

**NVLink** is NVIDIA's high-bandwidth, low-latency interconnect technology for GPU-to-GPU communication.

**Key Characteristics:**
- **Bandwidth**: Up to ~160 GB/s per direction on H800 GPUs (bidirectional ~300-400 GB/s)
- **Latency**: Sub-microsecond latency
- **Topology**: Connects GPUs within a single node (typically up to 8 GPUs)
- **Memory Model**: Enables peer-to-peer direct memory access between GPUs
- **Use in DeepEP**: Intranode (within-node) all-to-all communication

**How it works:**
```
GPU 0 ──NVLink──┐
GPU 1 ──NVLink──┼─── NVLink Switch ─── All GPUs can access each other's memory
GPU 2 ──NVLink──┤
   ...          └───
GPU 7 ──NVLink──
```

In DeepEP, NVLink is used via CUDA IPC (Inter-Process Communication) handles that allow one GPU to directly read/write another GPU's memory.

### RDMA (Remote Direct Memory Access)

**RDMA** allows direct memory access from one computer to another without involving the operating system, CPU, or cache.

**Key Characteristics:**
- **Bandwidth**: Up to ~50 GB/s per direction with InfiniBand CX7 400 Gb/s NICs
- **Latency**: Low microsecond range (~1-10 μs)
- **Zero-copy**: Data transferred directly between GPU memory across nodes
- **CPU offload**: Network card handles data transfer without CPU intervention

**RDMA Protocols:**
- **InfiniBand (IB)**: High-performance interconnect, what DeepEP primarily uses
- **RoCE (RDMA over Converged Ethernet)**: RDMA over Ethernet, theoretically supported

**How it works:**
```
Node 0                                    Node 1
┌─────────────┐                          ┌─────────────┐
│ GPU Memory  │                          │ GPU Memory  │
│   (VRAM)    │                          │   (VRAM)    │
└──────┬──────┘                          └──────┬──────┘
       │                                        │
┌──────▼──────┐                          ┌──────▼──────┐
│ InfiniBand  │◄──── InfiniBand Fabric ──┤ InfiniBand  │
│     NIC     │                          │     NIC     │
└─────────────┘                          └─────────────┘
```

### NVSHMEM

**NVSHMEM** (NVIDIA Symmetric Hierarchical Memory) is a parallel programming interface for multi-GPU and multi-node applications.

**Key Characteristics:**
- **PGAS Model**: Partitioned Global Address Space - each GPU has its own local memory but can access remote GPU memory
- **Symmetric Heaps**: Each GPU allocates a symmetric memory region that can be accessed by other GPUs
- **Built on RDMA**: Uses RDMA for internode communication
- **GPU-initiated**: GPU kernels can directly initiate remote memory operations

**NVSHMEM Operations:**
- `nvshmem_putmem()`: Write to remote GPU memory
- `nvshmem_getmem()`: Read from remote GPU memory
- `nvshmem_fence()`: Ensure memory operations complete
- `nvshmem_quiet()`: Wait for all outstanding operations

**How DeepEP uses NVSHMEM:**

DeepEP uses NVSHMEM with **IBGDA (InfiniBand GPU Direct Async)** for high-performance RDMA:

```c++
// From deep_ep/buffer.py lines 105-107
os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
```

**IBGDA** enables GPU kernels to directly post RDMA operations without CPU involvement using multiple **Queue Pairs (QPs)**.

### Queue Pairs (QPs)

A **Queue Pair** is a communication endpoint in RDMA consisting of:
- **Send Queue (SQ)**: Queue for outgoing messages
- **Receive Queue (RQ)**: Queue for incoming messages

**In DeepEP's low-latency mode:**
- Number of QPs per rank = Number of local experts ([buffer.py:257](../deep_ep/buffer.py#L257))
- Each expert has its own QP for parallel communication
- QP depth controlled by `NVSHMEM_QP_DEPTH` (default 1024)

### InfiniBand Virtual Lanes (VL)

**Virtual Lanes** provide traffic isolation on InfiniBand networks:

- Multiple virtual channels over a single physical link
- Prevent head-of-line blocking between different traffic types
- Controlled via `NVSHMEM_IB_SL` environment variable

**Recommended VL Assignment:**
1. Normal kernels (high-throughput)
2. Low-latency kernels
3. Other workloads

---

## High-Level Architecture

DeepEP is organized in layers, from user-facing Python APIs down to CUDA kernels:

```
┌────────────────────────────────────────────────────────────┐
│                     Python User Code                        │
│              (Training/Inference Framework)                 │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                  Python API Layer                           │
│          deep_ep.Buffer class (buffer.py)                   │
│   - dispatch(), combine(), low_latency_dispatch(), etc.     │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│              PyBind11 Bindings Layer                        │
│               deep_ep_cpp module                            │
│        (Python ←→ C++ interface)                            │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                   C++ Core Layer                            │
│          deep_ep::Buffer class (deep_ep.cpp/hpp)            │
│   - Buffer management, IPC setup, NVSHMEM init              │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                  CUDA Kernel Layer                          │
│                (csrc/kernels/*.cu)                          │
│  ┌──────────────┬──────────────┬────────────────┐          │
│  │  Intranode   │  Internode   │ Internode LL   │          │
│  │  (NVLink)    │ (RDMA+NVLink)│ (Low-Latency)  │          │
│  └──────────────┴──────────────┴────────────────┘          │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│              CUDA Driver & Hardware                         │
│  ┌──────────────────────┬─────────────────────────┐        │
│  │   NVLink (IPC)       │  NVSHMEM/IBGDA (RDMA)   │        │
│  └──────────────────────┴─────────────────────────┘        │
└────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
deepEP/
├── deep_ep/              # Python package
│   ├── __init__.py       # Package exports
│   ├── buffer.py         # Main Python API (Buffer class)
│   └── utils.py          # Helper utilities (EventOverlap, etc.)
│
├── csrc/                 # C++ source code
│   ├── deep_ep.cpp       # Main C++ implementation (Buffer class)
│   ├── deep_ep.hpp       # C++ Buffer class header
│   ├── config.hpp        # Configuration structures
│   ├── event.hpp         # Event handling
│   ├── CMakeLists.txt    # Build configuration
│   │
│   └── kernels/          # CUDA kernels
│       ├── api.cuh       # Kernel API declarations
│       ├── configs.cuh   # Kernel configurations
│       ├── buffer.cuh    # Buffer utilities
│       ├── utils.cuh     # Kernel utilities
│       ├── launch.cuh    # Kernel launch helpers
│       ├── exception.cuh # Error handling
│       │
│       ├── layout.cu     # Layout computation kernels
│       ├── runtime.cu    # Runtime initialization
│       │
│       ├── intranode.cu  # NVLink kernels (dispatch/combine)
│       ├── internode.cu  # RDMA+NVLink kernels (dispatch/combine)
│       ├── internode_ll.cu # Low-latency RDMA kernels
│       │
│       └── ibgda_device.cuh # IBGDA device-side utilities
│
├── tests/                # Test suite
│   ├── test_intranode.py
│   ├── test_internode.py
│   ├── test_low_latency.py
│   └── utils.py
│
├── setup.py              # Build & installation script
└── README.md             # Documentation
```

---

## Component Deep Dive

### Python API Layer

**File**: [deep_ep/buffer.py](../deep_ep/buffer.py)

The Python `Buffer` class is the main user-facing API. It wraps the C++ implementation and provides:

1. **Initialization and synchronization**
2. **Dispatch operations** (scatter tokens to experts)
3. **Combine operations** (gather tokens from experts)
4. **Low-latency operations** (for inference decoding)

#### Key Python API Methods

| Method | Description | Mode |
|--------|-------------|------|
| `__init__()` | Initialize buffer, setup IPC/NVSHMEM | All |
| `dispatch()` | Scatter tokens to experts | Normal |
| `combine()` | Gather tokens from experts | Normal |
| `get_dispatch_layout()` | Compute token routing layout | Normal |
| `low_latency_dispatch()` | Low-latency token scatter | Low-Latency |
| `low_latency_combine()` | Low-latency token gather | Low-Latency |
| `set_num_sms()` | Control SM usage (static method) | Normal |
| `get_comm_stream()` | Get communication CUDA stream | All |

#### Buffer Initialization Flow

**Entry Point**: `Buffer.__init__()` in [buffer.py:31-134](../deep_ep/buffer.py#L31-L134)

```python
def __init__(self,
             group: Optional[dist.ProcessGroup],
             num_nvl_bytes: int = 0,
             num_rdma_bytes: int = 0,
             low_latency_mode: bool = False,
             num_qps_per_rank: int = 24,
             ...)
```

**Initialization Steps:**

```
1. Check NVLink connections
   ↓
2. Get rank info from process group
   ↓
3. Create C++ Buffer runtime (deep_ep_cpp.Buffer)
   ↓
4. All-gather device IDs across ranks
   ↓
5. All-gather IPC handles for NVLink buffers
   ↓
6. If internode or low-latency:
   - Configure NVSHMEM environment variables
   - All-gather NVSHMEM unique ID from root
   ↓
7. Call runtime.sync() to make buffer available
   ↓
8. Assert buffer is available
```

**Code trace:**

```python
# deep_ep/buffer.py:63-90
check_nvlink_connections(group)

# Get rank information
self.rank = group.rank()
self.group = group
self.group_size = group.size()

# Create C++ runtime
self.runtime = deep_ep_cpp.Buffer(
    self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes,
    low_latency_mode, explicitly_destroy, enable_shrink
)

# Synchronize metadata
device_ids = all_gather_object(self.runtime.get_local_device_id())
ipc_handles = all_gather_object(self.runtime.get_local_ipc_handle())

# For RDMA: synchronize NVSHMEM
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    # Configure NVSHMEM environment
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
    # ... more config ...

    # Get unique ID from root rank
    root_unique_id = self.runtime.get_local_nvshmem_unique_id()
    nvshmem_unique_ids = all_gather_object(root_unique_id)
    root_unique_id = nvshmem_unique_ids[root_rank]

# Make runtime available
self.runtime.sync(device_ids, ipc_handles, root_unique_id)
```

---

### C++ Core and PyBind11 Bindings

**Files**:
- [csrc/deep_ep.cpp](../csrc/deep_ep.cpp) - C++ implementation
- [csrc/deep_ep.hpp](../csrc/deep_ep.hpp) - C++ headers

The C++ layer handles:
1. **Buffer allocation** (NVLink and RDMA)
2. **IPC setup** (NVLink peer-to-peer access)
3. **NVSHMEM initialization** (RDMA symmetric memory)
4. **Kernel launches** and parameter marshaling

#### C++ Buffer Class Structure

**Header**: [deep_ep.hpp:26-266](../csrc/deep_ep.hpp#L26-L266)

```cpp
namespace deep_ep {

struct Buffer {
private:
    // Low-latency mode
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer (IPC-based)
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};  // Up to 8 local GPUs
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer (RDMA-based)
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Device info
    int device_id;
    int num_device_sms;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Communication stream
    at::cuda::CUDAStream comm_stream;

    // Synchronization
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Host-mapped counters (for CPU-GPU synchronization)
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;
    // ... more counters ...

public:
    Buffer(int rank, int num_ranks, int64_t num_nvl_bytes,
           int64_t num_rdma_bytes, bool low_latency_mode, ...);
    ~Buffer() noexcept(false);

    // Synchronization
    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt);

    // Layout computation
    std::tuple<...> get_dispatch_layout(...);

    // Dispatch/combine operations
    std::tuple<...> intranode_dispatch(...);
    std::tuple<...> intranode_combine(...);
    std::tuple<...> internode_dispatch(...);
    std::tuple<...> internode_combine(...);
    std::tuple<...> low_latency_dispatch(...);
    std::tuple<...> low_latency_combine(...);
};

}  // namespace deep_ep
```

#### C++ Buffer Construction

**Implementation**: [deep_ep.cpp:16-102](../csrc/deep_ep.cpp#L16-L102)

```cpp
Buffer::Buffer(int rank, int num_ranks,
               int64_t num_nvl_bytes, int64_t num_rdma_bytes,
               bool low_latency_mode, bool explicitly_destroy, bool enable_shrink)
    : rank(rank), num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes), num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      comm_stream(at::cuda::getStreamFromPool(true)) {

    // 1. Get device info
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS;  // Which RDMA node
    nvl_rank = rank % NUM_MAX_NVL_PEERS;   // Which GPU in node
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // 2. Allocate NVLink buffer
    if (num_nvl_bytes > 0) {
        cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + metadata_bytes);
        cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        // Setup barrier signals, buffer pointers array
        barrier_signal_ptrs[nvl_rank] = ...;
        cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, ...);
    }

    // 3. Allocate workspace
    cudaMalloc(&workspace, NUM_WORKSPACE_BYTES);  // 32 MiB

    // 4. Allocate host-mapped counters
    cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&moe_recv_counter_mapped, moe_recv_counter, 0);
    *moe_recv_counter = -1;

    // ... expert counters, RDMA counters ...
}
```

**Key points:**
- `buffer_ptrs[nvl_rank]`: Local GPU's buffer that will be shared via IPC
- `barrier_signal_ptrs`: For intra-node synchronization
- `moe_recv_counter`: Host-mapped memory for GPU→CPU signaling
- `comm_stream`: Dedicated CUDA stream for communication

#### Buffer Synchronization

**Implementation**: [deep_ep.cpp:~230-340](../csrc/deep_ep.cpp#L230-L340) (approximate line numbers)

The `sync()` method completes the buffer setup after metadata exchange:

```cpp
void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
    const std::optional<pybind11::bytearray>& root_unique_id_opt)
{
    // 1. Open IPC handles from all other GPUs in the node
    if (num_nvl_bytes > 0) {
        for (int i = 0; i < num_nvl_ranks; ++i) {
            if (i != nvl_rank) {
                cudaIpcMemHandle_t remote_handle;
                memcpy(&remote_handle, all_gathered_handles[i]->data(), ...);
                cudaIpcOpenMemHandle(&buffer_ptrs[i], remote_handle, ...);
            }
        }

        // Copy buffer pointers to GPU
        cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, ...);

        // Setup barrier signal pointers
        for (int i = 0; i < num_nvl_ranks; ++i) {
            barrier_signal_ptrs[i] = ...;
        }
        cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, ...);
    }

    // 2. Initialize NVSHMEM for RDMA
#ifndef DISABLE_NVSHMEM
    if (num_rdma_bytes > 0 || low_latency_mode) {
        // Initialize NVSHMEM
        internode::init(root_unique_id_val, rdma_rank, num_rdma_ranks,
                       low_latency_mode);

        // Allocate symmetric RDMA buffer
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, alignment);

        // Zero out the buffer
        cudaMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes, comm_stream);
    }
#endif

    // 3. Synchronize all ranks
    cudaDeviceSynchronize();
    if (num_nvl_bytes > 0) {
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank,
                          num_nvl_ranks, comm_stream);
    }
    if (num_rdma_ranks > 1 || low_latency_mode) {
        internode::barrier();
    }

    available = true;
}
```

**Flow:**
```
For each rank:
  1. Open IPC handles → Can now access peer GPU memory
  2. Copy buffer_ptrs to GPU → Kernels can access all buffers
  3. Initialize NVSHMEM → RDMA infrastructure ready
  4. Allocate symmetric RDMA buffer → All ranks have rdma_buffer_ptr
  5. Barrier synchronization → Everyone ready
```

---

### Buffer Management

DeepEP uses two types of buffers depending on the communication pattern:

#### 1. NVLink Buffers (Intranode)

**Allocation**: [deep_ep.cpp:66-78](../csrc/deep_ep.cpp#L66-L78)

```cpp
// Each GPU allocates its own buffer
cudaMalloc(&buffer_ptrs[nvl_rank],
           num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + ...);

// Get IPC handle for this buffer
cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
```

**Sharing**: [deep_ep.cpp:~240-260](../csrc/deep_ep.cpp#L240-L260)

```cpp
// During sync(), each GPU opens handles to peer GPUs
for (int i = 0; i < num_nvl_ranks; ++i) {
    if (i != nvl_rank) {
        cudaIpcOpenMemHandle(&buffer_ptrs[i], remote_ipc_handles[i], ...);
    }
}

// Now buffer_ptrs[i] can be used to access GPU i's memory
```

**Memory Layout:**

```
buffer_ptrs[nvl_rank]:
┌─────────────────────────────────────┐
│                                     │
│      Communication Buffer           │  num_nvl_bytes
│        (data transfer)              │
│                                     │
├─────────────────────────────────────┤
│  Barrier Signals [0..NUM_MAX_PEERS] │  NUM_MAX_NVL_PEERS * sizeof(int)
├─────────────────────────────────────┤
│  Buffer Pointers Array              │  NUM_MAX_NVL_PEERS * sizeof(void*)
├─────────────────────────────────────┤
│  Barrier Signal Pointers Array      │  NUM_MAX_NVL_PEERS * sizeof(int*)
└─────────────────────────────────────┘
```

**Size Calculation**: [config.hpp:52-74](../csrc/config.hpp#L52-L74)

```cpp
size_t Config::get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    const auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
    const int num_channels = num_sms / 2;

    size_t num_bytes = 0;
    // Prefix matrices and metadata
    num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
    // Data buffer
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
    // Source metadata (for internode)
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * source_meta_bytes;
    // Top-k indices
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);
    // Top-k weights
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
    // Scales (for FP8)
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);

    return ((num_bytes + 127) / 128) * 128;  // Align to 128 bytes
}
```

#### 2. RDMA Buffers (Internode)

**Allocation**: Via NVSHMEM symmetric heap

```cpp
// In Buffer::sync() - deep_ep.cpp
rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
```

**Properties:**
- **Symmetric**: Every rank allocates the same size
- **Addressable**: Each rank can access any other rank's buffer via NVSHMEM
- **GPU-initiated**: Kernels can directly issue RDMA operations

**Size Calculation**: [config.hpp:76-104](../csrc/config.hpp#L76-L104)

```cpp
size_t Config::get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
    if (num_ranks <= NUM_MAX_NVL_PEERS) return 0;  // Intranode only

    const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int num_channels = num_sms / 2;

    size_t num_bytes = 0;
    // Prefix matrices (2x for double buffering)
    num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
    // Data buffers (2x for dispatch/combine)
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
    // Source metadata (2x)
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * source_meta_bytes * 2;
    // Top-k indices, weights, scales (2x each)
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
    // Control metadata
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;

    return ((num_bytes + 127) / 128) * 128;
}
```

#### 3. Low-Latency Buffers

For low-latency mode, buffers use a different layout optimized for minimal latency:

**Layout**: [config.hpp:136-187](../csrc/config.hpp#L136-L187)

```cpp
struct LowLatencyLayout {
    LowLatencyBuffer buffers[2];  // Odd/even double buffering

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank,
                     int hidden, int num_ranks, int num_experts) {
        // Calculate message sizes
        size_t num_bytes_per_dispatch_msg = sizeof(int4) +
            std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));
        size_t num_bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) +
            hidden * sizeof(nv_bfloat16);

        // Layout: [Send Buffer 0][Send Buffer 1][Recv Buffer 0][Recv Buffer 1]
        //         [Signal Buffer 0][Signal Buffer 1]

        // Send buffers
        size_t send_buffer_bytes = std::max(
            num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg,
            num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg
        );

        // Receive buffers
        size_t recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank *
            std::max(num_bytes_per_dispatch_msg, num_bytes_per_combine_msg);

        // Signaling buffers (for completion notification)
        size_t signaling_buffer_bytes = num_experts * sizeof(int);

        // Assign pointers with proper offsets
        // ...
    }
};
```

**Double Buffering:**
- Buffer 0: For odd iterations
- Buffer 1: For even iterations
- Allows pipelining: while processing buffer 0, buffer 1 receives next batch

---

## Kernel Layer

The CUDA kernels are organized by communication pattern:

| Kernel File | Communication | Use Case |
|------------|---------------|----------|
| `layout.cu` | None | Compute token routing |
| `intranode.cu` | NVLink | Single-node, high-throughput |
| `internode.cu` | RDMA + NVLink | Multi-node, high-throughput |
| `internode_ll.cu` | RDMA | Multi-node, low-latency |

### Kernel API

**Header**: [kernels/api.cuh](../csrc/kernels/api.cuh)

The API is organized into namespaces:

```cpp
namespace deep_ep {
    namespace intranode { /* NVLink barrier, dispatch, combine */ }
    namespace internode { /* NVSHMEM init, RDMA dispatch, combine */ }
    namespace internode_ll { /* Low-latency dispatch, combine */ }
    namespace layout { /* Token routing computation */ }
}
```

### Channel-based Parallelism

DeepEP uses a **channel-based** design for parallelism:

```
Number of channels = num_sms / 2
```

Each channel is handled by one or more thread blocks:

```
GPU with 108 SMs, num_sms = 20:
  → 10 channels (channels 0-9)
  → Each channel processes 1/10th of the data
```

**Why channels?**
- **Load balancing**: Divide work evenly across SMs
- **Overlap**: Different channels can send/receive independently
- **Buffering**: Each channel has its own buffer slice

---

## Data Flow Walkthrough

This section will walk through complete examples of how data flows through the system. We'll trace both **Python → C++ → CUDA** for key operations.

### Example 1: Intranode Dispatch (Training)

**Scenario**: 8 GPUs in a single node, dispatching tokens for MoE forward pass.

**Python Entry**: [buffer.py:~300-400](../deep_ep/buffer.py#L300-L400) (approximate)

```python
# User code
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
    buffer.dispatch(
        x=hidden_states,                    # [num_tokens, hidden] BF16
        topk_idx=topk_idx,                  # [num_tokens, num_topk] int32/int64
        topk_weights=topk_weights,          # [num_tokens, num_topk] FP32
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=True
    )
```

**Data Flow:**

```
Step 1: Python → C++ Binding
────────────────────────────
buffer.py dispatch() calls:
  → self.runtime.intranode_dispatch(...)  [PyBind11 call]

Step 2: C++ Layer
────────────────────────────
deep_ep.cpp Buffer::intranode_dispatch():
  1. Compute layout if needed
  2. Launch notify kernel (notify peers of incoming data)
  3. Launch dispatch kernel (send data)
  4. Return tensors and handle

Step 3: CUDA Kernels
────────────────────────────
Kernel 1: intranode::notify_dispatch()
  - Compute per-channel prefix sums
  - Write metadata to peer buffers
  - Signal peers via barrier

Kernel 2: intranode::dispatch()
  - Read local data (x, topk_idx, topk_weights)
  - Scatter to appropriate peer buffers based on routing
  - Each channel independently sends its data slice

Step 4: Data in Buffers
────────────────────────────
Each rank's buffer now contains tokens sent to it by all ranks
```

**Detailed trace:**

1. **Python dispatch()**: [buffer.py](../deep_ep/buffer.py) (search for `def dispatch`)

```python
def dispatch(self, x, topk_idx=None, topk_weights=None, ...):
    # Prepare arguments
    dispatch_args = {...}

    # Call C++ based on mode
    if self.low_latency_mode:
        return self.low_latency_dispatch(...)
    elif self.runtime.is_internode_available():
        return self._internode_dispatch(...)
    else:
        return self._intranode_dispatch(...)
```

2. **C++ intranode_dispatch()**: [deep_ep.cpp](../csrc/deep_ep.cpp) (search for `intranode_dispatch`)

This method marshals parameters and calls the kernel API:

```cpp
std::tuple<...> Buffer::intranode_dispatch(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_idx,
    ...)
{
    // 1. Compute metadata
    int num_tokens = x.size(0);
    int hidden_int4 = x.size(1) * x.element_size() / sizeof(int4);

    // 2. Allocate receive buffers
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx_options);
    // ... more allocations ...

    // 3. Launch notify kernel
    intranode::notify_dispatch(
        num_tokens_per_rank.data_ptr<int>(),
        moe_recv_counter_mapped,
        num_ranks,
        num_tokens_per_expert.data_ptr<int>(),
        // ... many parameters ...
        buffer_ptrs_gpu,
        barrier_signal_ptrs_gpu,
        rank,
        comm_stream,
        config.num_sms
    );

    // 4. Launch dispatch kernel
    intranode::dispatch(
        recv_x.data_ptr(),
        // ... receive buffer pointers ...
        x.const_data_ptr(),
        // ... send data pointers ...
        buffer_ptrs_gpu,
        rank,
        num_ranks,
        comm_stream,
        config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens
    );

    // 5. Return results
    return std::make_tuple(recv_x, recv_topk_idx, ...);
}
```

3. **CUDA Notify Kernel**: See [KERNEL_INTRANODE_NOTIFY.md](./KERNEL_INTRANODE_NOTIFY.md) (to be created)

4. **CUDA Dispatch Kernel**: See [KERNEL_INTRANODE_DISPATCH.md](./KERNEL_INTRANODE_DISPATCH.md) (to be created)

---

This is the foundation of the ARCHITECTURE.md. I'll continue building out:
- More detailed kernel walkthroughs
- Internode (RDMA) flow
- Low-latency flow
- Multi-device synchronization
- Test walkthroughs

Let me save this and create the supplementary kernel documentation files.

