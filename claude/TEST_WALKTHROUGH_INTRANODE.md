# Test Walkthrough: Intranode Dispatch and Combine

This document provides a complete, line-by-line walkthrough of the intranode test, tracing the call path from Python user code down to CUDA kernel execution.

## Overview

**Test File**: [tests/test_intranode.py](../tests/test_intranode.py)

**Purpose**: Test high-throughput intranode (NVLink-based) dispatch and combine operations for MoE expert parallelism.

**Communication Pattern**: Within a single node (up to 8 GPUs connected via NVLink)

---

## Test Setup

### Entry Point

**File**: [tests/test_intranode.py](../tests/test_intranode.py)

```python
# Main execution starts here
if __name__ == '__main__':
    main()
```

### Initialization Flow

#### Step 1: Parse Arguments

**Lines**: [test_intranode.py:~400-450](../tests/test_intranode.py#L400-L450) (approximate)

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num_topk', type=int, default=8)
    parser.add_argument('--num_experts', type=int, default=8)
    # ... more arguments ...
    args = parser.parse_args()
```

**Default Configuration** (matching DeepSeek-V3):
- `num_tokens`: 4096 tokens per batch
- `hidden`: 7168 hidden dimension
- `num_topk`: 8 experts selected per token
- `num_experts`: 8 total experts

#### Step 2: Initialize Distributed Environment

**Lines**: [tests/utils.py](../tests/utils.py) (search for `init_dist`)

```python
def init_dist():
    """Initialize PyTorch distributed environment"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    return local_rank, world_size, rank
```

**Call Path:**
```
Python code
  ↓
torch.distributed.init_process_group()
  ↓
NCCL initialization (background process group for PyTorch ops)
  ↓
Returns: local_rank, world_size, rank
```

**Note**: DeepEP uses its own communication primitives (NVLink IPC + NVSHMEM), not NCCL. The process group is only used for:
1. Rank coordination
2. All-gather operations for metadata exchange
3. Barriers for synchronization

#### Step 3: Create Buffer

**Lines**: [test_intranode.py:~30-75](../tests/test_intranode.py#L30-L75)

```python
# Set number of SMs to use
num_sms = 20
deep_ep.Buffer.set_num_sms(num_sms)

# Calculate buffer sizes
config = deep_ep.Config(num_sms, 8, 256)  # num_sms, send_tokens, recv_tokens
num_nvl_bytes = config.get_nvl_buffer_size_hint(hidden * 2, num_ranks)  # BF16 = 2 bytes

# Create buffer
buffer = deep_ep.Buffer(
    group=group,
    num_nvl_bytes=num_nvl_bytes,
    num_rdma_bytes=0,  # Intranode only, no RDMA
    low_latency_mode=False
)
```

**Buffer Initialization Call Path:**

```
┌─────────────────────────────────────────────────────────────┐
│ Python: deep_ep.Buffer.__init__()                           │
│ File: deep_ep/buffer.py:31-134                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─> 1. check_nvlink_connections(group)
                     │     Validates NVLink topology
                     │
                     ├─> 2. deep_ep_cpp.Buffer(rank, num_ranks, ...)
                     │     Creates C++ Buffer object (PyBind11 call)
                     │
                     ├─> 3. all_gather_object(device_ids)
                     │     Exchange CUDA device IDs across ranks
                     │
                     ├─> 4. all_gather_object(ipc_handles)
                     │     Exchange IPC handles for NVLink access
                     │
                     └─> 5. runtime.sync(device_ids, ipc_handles, None)
                           Finalize buffer setup
```

**C++ Buffer Constructor Call Path:**

```
┌─────────────────────────────────────────────────────────────┐
│ C++: deep_ep::Buffer::Buffer()                              │
│ File: csrc/deep_ep.cpp:16-102                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─> 1. cudaGetDevice(&device_id)
                     │     Get current CUDA device
                     │
                     ├─> 2. Calculate ranks:
                     │     rdma_rank = rank / 8  (RDMA node ID)
                     │     nvl_rank = rank % 8   (GPU ID within node)
                     │
                     ├─> 3. cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + ...)
                     │     Allocate local GPU buffer
                     │
                     ├─> 4. cudaIpcGetMemHandle(&ipc_handles[nvl_rank], ...)
                     │     Get IPC handle for this buffer
                     │
                     ├─> 5. Setup barrier signals
                     │     barrier_signal_ptrs[nvl_rank] = ...
                     │
                     ├─> 6. cudaMalloc(&workspace, 32 MiB)
                     │     Allocate workspace memory
                     │
                     └─> 7. cudaMallocHost(&moe_recv_counter, ...)
                           Allocate host-mapped counter for GPU→CPU signaling
```

**C++ Buffer::sync() Call Path:**

```
┌─────────────────────────────────────────────────────────────┐
│ C++: deep_ep::Buffer::sync()                                │
│ File: csrc/deep_ep.cpp:~230-340                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─> 1. For each peer GPU i != nvl_rank:
                     │     cudaIpcOpenMemHandle(&buffer_ptrs[i], remote_handle, ...)
                     │     Now can access peer GPU memory via buffer_ptrs[i]
                     │
                     ├─> 2. cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, ...)
                     │     Copy buffer pointers to GPU memory
                     │     Kernels will use buffer_ptrs_gpu to access peer buffers
                     │
                     ├─> 3. Setup barrier signal pointers
                     │     For each i: barrier_signal_ptrs[i] = ...
                     │     cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, ...)
                     │
                     ├─> 4. cudaDeviceSynchronize()
                     │     Wait for all allocations/copies to complete
                     │
                     ├─> 5. intranode::barrier(...)
                     │     Synchronize all ranks via NVLink barrier
                     │
                     └─> 6. available = true
                           Buffer is now ready for communication
```

**After Buffer Creation:**

Each rank now has:
- `buffer_ptrs[0..7]`: Pointers to all 8 GPU buffers (local + 7 peers)
- `buffer_ptrs_gpu`: Device copy of these pointers for kernel access
- `barrier_signal_ptrs_gpu`: Device pointers for synchronization

**Memory Layout Example (Rank 0's View):**

```
Local GPU 0:
  buffer_ptrs[0] → Own buffer (local memory)
  buffer_ptrs[1] → GPU 1's buffer (via IPC)
  buffer_ptrs[2] → GPU 2's buffer (via IPC)
  ...
  buffer_ptrs[7] → GPU 7's buffer (via IPC)

All reads/writes are direct GPU-to-GPU over NVLink!
```

---

## Dispatch Operation (Forward Pass)

### High-Level Flow

```
User has tokens → Route tokens to experts → Experts process → Combine results
                   ↑                                           ↑
              DISPATCH                                    COMBINE
```

**Dispatch** = Scatter tokens from all ranks to the ranks hosting their assigned experts

### Test Code

**File**: [test_intranode.py:~78-112](../tests/test_intranode.py#L78-L112)

```python
# Test data
x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
topk_idx = torch.topk(scores, num_topk, dim=-1)[1]  # [num_tokens, num_topk]
topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank

# Prepare layout
num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
    buffer.get_dispatch_layout(topk_idx, num_experts)

# Configuration
config = deep_ep.Config(num_sms, 8, 256)

# Execute dispatch
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
    buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=True
    )
```

### Step 1: Layout Computation

**Python Entry**: `buffer.get_dispatch_layout()`

**File**: [buffer.py](../deep_ep/buffer.py) (search for `get_dispatch_layout`)

```python
def get_dispatch_layout(
    self,
    topk_idx: torch.Tensor,     # [num_tokens, num_topk]
    num_experts: int,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[EventOverlap]]:
    """
    Compute token routing layout:
    - Which tokens go to which ranks
    - How many tokens each rank receives
    - Which local expert each token targets
    """

    # Call C++ implementation
    return self.runtime.get_dispatch_layout(
        topk_idx, num_experts,
        previous_event.handle if previous_event else None,
        async_finish, allocate_on_comm_stream
    )
```

**C++ Entry**: `Buffer::get_dispatch_layout()`

**File**: [deep_ep.cpp](../csrc/deep_ep.cpp) (search for `get_dispatch_layout`)

```cpp
std::tuple<...> Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx,
    int num_experts,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream)
{
    // 1. Parse inputs
    int num_tokens = topk_idx.size(0);
    int num_topk = topk_idx.size(1);

    // 2. Allocate output tensors
    auto num_tokens_per_rank = torch::empty({num_ranks}, torch::kInt32);
    auto num_tokens_per_rdma_rank = num_rdma_ranks > 1 ?
        torch::empty({num_rdma_ranks}, torch::kInt32) : std::nullopt;
    auto num_tokens_per_expert = torch::empty({num_experts}, torch::kInt32);
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, torch::kBool);

    // 3. Select stream
    cudaStream_t stream = allocate_on_comm_stream ? comm_stream : at::cuda::getCurrentCUDAStream();

    // 4. Wait for previous event if needed
    if (previous_event) {
        previous_event->current_stream_wait(stream);
    }

    // 5. Launch layout kernel
    layout::get_dispatch_layout(
        topk_idx.const_data_ptr<topk_idx_t>(),
        num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank ? num_tokens_per_rdma_rank->data_ptr<int>() : nullptr,
        num_tokens_per_expert.data_ptr<int>(),
        is_token_in_rank.data_ptr<bool>(),
        num_tokens,
        num_topk,
        num_ranks,
        num_experts,
        stream
    );

    // 6. Create event if async
    std::optional<EventHandle> event = async ? EventHandle() : std::nullopt;
    if (event) {
        event->record(stream);
    }

    // 7. Return results
    return std::make_tuple(
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        event
    );
}
```

**CUDA Kernel**: `layout::get_dispatch_layout()`

**File**: [kernels/layout.cu](../csrc/kernels/layout.cu)

```cuda
__global__ void get_dispatch_layout_kernel(
    const topk_idx_t* topk_idx,       // [num_tokens, num_topk]
    int* num_tokens_per_rank,         // [num_ranks] - output
    int* num_tokens_per_rdma_rank,    // [num_rdma_ranks] - output
    int* num_tokens_per_expert,       // [num_experts] - output
    bool* is_token_in_rank,           // [num_tokens, num_ranks] - output
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts)
{
    // Each thread processes one or more tokens
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_id < num_tokens) {
        // For this token, iterate over its top-k experts
        for (int k = 0; k < num_topk; k++) {
            int expert_id = topk_idx[token_id * num_topk + k];

            if (expert_id >= 0 && expert_id < num_experts) {
                // Which rank owns this expert?
                int target_rank = expert_id / (num_experts / num_ranks);

                // Mark that this token goes to target_rank
                is_token_in_rank[token_id * num_ranks + target_rank] = true;

                // Atomically increment counters
                atomicAdd(&num_tokens_per_rank[target_rank], 1);
                atomicAdd(&num_tokens_per_expert[expert_id], 1);

                if (num_tokens_per_rdma_rank != nullptr) {
                    int rdma_rank = target_rank / NUM_MAX_NVL_PEERS;
                    atomicAdd(&num_tokens_per_rdma_rank[rdma_rank], 1);
                }
            }
        }
    }
}
```

**What This Kernel Computes:**

Example with 4 tokens, 8 experts, 8 ranks (1 expert per rank):

```
Token 0: experts [2, 5, 7, 1] → ranks [2, 5, 7, 1]
Token 1: experts [0, 3, 4, 6] → ranks [0, 3, 4, 6]
Token 2: experts [1, 2, 5, 7] → ranks [1, 2, 5, 7]
Token 3: experts [3, 4, 5, 6] → ranks [3, 4, 5, 6]

After kernel:
num_tokens_per_rank = [1, 2, 2, 2, 2, 3, 2, 2]  (how many tokens each rank receives)
num_tokens_per_expert = [1, 2, 2, 2, 2, 3, 2, 2]  (how many tokens each expert processes)
is_token_in_rank:
  Token 0: [F, T, T, F, F, T, F, T]  (goes to ranks 1,2,5,7)
  Token 1: [T, F, F, T, T, F, T, F]  (goes to ranks 0,3,4,6)
  Token 2: [F, T, T, F, F, T, F, T]  (goes to ranks 1,2,5,7)
  Token 3: [F, F, F, T, T, T, T, F]  (goes to ranks 3,4,5,6)
```

**Layout Computation Summary:**

```
Input: topk_idx [num_tokens, num_topk]
  ↓
Kernel: For each token, for each selected expert:
  - Determine target rank
  - Mark is_token_in_rank[token, rank] = true
  - Increment num_tokens_per_rank[rank]
  - Increment num_tokens_per_expert[expert]
  ↓
Output: Routing information for dispatch
```

### Step 2: Dispatch Execution

**Python Entry**: `buffer.dispatch()`

**File**: [buffer.py](../deep_ep/buffer.py) (search for `def dispatch`)

```python
def dispatch(
    self,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    is_token_in_rank: Optional[torch.Tensor] = None,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    config: Optional[Config] = None,
    async_finish: bool = False,
    ...
) -> Tuple[...]:
    """
    Dispatch tokens to experts across ranks.

    For intranode: Uses NVLink for direct GPU-to-GPU transfer
    For internode: Uses RDMA + NVLink

    Returns:
        recv_x: Received hidden states
        recv_topk_idx: Received expert indices
        recv_topk_weights: Received routing weights
        num_recv_tokens_per_expert_list: Tokens per local expert
        handle: Opaque handle for combine operation
        event: CUDA event for synchronization
    """

    # Route to appropriate implementation
    if self.low_latency_mode:
        return self.low_latency_dispatch(...)
    elif self.runtime.is_internode_available():
        return self._internode_dispatch(...)
    else:
        return self._intranode_dispatch(...)
```

**Python Intranode Dispatch**: `buffer._intranode_dispatch()`

```python
def _intranode_dispatch(self, x, topk_idx, topk_weights, ...):
    # Prepare configuration
    if config is None:
        config = self.get_dispatch_config(self.group_size)

    # Call C++ runtime
    (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
     num_recv_tokens_per_expert_list,
     rank_prefix_matrix, channel_prefix_matrix,
     recv_channel_offset, send_head, recv_src_idx,
     event_handle) = self.runtime.intranode_dispatch(
        x,
        x_scales if isinstance(x, tuple) else None,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        cached_num_recv_tokens,
        cached_rank_prefix_matrix,
        cached_channel_prefix_matrix,
        expert_alignment,
        num_worst_tokens,
        config,
        previous_event.handle if previous_event else None,
        async_finish,
        allocate_on_comm_stream
    )

    # Package results
    handle = (rank_prefix_matrix, channel_prefix_matrix, ...)
    event = EventOverlap(event_handle) if event_handle else None

    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event
```

**C++ Intranode Dispatch**: `Buffer::intranode_dispatch()`

**File**: [deep_ep.cpp](../csrc/deep_ep.cpp) (search for `intranode_dispatch`)

```cpp
std::tuple<...> Buffer::intranode_dispatch(
    const torch::Tensor& x,  // [num_tokens, hidden]
    const std::optional<torch::Tensor>& x_scales,  // FP8 scales if applicable
    const std::optional<torch::Tensor>& topk_idx,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& num_tokens_per_rank,
    const torch::Tensor& is_token_in_rank,
    const std::optional<torch::Tensor>& num_tokens_per_expert,
    int cached_num_recv_tokens,
    ...,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream)
{
    // ─────────────────────────────────────────────────────────────
    // 1. SETUP: Parse inputs and validate
    // ─────────────────────────────────────────────────────────────
    int num_tokens = x.size(0);
    int hidden = x.size(1);
    int hidden_int4 = hidden * x.element_size() / sizeof(int4);
    bool has_topk = topk_idx.has_value();
    int num_topk = has_topk ? topk_idx->size(1) : 0;
    int num_experts = num_tokens_per_expert ? num_tokens_per_expert->size(0) : 0;

    // FP8 support
    bool use_fp8 = x_scales.has_value();
    int num_scales = use_fp8 ? x_scales->size(0) : 0;

    // ─────────────────────────────────────────────────────────────
    // 2. ALLOCATE: Create output tensors
    // ─────────────────────────────────────────────────────────────

    // Determine number of tokens this rank will receive
    int num_recv_tokens;
    if (cached_num_recv_tokens > 0) {
        num_recv_tokens = cached_num_recv_tokens;
    } else {
        // Wait for GPU to compute the count
        // (notify_dispatch kernel will signal via moe_recv_counter)
        num_recv_tokens = /* will be set by kernel, see below */;
    }

    // Allocate receive buffers
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = has_topk ?
        torch::empty({num_recv_tokens, num_topk}, topk_idx->options()) :
        std::nullopt;
    auto recv_topk_weights = has_topk ?
        torch::empty({num_recv_tokens, num_topk}, torch::kFloat32) :
        std::nullopt;

    // Allocate metadata tensors
    auto rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, torch::kInt32);
    auto channel_prefix_matrix = torch::empty({num_ranks, num_channels}, torch::kInt32);
    // ... more metadata ...

    // ─────────────────────────────────────────────────────────────
    // 3. SELECT STREAM: Choose CUDA stream for kernels
    // ─────────────────────────────────────────────────────────────
    cudaStream_t stream = allocate_on_comm_stream ? comm_stream : at::cuda::getCurrentCUDAStream();

    // Wait for previous event if specified
    if (previous_event) {
        previous_event->current_stream_wait(stream);
    }

    // ─────────────────────────────────────────────────────────────
    // 4. LAUNCH NOTIFY KERNEL: Setup metadata and synchronize ranks
    // ─────────────────────────────────────────────────────────────

    // This kernel:
    // - Exchanges token counts between ranks
    // - Computes prefix sums for addressing
    // - Signals CPU with total receive count
    // - Synchronizes all ranks via barriers

    intranode::notify_dispatch(
        num_tokens_per_rank->data_ptr<int>(),
        moe_recv_counter_mapped,           // GPU→CPU signal
        num_ranks,
        num_tokens_per_expert->data_ptr<int>(),
        moe_recv_expert_counter_mapped,    // Per-expert counts to CPU
        num_experts,
        num_tokens,
        is_token_in_rank.data_ptr<bool>(),
        channel_prefix_matrix.data_ptr<int>(),
        rank_prefix_matrix.data_ptr<int>(),
        /* ... more parameters ... */
        buffer_ptrs_gpu,                   // Peer buffer pointers
        barrier_signal_ptrs_gpu,           // Barrier signals
        rank,
        stream,
        config.num_sms
    );

    // ─────────────────────────────────────────────────────────────
    // 5. CPU WAIT: Wait for GPU to signal receive count
    // ─────────────────────────────────────────────────────────────

    if (cached_num_recv_tokens <= 0) {
        // Busy-wait on host-mapped memory
        // GPU writes to moe_recv_counter_mapped, CPU polls it
        while (*moe_recv_counter == -1) {
            // Spin wait
        }
        num_recv_tokens = static_cast<int>(*moe_recv_counter);
        *moe_recv_counter = -1;  // Reset for next call

        // Resize receive tensors to actual size
        recv_x = recv_x.slice(0, 0, num_recv_tokens);
        if (recv_topk_idx) {
            recv_topk_idx = recv_topk_idx->slice(0, 0, num_recv_tokens);
            recv_topk_weights = recv_topk_weights->slice(0, 0, num_recv_tokens);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 6. LAUNCH DISPATCH KERNEL: Transfer data across ranks
    // ─────────────────────────────────────────────────────────────

    // This kernel:
    // - Reads local x, topk_idx, topk_weights
    // - Scatters data to peer GPU buffers via NVLink
    // - Uses channel-based parallelism for load balancing

    intranode::dispatch(
        recv_x.data_ptr(),                 // Receive buffer (local)
        use_fp8 ? recv_x_scales->data_ptr<float>() : nullptr,
        recv_src_idx.data_ptr<int>(),
        has_topk ? recv_topk_idx->data_ptr<topk_idx_t>() : nullptr,
        has_topk ? recv_topk_weights->data_ptr<float>() : nullptr,
        recv_channel_offset.data_ptr<int>(),
        send_head.data_ptr<int>(),
        x.const_data_ptr(),                // Send data (local)
        use_fp8 ? x_scales->const_data_ptr<float>() : nullptr,
        has_topk ? topk_idx->const_data_ptr<topk_idx_t>() : nullptr,
        has_topk ? topk_weights->const_data_ptr<float>() : nullptr,
        is_token_in_rank.const_data_ptr<bool>(),
        channel_prefix_matrix.data_ptr<int>(),
        num_tokens,
        num_worst_tokens,
        hidden_int4,
        num_topk,
        num_experts,
        num_scales,
        /* ... */
        buffer_ptrs_gpu,                   // Peer buffers for writing
        rank,
        num_ranks,
        stream,
        config.num_sms,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens
    );

    // ─────────────────────────────────────────────────────────────
    // 7. FINALIZE: Create event and return
    // ─────────────────────────────────────────────────────────────

    std::optional<EventHandle> event = std::nullopt;
    if (async) {
        event = EventHandle();
        event->record(stream);
    }

    // Get per-expert counts from GPU
    std::vector<int> num_recv_tokens_per_expert_list;
    if (num_tokens_per_expert) {
        // CPU reads from host-mapped memory
        for (int i = 0; i < num_experts / num_ranks; ++i) {
            while (moe_recv_expert_counter[i] == -1) {
                // Spin wait
            }
            num_recv_tokens_per_expert_list.push_back(moe_recv_expert_counter[i]);
            moe_recv_expert_counter[i] = -1;
        }
    }

    return std::make_tuple(
        recv_x,
        recv_x_scales,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_offset,
        send_head,
        recv_src_idx,
        event
    );
}
```

---

### Step 3: Notify Dispatch Kernel

**Purpose**: Exchange metadata and synchronize all ranks before data transfer

**File**: [kernels/intranode.cu:11-112](../csrc/kernels/intranode.cu#L11-L112)

**Kernel Signature:**

```cuda
template <int kNumRanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank,        // [num_ranks] - input
    int* moe_recv_counter_mapped,          // [1] - output to CPU
    const int* num_tokens_per_expert,      // [num_experts] - input
    int* moe_recv_expert_counter_mapped,   // [num_local_experts] - output to CPU
    int num_experts,
    int num_tokens,
    int num_channels,
    const bool* is_token_in_rank,          // [num_tokens, num_ranks] - input
    int* channel_prefix_matrix,            // [num_ranks, num_channels] - output
    int* rank_prefix_matrix_copy,          // [num_ranks, num_ranks] - output
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,                    // Peer buffer pointers
    int** barrier_signal_ptrs,             // Barrier signals
    int rank)
```

**Grid/Block Configuration:**

```
Grid: 1 + num_ranks blocks
  Block 0: Coordination block (handles barriers, aggregation)
  Blocks 1..num_ranks: Per-rank workers (compute channel metadata)

Threads per block: 128
```

**Kernel Flow:**

```cuda
// ═══════════════════════════════════════════════════════════════
// BLOCK 0: Coordination Block
// ═══════════════════════════════════════════════════════════════
if (sm_id == 0) {
    // ───────────────────────────────────────────────────────────
    // Step 1: Initial Barrier
    // ───────────────────────────────────────────────────────────
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);
    // All ranks synchronized before proceeding

    // ───────────────────────────────────────────────────────────
    // Step 2: Write Local Counts to Peer Buffers
    // ───────────────────────────────────────────────────────────
    int* per_rank_buffer;   // Buffer at buffer_ptrs[thread_id]
    int* per_expert_buffer; // Offset into same buffer

    if (thread_id < kNumRanks) {
        per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
        per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;

        // Write: "Rank `rank` is sending X tokens to rank `thread_id`"
        per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];

        // Write per-expert counts
        int num_experts_per_rank = num_experts / kNumRanks;
        for (int i = 0; i < num_experts_per_rank; ++i) {
            per_expert_buffer[rank * num_experts_per_rank + i] =
                num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }
    }

    // ───────────────────────────────────────────────────────────
    // Step 3: Second Barrier (wait for all writes)
    // ───────────────────────────────────────────────────────────
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Now all ranks have written their counts to all peer buffers
    // Each rank can read its own buffer to see counts from all peers

    // ───────────────────────────────────────────────────────────
    // Step 4: Read Local Buffer and Compute Prefix Sums
    // ───────────────────────────────────────────────────────────
    auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);

    // Each thread handles one destination rank
    if (thread_id < kNumRanks) {
        // Compute prefix sum: how many tokens from ranks 0..i
        for (int i = 1; i < kNumRanks; ++i) {
            local_per_rank_buffer[i * kNumRanks + thread_id] +=
                local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
        }

        // Signal CPU with total receive count for this rank
        if (thread_id == rank) {
            *moe_recv_counter_mapped =
                local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }
    }

    // ───────────────────────────────────────────────────────────
    // Step 5: Compute Per-Expert Receive Counts
    // ───────────────────────────────────────────────────────────
    auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
    int num_experts_per_rank = num_experts / kNumRanks;

    if (thread_id < num_experts_per_rank) {
        int sum = 0;
        for (int i = 0; i < kNumRanks; ++i) {
            sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
        }
        // Align to expert_alignment
        sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;

        // Signal CPU
        moe_recv_expert_counter_mapped[thread_id] = sum;
    }

    __syncthreads();

    // ───────────────────────────────────────────────────────────
    // Step 6: Copy Results and Clean Buffers
    // ───────────────────────────────────────────────────────────
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads) {
        rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];
    }

    // Zero out buffer space for communication queues
    for (int i = thread_id; i < num_memset_int; i += num_threads) {
        local_per_expert_buffer[i] = 0;
    }

    // ───────────────────────────────────────────────────────────
    // Step 7: Final Barrier
    // ───────────────────────────────────────────────────────────
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    // All ranks ready for dispatch kernel
}

// ═══════════════════════════════════════════════════════════════
// BLOCKS 1..num_ranks: Per-Rank Channel Workers
// ═══════════════════════════════════════════════════════════════
else {
    int dst_rank = sm_id - 1;  // Which rank this block is responsible for

    // ───────────────────────────────────────────────────────────
    // Compute Per-Channel Token Counts
    // ───────────────────────────────────────────────────────────
    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
        // Get token range for this channel
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id,
                              token_start_idx, token_end_idx);

        // Count tokens going to dst_rank in this channel
        int count = 0;
        for (int i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
            count += is_token_in_rank[i * kNumRanks + dst_rank];
        }

        // Warp-level reduction
        count = warp_reduce_sum(count);

        // One thread writes result
        if (elect_one_sync()) {
            channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
    }

    __syncthreads();

    // ───────────────────────────────────────────────────────────
    // Compute Channel Prefix Sum
    // ───────────────────────────────────────────────────────────
    if (thread_id == 0) {
        for (int i = 1; i < num_channels; ++i) {
            channel_prefix_matrix[dst_rank * num_channels + i] +=
                channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}
```

**Example Execution** (4 ranks, 10 channels):

**Before Barrier 1:**
```
Rank 0: num_tokens_per_rank = [5, 3, 7, 2]
Rank 1: num_tokens_per_rank = [4, 6, 1, 8]
Rank 2: num_tokens_per_rank = [2, 7, 5, 3]
Rank 3: num_tokens_per_rank = [6, 2, 4, 5]
```

**After Step 2 (writes to peer buffers):**
```
Rank 0's buffer (buffer_ptrs[0]):
  per_rank_buffer[0][0..3] = [5, 3, 7, 2]  ← Written by Rank 0
  per_rank_buffer[1][0..3] = [4, 6, 1, 8]  ← Written by Rank 1
  per_rank_buffer[2][0..3] = [2, 7, 5, 3]  ← Written by Rank 2
  per_rank_buffer[3][0..3] = [6, 2, 4, 5]  ← Written by Rank 3

(All ranks have similar buffers with same data)
```

**After Step 4 (prefix sums):**
```
Rank 0's local_per_rank_buffer (now contains prefix sums):
  Row 0: [5, 3, 7, 2]
  Row 1: [9, 9, 8, 10]   ← Cumulative sums
  Row 2: [11, 16, 13, 13]
  Row 3: [17, 18, 17, 18]
          ↑
  Column 0 shows: Rank 0 receives 17 total tokens
                  (5 from rank 0, 4 from rank 1, 2 from rank 2, 6 from rank 3)
```

**After Blocks 1-4 (channel computation):**
```
channel_prefix_matrix (each row = one destination rank):
  Rank 0: [1, 3, 5, 8, 10, 11, 13, 14, 16, 17]  ← Prefix sums per channel
  Rank 1: [2, 4, 6, 9, 11, 13, 15, 17, 18, 18]
  Rank 2: [1, 2, 4, 6, 8, 10, 12, 14, 16, 17]
  Rank 3: [2, 3, 5, 7, 9, 11, 13, 15, 17, 18]
```

This tells the dispatch kernel: "Channel 0 sends tokens 0-1 to rank 0, channel 1 sends tokens 1-3, etc."

---

This document is getting quite long. Let me save it and continue with the dispatch kernel details and combine operation in the next section.

