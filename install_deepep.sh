# TORCH_CUDA_ARCH_LIST="10.0a" python setup.py -v build
# sudo apt-get update
# sudo apt-get install -y rdma-core libibverbs1 libibverbs-dev ibverbs-providers libmlx5-1 libmlx5-dev

uv pip install nvidia-nvshmem-cu12
CUDA_ARCH="10.0a"
TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} python setup.py -v develop