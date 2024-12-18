# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12346 \
autoregressive/train/train_c2i_fsdp.py "$@"
