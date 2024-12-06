#!/bin/bash
# MASTER_ADDR=localhost
# MASTER_PORT=12345
# NNODES=1
# NODE_RANK=$1
# GPUS_PER_NODE=8

MASTER_ADDR=11.11.3.3
MASTER_PORT=12345
NNODES=2
NODE_RANK=$1
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS dis_test.py
