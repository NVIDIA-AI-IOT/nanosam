#!/bin/bash


docker run \
    -it \
    -d \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    --device /dev/video0 \
    -v $(pwd):/clip_distillation \
    clip_distillation:23-01