#!/bin/bash


docker run \
    -it \
    -d \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    -v $(pwd):/nanosam \
    nanosam:23-01