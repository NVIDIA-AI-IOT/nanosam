#!/bin/bash


docker run \
    -it \
    -d \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    --device /dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -p 5000:5000 \
    -p 8888:8888 \
    -v $(pwd):/nanosam \
    nanosam:23-01