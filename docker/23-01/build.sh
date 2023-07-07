#!/bin/bash

docker build -t clip_distillation:23-01 -f $(pwd)/docker/23-01/Dockerfile $(pwd)/docker/23-01