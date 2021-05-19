#!/bin/bash
docker rm python_processing -f
docker run -it --name=python_processing \
    --volume /$(pwd):/processing \
    -p 15200:15200 \
    python_processing:build \
    python test_preprocessing.py \