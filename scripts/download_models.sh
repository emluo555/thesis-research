#!/bin/sh
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir /scratch/gpfs/ZHUANGL/el5267/thesis-research/models/models--meta-llama-Llama-3.2-11B-Vision-Instruct \
  --local-dir-use-symlinks False