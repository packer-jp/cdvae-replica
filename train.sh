#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g gc64
module load cuda/11.1
source venv/bin/activate
pip install -e .
HYDRA_FULL_ERROR=1 python cdvae/run.py data=perov expname=perov