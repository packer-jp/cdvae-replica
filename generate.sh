#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g gc64
module load cuda/11.1
source venv/bin/activate
pip install -e .
python scripts/evaluate.py --model_path /work/04/gc64/c64080/materials/cdvae-replica/hydra/singlerun/2023-11-28/perov --tasks recon gen