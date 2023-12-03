#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g gc64
module load cuda/11.1
source venv/bin/activate
pip install -e .
python scripts/compute_metrics.py --root_path /work/04/gc64/c64080/materials/cdvae-replica/hydra/singlerun/2023-12-02/perov --tasks recon gen