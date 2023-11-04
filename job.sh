#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g gc64
source ~/.bashrc
module load cuda/11.8
conda activate cdvae
pip install -e .
python cdvae/run.py data=perov expname=perov