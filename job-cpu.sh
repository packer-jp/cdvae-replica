#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gc64
source ~/.bashrc
conda activate cdvae-cpu
pip install -e .
python cdvae/run.py data=perov expname=perov