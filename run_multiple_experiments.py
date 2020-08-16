# ! /bin/env python
import os
import subprocess

scenario = 'simple_speaker_listener'
OUTPUT_DIR = os.getcwd() + "/output"

if os.path.exists(OUTPUT_DIR):
    import shutil
    shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
else:
    os.mkdir(OUTPUT_DIR)

GPU_PER_JOB = 1
CPU_PER_JOB = 4
POOL_SIZE = 1
ENVIRONMENT = None

import numpy as np

output_dirs = ['cyclegan_bn', 'cyclegan_in']
uses_bn = ['True', 'False']

for i in range(2):
    with open('.submit', 'w') as outF:
        outF.write('# !/bin/sh\n')
        outF.write('# BSUB -q normal\n')
        outF.write('# BSUB -n {}\n'.format(CPU_PER_JOB))
        outF.write('# BSUB -J a.fritzler\n')
        if GPU_PER_JOB:
            outF.write('# BSUB -gpu "num={}:mode=shared:j_exclusive=yes"\n'.format(GPU_PER_JOB))
        outF.write('# BSUB -o %J.out\n')
        outF.write('# BSUB -e %J.err\n')
        outF.write("export OMP_NUM_THREADS=1\n")
        outF.write("export MKL_NUM_THREADS=1\n")
        if ENVIRONMENT:
            outF.write('. activate {}\n'.format(ENVIRONMENT))
        folder = os.getcwd()
        outF.write(
                f'~/anaconda3/bin/python train.py --dataroot datasets/horse2zebra/ --cuda --output_dir {output_dirs[i]} --use_bn {uses_bn[i]} &\n')

#    subprocess.call('bsub', stdin=open('.submit'))
