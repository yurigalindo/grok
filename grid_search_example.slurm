#!/bin/bash

#SBATCH --job-name=gridsearch_grokking
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=01:59:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=@nyu.edu
#SBATCH -c 1

#SBATCH --array=1-12

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "

source /ext3/env.sh

python -u ./grid_search_example.py $SLURM_ARRAY_TASK_ID

"
