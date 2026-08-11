#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=1-5
#SBATCH --output=coco_results/logs/%x-%a_%A.out

ALGO=$1
DIM=40
F_ID=$2

module purge

source /path/to/.venv/bin/activate

python /path/to/coco_case_study.py ${ALGO} ${F_ID} ${DIM} ${SLURM_ARRAY_TASK_ID} cpu