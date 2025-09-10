#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="{job}_budget_highway"       # Name of the job

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:1                  # Request n gpus
#SBATCH --cpus-per-task=16            # number of cpus per task and per node

#SBATCH -w clair1,plato1,plato2,bruno1,bruno2,bruno3,bruno4,nlp-l40-1,nlp-l40-2,tdk-bm4

#SBATCH -o sbatch_runs/%N_%j_{job}_out.txt       # stdout goes here
#SBATCH -e sbatch_runs/%N_%j_{job}_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il

nvidia-smi
{python_cmd}