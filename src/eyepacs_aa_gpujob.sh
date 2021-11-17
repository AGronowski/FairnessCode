#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=175G
#SBATCH --account=def-hpcg1626
#SBATCH --qos=gpu
#SBATCH --time=240:00:0
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mail-type=begin          # send email when job ends
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=adam.gronowski@gmail.com

source ~/ENV/bin/activate


python test_eyepacs.py