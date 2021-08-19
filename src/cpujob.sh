#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH --qos=privileged
#SBATCH --time=24:00:0
#SBATCH --mail-type=begin          # send email when job ends
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=adam.gronowski@gmail.com

source ~/ENV/bin/activate

python main.py
