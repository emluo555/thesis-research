#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/gpfs/ZHUANGL/el5267/thesis-research/slrm_log/train_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/el5267/thesis-research/slrm_log/train_%j.err
#SBATCH --constraint=a100
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=el5267@princeton.edu

source ~/.bashrc
conda activate nlp-new
cd /scratch/gpfs/ZHUANGL/el5267/thesis-research/

python inference_non_cot.py