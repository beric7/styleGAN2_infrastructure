#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -t 6-00:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate stylegan2-pytorch

cd $PBS_O_WORKDIR
cd ~/styleGAN2/stylegan2-ada-pytorch-main/

python train.py --data 'images/preprocessed_150k_512x512/' --outdir 'output/150k/v2/' --gpus=2

exit
