#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -t 6-00:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate stylegan2_ada

cd $PBS_O_WORKDIR
cd ~/styleGAN2/stylegan2-ada-pytorch-main/

python train.py --data 'images/preprocessed_512x512_no_culvert_mirror/' --outdir 'output/fine_tuned/' --snap 50 \
--resume './output/fine_tuned/00004--auto2-resumecustom/network-snapshot-004000.pkl' --gpus=2

exit
