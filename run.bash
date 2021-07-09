#!/bin/bash
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=24:00:00
#SBATCH --qos=kostas-med
#SBATCH --partition=kostas-compute

source /home/shiyani/anaconda3/etc/profile.d/conda.sh
conda activate se3

cd /home/shiyani/se3posenets-pytorch

#python train_flow_se3_nets.py -c config/icra18final/simdata/flow/def.yaml

python train_se3posenets.py -c config/icra18final/simdata/se3pose/def_rmsprop.yaml
