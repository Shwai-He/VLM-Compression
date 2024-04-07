#!/usr/bin/bash

#SBATCH --job-name=moe-p
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
# SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=spot
# SBATCH --quotatype=auto

# SBATCH --nodes=1
# SBATCH --gres=gpu:0

# cd /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS
source ~/anaconda3/bin/activate smoe

#python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_flickr.py # failed

#python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_msrvtt.py # Failed to merging datasets. Aborting.

# python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_vqa.py # failed

#python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_gqa.py # failed

#python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_coco.py # Failed to download or extracting datasets. Aborting.

# python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_vg.py

# python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_sbu.py

python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_nocaps.py

# python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_msvd.py

# python /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS/lavis/datasets/download_scripts/download_didemo.py
