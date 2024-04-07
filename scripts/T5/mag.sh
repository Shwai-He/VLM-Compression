t5_ratio=0.5
budget=1.0

cd /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS
python ./scripts/T5/mag.py 0,1 12344 $t5_ratio $budget
