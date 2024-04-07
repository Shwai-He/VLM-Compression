t5_ratio=0.5
vit_ratio=0.5

cd /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS
python scripts/T5/ecoflap_zeroth.py 0,1 10341 $t5_ratio $vit_ratio
