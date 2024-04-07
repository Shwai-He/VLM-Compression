t5_ratio=0.5
vit_ratio=0.5

cd /mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS
python scripts/T5/ecoflap_first.py 0,1 11341 $t5_ratio $vit_ratio
