ratio=0.5
t5_ratio=$ratio
vit_ratio=$ratio
instruct=true

prune_n=0
prune_m=0

# prune_n=2
# prune_m=4

# prune_n=4
# prune_m=8

cd ./LAVIS
python scripts/T5/dsnot.py 0,1 11341 $t5_ratio $vit_ratio $instruct $prune_n $prune_m
