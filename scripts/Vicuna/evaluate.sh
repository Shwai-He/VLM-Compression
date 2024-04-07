t5_ratio=0.5
vit_ratio=0.5
kl_weight=0.001
prune_n=0
prune_m=0
iterations=1
max_train_samples=25000


cd ./LAVIS
python ./scripts/Vicuna/evaluate.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $iterations $max_train_samples