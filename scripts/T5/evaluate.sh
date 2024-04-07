ratio=0.5
t5_ratio=$ratio
vit_ratio=$ratio
kl_weight=0.1
prune_n=0
prune_m=0
iterations=1
max_train_samples=25000
instruct=false
model_size=xl
pruner=wanda


cd ./LAVIS
python ./scripts/T5/evaluate.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $iterations $max_train_samples $instruct $model_size $pruner