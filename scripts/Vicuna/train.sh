ratio=0.5
t5_ratio=$ratio
vit_ratio=$ratio
tune_opt=QLV

kl_weight=0.1

prune_n=0
prune_m=0

#prune_n=2
#prune_m=4

#prune_n=4
#prune_m=8

lora_r_v=4
lora_r_l=8 
lora_r_q=2 

max_train_samples=10000
max_train_samples=20000

model_size="7b"
pruner=wanda

cd ./LAVIS

python ./scripts/Vicuna/train.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $max_train_samples $model_size $pruner $tune_opt $lora_r_v $lora_r_l $lora_r_q
python ./scripts/Vicuna/evaluate_new.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $max_train_samples $model_size $pruner $tune_opt $lora_r_v $lora_r_l $lora_r_q