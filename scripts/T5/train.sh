ratio=0.5
t5_ratio=$ratio
vit_ratio=$ratio
kl_weight=0.1
tune_opt=QLV

prune_n=0
prune_m=0

#prune_n=2
#prune_m=4

#prune_n=4
#prune_m=8

max_train_samples=25000

instruct=false
instruct=true
model_size=xl
pruner=wanda
cc3m=false

cd ./LAVIS
python ./scripts/T5/train.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $max_train_samples $instruct $model_size $pruner $cc3m $tune_opt
python ./scripts/T5/evaluate.py 0,1 12344 $t5_ratio $vit_ratio $kl_weight $prune_n $prune_m $max_train_samples $instruct $model_size $pruner $cc3m $tune_opt