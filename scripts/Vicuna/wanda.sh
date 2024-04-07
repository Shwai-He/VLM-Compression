ratio=0.2
t5_ratio=$ratio
vit_ratio=$ratio


cd ./LAVIS
python ./scripts/Vicuna/wanda.py 0,1 12344 $t5_ratio $vit_ratio
