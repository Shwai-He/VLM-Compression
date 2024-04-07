ratio=0.5
t5_ratio=$ratio
vit_ratio=$ratio


cd ./LAVIS
python ./scripts/Vicuna/wanda.py 0,1 12344 $t5_ratio $vit_ratio
