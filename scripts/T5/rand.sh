t5_ratio=0.5
budget=1.0

cd ./LAVIS
python ./scripts/T5/rand.py 0,1 12344 $t5_ratio $budget
