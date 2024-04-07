import subprocess
import random
import os
import sys

GPU = sys.argv[1]
port = sys.argv[2]
prune_n, prune_m = int(sys.argv[5]), int(sys.argv[6])
t5_model_prefix = "llm_model"

pretrain="continue_stage2"
prune_cfg_path = "lavis/projects/blip2/eval/prune_stage2_vicuna_instruct.yaml"

nproc_per_node = len(str(GPU).split(','))
method = "blipt5_dsnot_pruner"

t5_ratio = float(sys.argv[3])
t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratio = float(sys.argv[4])
vit_ratios = f"{vit_ratio}-1.0-1.0"

without_DSnoT = False
initial_method = "wanda" # "sparsegpt", "wanda", "magnitude"

port = str(int(port) + int(10 * (12 * t5_ratio + 13 * vit_ratio)) + random.choice(range(100)))
job_id = f"{pretrain}-{method}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{method}_{prune_n}:{prune_m}"


if without_DSnoT: 
    port = str(int(port) - 111)
    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
        f" --cfg-path {prune_cfg_path}"
        f" --pruning_method '{method}' --save_pruned_model" 
        f" --without_DSnoT"
        f" --initial_method {initial_method}"
        f" --prune"
        f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
        f" --prune_m {prune_m} --prune_n {prune_n}"
        f" --t5_model_prefix {t5_model_prefix}"
    )

else: 
    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
        f" --cfg-path {prune_cfg_path}"
        f" --pruning_method '{method}' --save_pruned_model" 
        f" --prune"
        f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
        f" --prune_m {prune_m} --prune_n {prune_n}"
        f" --t5_model_prefix {t5_model_prefix}"
    )

print(program)  
subprocess.call(program, shell=True) # TODO prepare sparse models

save_dir = f"pruned_checkpoint/V+L/"
if not os.path.exists(os.path.join(save_dir, method)): 
    os.mkdir(os.path.join(save_dir, method))
        
for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval", "ret_flickr_eval"]:

    task = task.replace("flant5xl", "vicuna_instruct")

    vit_pruned_checkpoint = os.path.join(save_dir, method, f"{job_id}.pth")
    t5_pruned_checkpoint = os.path.join(save_dir, method, f"{job_id}.pth")

    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruning_method '{method}'"
        f" --t5_pruned_checkpoint {t5_pruned_checkpoint}"
        f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
        f" --job_id '{job_id}'"
    )

    print(program)
    subprocess.call(program, shell=True)
