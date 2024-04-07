import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]
pruner = sys.argv[5]
instruct = sys.argv[6]
prune_n, prune_m = int(sys.argv[7]), int(sys.argv[8])
model_size = "xl"

if pruner == "dsnot":
    method = "blipt5_dsnot_pruner"
    sparsity_ratio_granularity = "none"
    score_method = "obd_avg"
elif pruner == "wanda":
    method = "blipt5_wanda_pruner"
    sparsity_ratio_granularity = "none"
    score_method = "obd_avg"
elif pruner == "zeroth":
    method = "blipt5_wanda_pruner"
    # zeroth-order gradient
    sparsity_ratio_granularity = "block"
    score_method = "olmezo-gradient_sum"
elif pruner == "first":
    method = "blipt5_wanda_pruner"
    # first-order gradient. 
    sparsity_ratio_granularity = "block"
    score_method = "aobd_sum"
elif pruner == "sparsegpt":
    method = "blipt5_sparsegpt_pruner"
    sparsity_ratio_granularity = "none"
    score_method = "obd_avg"

if instruct == "true":
    prune_cfg_path = "lavis/projects/blip2/eval/prune_stage2_t5_instruct.yaml"
else:
    prune_cfg_path = "lavis/projects/blip2/eval/prune_stage2.yaml"

pretrain="continue_stage2"

nproc_per_node = len(str(GPU).split(','))
initial_method = "wanda"

t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"

port = str(int(port) + int(10 * (2 * t5_ratio + 3 * vit_ratio)) + random.randint(0, 100))
job_id = f"{pretrain}-{model_size}-{pruner}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{model_size}-{pruner}_{prune_n}:{prune_m}"

program = (
    f"python -m torch.distributed.run"
    f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
    f" --cfg-path {prune_cfg_path}"
    f" --pruning_method '{method}' --save_pruned_model"
    f" --initial_method {initial_method}"
    f" --prune"
    f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
    f" --prune_m {prune_m} --prune_n {prune_n}"
    f" --model_size '{model_size}'"
    f" --score_method {score_method} --sparsity_ratio_granularity {sparsity_ratio_granularity}"
)

print(program)
subprocess.call(program, shell=True) # TODO prepare sparse models

for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval", "ret_flickr_eval"]:

    if instruct == "true":
        task = task.replace("_eval", "_instruct_eval")
        
    vit_pruned_checkpoint = f"pruned_checkpoint/V+L/{method}/{job_id}.pth"
    t5_pruned_checkpoint = f"pruned_checkpoint/V+L/{method}/{job_id}.pth"

    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruning_method '{method}'"
        f" --t5_pruned_checkpoint {t5_pruned_checkpoint}"
        f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
        f" --job_id '{job_id}'"
        f" --model_size '{model_size}'"
    )

    print(program)
    subprocess.call(program, shell=True)
