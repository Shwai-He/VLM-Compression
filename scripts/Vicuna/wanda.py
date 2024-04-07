import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]
t5_model_prefix = "llm_model"
model_size = "7b"

prune_cfg_path = "lavis/projects/blip2/eval/prune_stage2_vicuna_instruct.yaml"
pretrain="continue_stage2"

nproc_per_node = len(str(GPU).split(','))
method = "blipt5_wanda_pruner"

t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"

prune_n, prune_m = 0, 0
# prune_n, prune_m = 2, 4
# prune_n, prune_m = 4, 8

port = str(int(port) + int(10 * (2 * t5_ratio + 3 * vit_ratio)) + random.randint(0, 100))
job_id = f"{pretrain}-{model_size}-{method}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{model_size}-{method}_{prune_n}:{prune_m}"

program = (
    f"python -m torch.distributed.run"
    f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
    f" --cfg-path {prune_cfg_path}"
    f" --pruning_method '{method}' --save_pruned_model"
    f" --prune"
    f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
    f" --prune_m {prune_m} --prune_n {prune_n}"
    f" --t5_model_prefix {t5_model_prefix}"
    f" --model_size '{model_size}'"
)

print(program)
subprocess.call(program, shell=True) # TODO prepare sparse models

for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval", "ret_flickr_eval"]:

    task = task.replace("flant5xl", "vicuna_instruct")
        
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
