import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]
pruner = sys.argv[10]


nproc_per_node = len(str(GPU).split(','))
if pruner == "dsnot":
    method = "blipt5_dsnot_pruner"
elif pruner == "wanda":
    method = "blipt5_wanda_pruner"
elif pruner == "zeroth":
    method = "blipt5_wanda_pruner"
elif pruner == "first":
    method = "blipt5_wanda_pruner"
    
pretrain="continue_stage2_vicuna_instruct"


t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
kl_weight = float(sys.argv[5])
tune_opt = sys.argv[11]

t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"

port = str(int(port) + int(10 * (t5_ratio + vit_ratio)) + random.randint(100, 200))

t5_lora_target_modules = ".q, .k, .v, .o, .wi_0, wi_1, wo"
vit_lora_target_modules = ".qkv, .proj, .fc1, .fc2"
qformer_lora_target_modules = ".query, .key, .value, .dense"
remain_grads = ""

prune_n, prune_m = 0, 0
prune_n, prune_m = int(sys.argv[6]), int(sys.argv[7])
max_train_samples = int(sys.argv[8])
model_size = sys.argv[9]
lora_r_v, lora_r_l, lora_r_q = sys.argv[12], sys.argv[13], sys.argv[14]

job_id = f"{method}/{pretrain}-{method}_{kl_weight}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{method}/{pretrain}-{method}_{kl_weight}_{prune_n}:{prune_m}"
job_id = job_id + "_" + tune_opt + "_" + remain_grads + "_" + str(int(max_train_samples)) + f"_{lora_r_v}_{lora_r_l}_{lora_r_q}"
        

for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval", "ret_flickr_eval"]:
# for task in ["nocaps_flant5xl_eval"]:
    task = task.replace("flant5xl", "vicuna_instruct")
    t5_ratios = f"{t5_ratios}-1.0-1.0"
    vit_ratios = f"{vit_ratios}-1.0-1.0"

    vit_pruned_checkpoint = f"pruned_checkpoint/V+L/{job_id}.pth"
    t5_pruned_checkpoint = f"pruned_checkpoint/V+L/{job_id}.pth"
    qformer_pruned_checkpoint = f"pruned_checkpoint/V+L/{job_id}.pth"
    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_new.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruning_method '{method}'"
        f" --t5_pruned_checkpoint '{t5_pruned_checkpoint}'"
        f" --vit_pruned_checkpoint '{vit_pruned_checkpoint}'"
        f" --job_id '{job_id}'"
        f" --model_size {model_size}"
    )

    print(program)
    subprocess.call(program, shell=True)
