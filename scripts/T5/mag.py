import subprocess
import random

import sys
import os

GPU = sys.argv[1]
port = sys.argv[2]


nproc_per_node = len(str(GPU).split(','))
method = "blipt5_mag_pruner"
prune_per_model = False

t5_ratio = float(sys.argv[3])
budget = float(sys.argv[4])
t5_ratios = f"{t5_ratio}-1.0-1.0"

vit_ratio = round(budget - t5_ratio, 1)
vit_ratios = f"{vit_ratio}-1.0-1.0"

port = str(int(port) + int(10 * (2 * t5_ratio + 3 * vit_ratio)) + random.randint(0, 100))
job_id = f"cc3m-{method}_{t5_ratio}_{vit_ratio}_pure"

max_train_samples = 128
adapter_name = "none"
t5_lora_target_modules = ".q, .k, .v, .o, .wi_0, wi_1, wo"
vit_lora_target_modules = ".qkv, .proj, .fc1, .fc2"

program = (
    f"python -m torch.distributed.run"
    f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_new.py"
    f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
    f" --pruning_method '{method}' --save_pruned_model"
    f" --prune_per_model"
    f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
    f" --adapter_name {adapter_name}"
    f" --t5_lora_target_modules '{t5_lora_target_modules}'"
    f" --vit_lora_target_modules '{vit_lora_target_modules}'"
    f" --without_DSnoT"
)

print(program)
subprocess.call(program, shell=True) # TODO prepare sparse models

save_dir = f"pruned_checkpoint/V+L/"
if not os.path.exists(os.path.join(save_dir, method)): 
    os.mkdir(os.path.join(save_dir, method))
    
for task in ["vqav2_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "okvqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "ret_flickr_eval"]:

    t5_ratios = f"{t5_ratios}-1.0-1.0"
    vit_ratios = f"{vit_ratios}-1.0-1.0"

    job_id = f"cc3m-{method}_{t5_ratio}_{vit_ratio}"

    vit_pruned_checkpoint = os.path.join(save_dir, method, f"{job_id}.pth")
    t5_pruned_checkpoint = os.path.join(save_dir, method, f"{job_id}.pth")

    program = (
        f"python -m torch.distributed.run"
        f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_new.py"
        f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
        f" --pruning_method '{method}'"
        f" --t5_pruned_checkpoint {t5_pruned_checkpoint}"
        f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
        f" --job_id '{job_id}'"
        f" --adapter_name {adapter_name}"
        f" --t5_lora_target_modules '{t5_lora_target_modules}'"
        f" --vit_lora_target_modules '{vit_lora_target_modules}'"
    )

    print(program)
    subprocess.call(program, shell=True)
