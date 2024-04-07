import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]
pruner = sys.argv[10]
nproc_per_node = len(str(GPU).split(','))

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

t5_model_prefix = "llm_model"
initial_method = "wanda" # "sparsegpt", "wanda", "magnitude"

t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
kl_weight = float(sys.argv[5])
model_size = sys.argv[9]

if vit_ratio < 1.0 and t5_ratio < 1.0:
    tune_opt = "LV"
elif t5_ratio >= 1.0:
    tune_opt = "V"
elif vit_ratio >= 1.0:
    tune_opt = "L"

tune_opt = sys.argv[11]
lora_r_v, lora_r_l, lora_r_q = sys.argv[12], sys.argv[13], sys.argv[14]
instruct = True
peft_postfix = ".model"
task = "okvqa_zeroshot_flant5xl_eval"

task = task.replace("flant5xl", "vicuna_instruct")
prune_cfg_path = "lavis/projects/blip2/eval/prune_stage2_vicuna_instruct.yaml"
pretrain="continue_stage2_vicuna_instruct"

iterations = 5
t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"
port = str(int(port) + int(100 * (t5_ratio + vit_ratio)) + random.randint(100, 200))

prune_n, prune_m = 0, 0
prune_n, prune_m = int(sys.argv[6]), int(sys.argv[7])
max_train_samples = int(sys.argv[8])

t5_lora_target_modules = ".q_proj, .k_proj, .v_proj, .o_proj, .gate_proj, .down_proj, .up_proj"
vit_lora_target_modules = ".qkv, .proj, .fc1, .fc2"
qformer_lora_target_modules = ".query, .key, .value, .dense"
remain_grads = ""
num_data = 128

job_id = f"{pretrain}-{method}_{kl_weight}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{method}_{kl_weight}_{prune_n}:{prune_m}"
job_id = job_id + "_" + tune_opt + "_" + remain_grads + "_" + \
            str(int(max_train_samples)) + f"_{lora_r_v}_{lora_r_l}_{lora_r_q}"

program = (
    f"python -m torch.distributed.run"
    f" --nproc_per_node={nproc_per_node} --master_port {port} train.py"
    f" --cfg-path lavis/projects/blip2/train/{pretrain.lower()}.yaml"
    f" --prune-cfg-path {prune_cfg_path}"
    f" --eval-cfg-path lavis/projects/blip2/eval/{task}.yaml"
    f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
    f" --t5_lora_target_modules '{t5_lora_target_modules}'"
    f" --vit_lora_target_modules '{vit_lora_target_modules}'"
    f" --qformer_lora_target_modules '{qformer_lora_target_modules}'"
    f" --max_train_samples {int(max_train_samples)}"
    f" --tune_opt {tune_opt}"
    f" --peft_postfix {peft_postfix}"
    f" --remain_grads '{remain_grads}'"
    f" --pruning_method '{method}' --t5_model_prefix '{t5_model_prefix}'"
    f" --save_pruned_model" 
    f" --initial_method {initial_method}"
    f" --prune" 
    f" --train"
    f" --sparse"
    f" --lora_r_v {lora_r_v} --lora_r_l {lora_r_l} --lora_r_q {lora_r_q}"
    f" --num_data {num_data}"
    f" --prune_m {prune_m} --prune_n {prune_n}"
    f" --model_size {model_size}"
    f" --kl_weight {kl_weight}"
    f" --score_method {score_method} --sparsity_ratio_granularity {sparsity_ratio_granularity}"
)

print(program)
subprocess.call(program, shell=True) # TODO prepare sparse models
