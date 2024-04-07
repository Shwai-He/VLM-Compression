import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]
pruner = sys.argv[11]
cc3m = sys.argv[12]
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

t5_model_prefix = "t5_model"
initial_method = "wanda" # "sparsegpt", "wanda", "magnitude"

t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
kl_weight = float(sys.argv[5])

instruct = sys.argv[9]
model_size = sys.argv[10]
tune_opt = sys.argv[13]


task = "okvqa_zeroshot_flant5xl_eval"
if instruct == "true":
    prune_cfg_path = "lavis/projects/blip2/eval/cc_prefix_derivative_compute_t5_instruct.yaml"
    pretrain="continue_stage2_cc3m_t5_instruct"
    task = task.replace("_eval", "_instruct_eval")
else:
    prune_cfg_path = "lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
    pretrain="continue_stage2_cc3m"

t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"

port = str(int(port) + int(100 * (t5_ratio + vit_ratio)) + random.randint(100, 200))

t5_lora_target_modules = ".q, .k, .v, .o, .wi_0, wi_1, wo"
vit_lora_target_modules = ".qkv, .proj, .fc1, .fc2"
qformer_lora_target_modules = ".query, .key, .value, .dense"
remain_grads = ""

num_data = 128

# prune_n, prune_m = 4, 8
# prune_n, prune_m = 2, 4
# prune_n, prune_m = 0, 0
prune_n, prune_m = int(sys.argv[6]), int(sys.argv[7])
max_train_samples = int(sys.argv[8])

job_id = f"{pretrain}-{pruner}_{kl_weight}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{pruner}_{kl_weight}_{prune_n}:{prune_m}"
job_id = job_id + "_" + tune_opt + "_" + remain_grads + "_" + \
            str(int(max_train_samples))

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
    f" --remain_grads '{remain_grads}'"
    f" --pruning_method '{method}' --t5_model_prefix '{t5_model_prefix}'"
    f" --save_pruned_model" 
    f" --initial_method {initial_method}"
    f" --prune" 
    f" --train"
    f" --num_data {num_data}"
    f" --prune_m {prune_m} --prune_n {prune_n}"
    f" --model_size {model_size}"
    f" --kl_weight {kl_weight}"
    f" --score_method {score_method} --sparsity_ratio_granularity {sparsity_ratio_granularity}"
)

print(program)
subprocess.call(program, shell=True) # TODO prepare sparse models
