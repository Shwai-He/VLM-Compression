import subprocess
import sys

GPU = sys.argv[1]
port = sys.argv[2]

pretrain="continue_stage2"
prune_cfg_path = "lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
nproc_per_node = len(str(GPU).split(','))
method = "blipt5_wanda_pruner"
sparsity_ratio_granularity = "block"
score_method = "aobd_sum"

t5_ratio = float(sys.argv[3])
vit_ratio = float(sys.argv[4])
t5_ratios = f"{t5_ratio}-1.0-1.0"
vit_ratios = f"{vit_ratio}-1.0-1.0"

port = str(int(port) + int(10 * (4 * t5_ratio + 3 * vit_ratio)))
max_sparsity_per_layer = 1.01

prune_n, prune_m = 2, 4
prune_n, prune_m = 4, 8
prune_n, prune_m = 0, 0

job_id = f"{pretrain}-{method}_{t5_ratio}_{vit_ratio}" if prune_n==0 else f"{pretrain}-{method}_{prune_n}:{prune_m}"

program = (
    f"python -m torch.distributed.run"
    f" --nproc_per_node={nproc_per_node} --master_port {port} evaluate_old.py"
    f" --cfg-path {prune_cfg_path}"
    f" --pruning_method '{method}' --save_pruned_model"
    f" --score_method {score_method}"
    f" --max_sparsity_per_layer {max_sparsity_per_layer}"
    f" --num_data_first_stage 128"
    f" --prune_per_model"
    f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
    f" --prune"
    f" --t5_prune_spec 24-{t5_ratios} --vit_prune_spec 39-{vit_ratios} --job_id '{job_id}'"
    f" --prune_m {prune_m} --prune_n {prune_n}"
)

print(program)
subprocess.call(program, shell=True)


for task in ["okvqa_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "vqav2_zeroshot_flant5xl_eval", "ret_flickr_eval"]:

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
    )

    print(program)
    subprocess.call(program, shell=True)
