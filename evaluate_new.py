import argparse
import random
import time
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from lavis.compression import load_pruner
import sys

sys.path = [os.path.join(os.getcwd(), "lavis/peft/src/")] + sys.path


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument("--cfg-path",
                        default=None, help="path to configuration file.")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="The id of the Job"
    )

    parser.add_argument(
        "--num_data", type=int, default=128
    )

    parser.add_argument(
        "--power", type=int, default=2
    )
    
    parser.add_argument(
        "--t5_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_pruned_checkpoint", type=str, default=None
    )
    parser.add_argument(
        "--qformer_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--t5_prune_spec", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_prune_spec", type=str, default=None
    )

    parser.add_argument(
        "--pruning_method", type=str, default="blipt5_wanda_pruner",
    )
    
    parser.add_argument(
        "--save_pruned_model", action="store_true"
    )
    
    parser.add_argument(
        "--sparsity_ratio_granularity",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--max_sparsity_per_layer", type=float, default=1.01
    )
    
    parser.add_argument(
        "--score_method",
        type=str,
        default="obd_avg",
    )
    
    parser.add_argument(
        "--num_data_first_stage", type=int, default=32
    )
    
    parser.add_argument(
        "--num_noise", default=1, type=int,
    )
    
    parser.add_argument(
        "--sparsity_dict",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--prune_per_model",
        action="store_true"
    )
    
    parser.add_argument(
        "--is_global",
        action="store_true"
    )
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prunining_dataset_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--without_DSnoT",
        action="store_true",
    )
    parser.add_argument(
        "--initial_method", 
        default="wanda", 
        choices=["sparsegpt", "wanda", "magnitude"], 
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_final_activations(args, cfg, task, model, datasets):
    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    start = time.time()

    print("Start to get final activation")
    outputs = runner.get_last_activations(num_data=args.num_data, power=args.power)

    end = time.time()
    print(f"Finish get final activation, using {end - start:.3f}s")

    return outputs


def main():

    args = parse_args()
    print(args)
    print(f"vit_prune_spec: {args.vit_prune_spec}, t5_prune_spec: {args.t5_prune_spec}")

    if args.job_id is not None:
        job_id = args.job_id
    else:
        job_id = now()

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # TODO set after init_distributed_mode() to only log on master.
    setup_logger()
    if "vicuna" in cfg.model_cfg.model_type and "b" in args.model_size:
        cfg.model_cfg.model_type = "vicuna" + args.model_size
        cfg.model_cfg.llm_model = "lmsys/vicuna-" + args.model_size + "-v1.1"
        cfg.model_cfg.pretrained = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna" + args.model_size + "_trimmed.pth"

    elif "t5" in cfg.model_cfg.model_type and "xl" in args.model_size: 
        cfg.model_cfg.model_type = "pretrain_flant5" + args.model_size
        cfg.model_cfg.t5_model = "google/flan-t5-" + args.model_size
        if "instruct" in cfg.model_cfg.arch:
            cfg.model_cfg.pretrained = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flan" + args.model_size + "_trimmed.pth"
        else:
            cfg.model_cfg.pretrained = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5" + args.model_size + ".pth"
    
    if args.model_size in ["13b", "xxl"]:
        cfg.run_cfg.batch_size_eval = cfg.run_cfg.batch_size_eval // 2
    
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
        
    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    # TODO load t5_model
    if args.t5_pruned_checkpoint is not None and getattr(model, "t5_model", None) is not None:
        print("Load t5 pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("t5_model") and "lora" not in k and "mask" not in k}
        prune_state_dict = {k.replace("t5_model.", ""): v for k, v in prune_state_dict.items()}
        prune_state_dict = {k.replace("base_model.model.", "").replace("base_model.Model.", ""): v for k, v in prune_state_dict.items()}
        model.t5_model.load_state_dict(prune_state_dict)
        
    elif args.t5_pruned_checkpoint is not None and getattr(model, "opt_model", None) is not None:
        print("Load opt_model pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("opt_model") and "lora" not in k and "mask" not in k}
        prune_state_dict = {k.replace("opt_model.", ""): v for k, v in prune_state_dict.items()}
        prune_state_dict = {k.replace("base_model.model.", ""): v for k, v in prune_state_dict.items()}
        model.opt_model.load_state_dict(prune_state_dict)
    
    elif args.t5_pruned_checkpoint is not None and getattr(model, "llm_model", None) is not None:
        print("Load llm_model pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("llm_model") and "lora" not in k and "mask" not in k}
        prune_state_dict = {k.replace("llm_model.", ""): v for k, v in prune_state_dict.items()}
        prune_state_dict = {k.replace("base_model.model.", ""): v for k, v in prune_state_dict.items()}
        model.llm_model.load_state_dict(prune_state_dict)
        
    # TODO load vit
    if args.vit_pruned_checkpoint is not None:
        print("Load vit pruned weight")
        prune_state_dict = torch.load(args.vit_pruned_checkpoint, map_location="cpu")
        model_prefix = None
        for candidate_prefix in ["visual.", "visual_encoder."]:
            if any(k.startswith(candidate_prefix) for k in prune_state_dict.keys()):
                model_prefix = candidate_prefix
                break
            
        assert model_prefix is not None
        
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith(model_prefix) and "lora" not in k and "mask" not in k}
        print(f"VIT checkpoint prefix: {model_prefix}")
        
        prune_state_dict = {k.replace(model_prefix, ""): v for k, v in prune_state_dict.items()}
        prune_state_dict = {k.replace("base_model.model.", ""): v for k, v in prune_state_dict.items()}
        original_state_dict = model.visual_encoder.state_dict()
        
        for k, v in prune_state_dict.items():
            if k in original_state_dict:
                original_state_dict[k] = v
                
        prune_state_dict = original_state_dict
        from lavis.models.eva_vit import interpolate_pos_embed
        interpolate_pos_embed(model.visual_encoder, prune_state_dict)
        model.visual_encoder.load_state_dict(prune_state_dict)
        

    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
    
    print(f"Remaining Proportion: {distilled_total_size / orig_total_size * 100}%")
                
    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets, 
    )

    runner.orig_total_size = orig_total_size
    runner.distilled_total_size = distilled_total_size
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":

    main()
