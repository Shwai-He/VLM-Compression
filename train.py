import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cuda

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now
from lavis.compression import load_pruner
from lavis.runners import *

def print_gpu_memory_device(device=None):
    if device is None:
        device = cuda.current_device()
    used_memory = cuda.memory_allocated(device) // 1024 ** 2
    print(f"GPU {device} Used Memory: {used_memory}MB")

from lavis.peft.src.peft.tuners.lora import Linear, LoraLayer, Linear8bitLt

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path",
                        default=None, help="path to configuration file.")

    parser.add_argument("--prune-cfg-path",
                        default=None, help="path to configuration file.")

    parser.add_argument("--eval-cfg-path",
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
        "--lora_r_l",
        type=int,
        default=8,
    )
    
    parser.add_argument(
        "--lora_r_v",
        type=int,
        default=8,
    )
    
    parser.add_argument(
        "--lora_r_q",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--t5_lora_target_modules",  # TODO align with other language models
        type=str,
        default=None,
    )

    parser.add_argument(
        "--vit_lora_target_modules",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--qformer_lora_target_modules",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--remain_grads",
        type=str,
        default="",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--tune_opt",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--train", action="store_true"
    )

    parser.add_argument(
        "--evaluate", action="store_true"
    )

    # TODO configs for pruning.
    parser.add_argument(
        "--prune", action="store_true"
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
        "--t5_prune_spec", type=str, default="-1.0-1.0-1.0"
    )

    parser.add_argument(
        "--vit_prune_spec", type=str, default="-1.0-1.0-1.0"
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
        "--sparse",
        action="store_true",
    )

    parser.add_argument(
        "--prune_n",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--prune_m",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--t5_model_prefix",
        type=str,
        default="t5_model",
    )
    parser.add_argument(
        "--peft_postfix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.,
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


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    args = parse_args()

    args.t5_lora_target_modules = args.t5_lora_target_modules.split(', ') if args.t5_lora_target_modules is not None else None
    args.vit_lora_target_modules = args.vit_lora_target_modules.split(', ') if args.vit_lora_target_modules is not None else None
    args.qformer_lora_target_modules = args.qformer_lora_target_modules.split(', ') if args.qformer_lora_target_modules is not None else None
    remain_grads = args.remain_grads.split('-') if (args.remain_grads is not None and args.remain_grads != "") else []

    print(args)
    print(f"vit_prune_spec: {args.vit_prune_spec}, t5_prune_spec: {args.t5_prune_spec}")
    # print(f"without_DSnoT: {args.without_DSnoT}")

    _, vit_ratio_org, _, _ = args.vit_prune_spec.split("-")
    _, t5_ratio_org, _, _ = args.t5_prune_spec.split("-")

    if args.job_id is not None:
        job_id = args.job_id
    else:
        job_id = now()

    prune_job_id = job_id + "_prune"
    eval_job_id = job_id + "_eval"

    cfg = Config(args)
    cfg.run_cfg.max_epoch = 1
    cfg.run_cfg.weight_decay = args.weight_decay
    cfg.run_cfg.warmup_steps = args.warmup_steps
    init_distributed_mode(cfg.run_cfg)

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
        cfg.run_cfg.batch_size_train = cfg.run_cfg.batch_size_train // 4

    setup_seeds(cfg)

    # TODO set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    # TODO cfg for training. 
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    if args.t5_pruned_checkpoint is not None and getattr(model, args.t5_model_prefix, None) is not None:
        # TODO align with other language models. 
        print(f"Load {args.t5_model_prefix} pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("t5_model")}
        prune_state_dict = {k.replace(args.t5_model_prefix, ""): v for k, v in prune_state_dict.items()}
        model.t5_model.load_state_dict(prune_state_dict)

    if args.vit_pruned_checkpoint is not None:
        print("Load vit pruned weight")
        prune_state_dict = torch.load(args.vit_pruned_checkpoint, map_location="cpu")
        model_prefix = None
        for candidate_prefix in ["visual.", "visual_encoder."]:
            if any(k.startswith(candidate_prefix) for k in prune_state_dict.keys()):
                model_prefix = candidate_prefix
                break

        assert model_prefix is not None

        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith(model_prefix)}
        print(f"VIT checkpoint prefix: {model_prefix}")
        prune_state_dict = {k.replace(model_prefix, ""): v for k, v in prune_state_dict.items()}
        original_state_dict = model.visual_encoder.state_dict()

        for k, v in prune_state_dict.items():
            if k in original_state_dict:
                original_state_dict[k] = v

        prune_state_dict = original_state_dict
        from lavis.models.eva_vit import interpolate_pos_embed
        interpolate_pos_embed(model.visual_encoder, prune_state_dict)
        model.visual_encoder.load_state_dict(prune_state_dict)

    distilled_total_size = sum(
        (param != 0).float().sum() for name, param in model.named_parameters() if "lora_" not in name
    )

    print(f"{orig_total_size / 10 ** 9 :.3f} B, {distilled_total_size / 10 ** 9 :.3f} B")
    print(f"Original Remaining Proportion: {distilled_total_size / orig_total_size * 100}%")
    remain_grads.append("lora")
    # TODO Insert LoRA to large language models.
    if "t5_model" in args.t5_model_prefix:
        print("Insert LoRA to T5_model... ")
        t5_lora_config = LoraConfig(
            r=args.lora_r_l,
            lora_alpha=args.lora_alpha,
            target_modules=args.t5_lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.t5_model = get_peft_model(model.t5_model, t5_lora_config)
        model.t5_model.print_trainable_parameters()

    elif "opt_model" in args.t5_model_prefix:
        print("Insert LoRA to OPT_model... ")
        t5_lora_config = LoraConfig(
            r=args.lora_r_l,
            lora_alpha=args.lora_alpha,
            target_modules=args.t5_lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.opt_model = get_peft_model(model.opt_model, t5_lora_config)
        model.opt_model.print_trainable_parameters()

    elif "llm_model" in args.t5_model_prefix:
        print("Insert LoRA to LLM_model... ")
        t5_lora_config = LoraConfig(
            r=args.lora_r_l,
            lora_alpha=args.lora_alpha,
            target_modules=args.t5_lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.llm_model = get_peft_model(model.llm_model, t5_lora_config)
        model.llm_model.print_trainable_parameters()

    if "L" not in args.tune_opt:
        for name, param in model.llm_model.named_parameters():
            param.requires_grad = False

    # TODO Insert LoRA to VIT_model. 
    print("Insert LoRA to VIT_model... ")
    vit_lora_config = LoraConfig(
        r=args.lora_r_v,
        lora_alpha=args.lora_alpha,
        target_modules=args.vit_lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="ViT",
    )
    
    model.visual_encoder = get_peft_model(model.visual_encoder, vit_lora_config)
    if "V" not in args.tune_opt:
        for name, param in model.visual_encoder.named_parameters():
            param.requires_grad = False
    model.visual_encoder.print_trainable_parameters()

    # TODO Insert LoRA to Qformer. 
    if hasattr(model, "Qformer") and "Q" in args.tune_opt:
        print("Insert LoRA to Qformer... ")
        qformer_lora_config = LoraConfig(
            r=args.lora_r_q,
            lora_alpha=args.lora_alpha,
            target_modules=args.qformer_lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="Qformer",
        )
        
        model.Qformer = get_peft_model(model.Qformer, qformer_lora_config)
        model.Qformer.print_trainable_parameters()

    config = {
        "t5_prune_spec": args.t5_prune_spec if args.t5_pruned_checkpoint is None else None,
        "vit_prune_spec": args.vit_prune_spec if args.vit_pruned_checkpoint is None else None,
        "t5_pruning_method": "none",
        "vit_pruning_method": "none",
        "importance_scores_cache": None,
        "keep_indices_cache": None,
        "is_strct_pruning": False,
        "is_global": args.is_global,
        "num_samples": args.num_data,
        "sparsity_ratio_granularity": args.sparsity_ratio_granularity,
        "max_sparsity_per_layer": args.max_sparsity_per_layer,
        "score_method": args.score_method,
        "num_data_first_stage": args.num_data_first_stage,
        "num_noise": args.num_noise,
        "sparsity_dict": args.sparsity_dict,
        "prune_per_model": args.prune_per_model,
        "iteration": args.iteration,
        "prune_n": args.prune_n,
        "prune_m": args.prune_m,
        "without_DSnoT": args.without_DSnoT,
        "initial_method": args.initial_method,
        "t5_model_prefix": args.t5_model_prefix,
        "peft_postfix": args.peft_postfix,
    }

    use_cache = getattr(model, args.t5_model_prefix).config.use_cache
    prune_task, prune_datasets, pruner = None, None, None
    prune_runner, train_runner, eval_runner = None, None, None
    eval_task, eval_datasets = None, None
    train_datasets = None
    sparsity_dict, data_loader = None, None
    _wrapped_model, _model = None, None

    t5_ratio = float(t5_ratio_org)
    vit_ratio = float(vit_ratio_org)

    if float(t5_ratio_org) != 1.0:
        config["t5_prune_spec"] = args.t5_prune_spec.replace(t5_ratio_org, str(t5_ratio))
    if float(vit_ratio_org) != 1.0:
        config["vit_prune_spec"] = args.vit_prune_spec.replace(vit_ratio_org, str(vit_ratio))

    print(config["t5_prune_spec"], config["vit_prune_spec"])

    # TODO cycles for pruning and training. 
    start = time.time()

    if args.prune:
        # TODO prune model
        prune_args = copy.deepcopy(args)
        prune_args.cfg_path = prune_args.prune_cfg_path
        prune_cfg = Config(prune_args)
        prune_cfg.job_id = prune_job_id
        prune_cfg.run_cfg.output_dir = prune_cfg.run_cfg.output_dir + "_prune"
        prune_cfg.run_cfg.distributed = False
        if prune_task is None:
            prune_task = tasks.setup_task(prune_cfg)
        if prune_datasets is None:
            prune_datasets = task.build_datasets(prune_cfg)

        prune_runner = RunnerBase(
            cfg=prune_cfg,
            job_id=prune_job_id,
            task=prune_task,
            model=model,
            datasets=prune_datasets,
        )
        prune_runner._wrapped_model = _wrapped_model
        
        if data_loader is None:
            data_loader = prune_runner.get_dataloader_for_importance_computation(
                num_data=args.num_data, power=args.power, batch_size=args.prunining_dataset_batch_size
            )
        if pruner is None:
            pruner = load_pruner(
                args.pruning_method,
                prune_runner.unwrap_dist_model(prune_runner.model).eval(),
                data_loader,
                cfg=config,
            )

        model, sparsity_dict = pruner.prune(lora_model=True)
        _wrapped_model = prune_runner.model if prune_runner.use_distributed else _wrapped_model

        print(f"**** Pruning Done: ", end="")
        print_gpu_memory_device(cfg.run_cfg.device)
        torch.cuda.empty_cache()
        print(f"**** Empty Cache: ", end="")
        print_gpu_memory_device(cfg.run_cfg.device)
        # TODO setting masks after pruning. 
        distilled_total_size = sum(
            (param != 0).float().sum() for name, param in model.named_parameters() if "lora" not in name
        )

        print(f"{orig_total_size / 10 ** 9 :.3f} B, {distilled_total_size / 10 ** 9 :.3f} B")
        print(f"After Pruning, Remaining Proportion: {distilled_total_size / orig_total_size * 100}%")
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2) / 1000
        print(f"peak_memory: {peak_memory}")

    if args.train:
        getattr(model, args.t5_model_prefix).config.use_cache = False
        setattr(task, "kl_weight", args.kl_weight)
        setattr(task, "T", args.T)

        if train_datasets is None:
            train_datasets = task.build_datasets(cfg, args.max_train_samples)

        for name, param in model.named_parameters():  # TODO exclude the grad from other modules.
            if param.requires_grad:
                if not any(remained in name for remained in remain_grads) or "LayerNorm" in name:
                    param.requires_grad = False

        setup_seeds(cfg)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"finetune {name}")

        sparse = args.sparse
        print(f"sparse: {sparse}")
        for name, module in model.named_modules():
            if isinstance(module, (Linear, LoraLayer, Linear8bitLt)):
                setattr(module, "sparse", sparse)
                    
        if train_runner is None:
            train_runner = RunnerBase(
                cfg=cfg,
                job_id=job_id,
                task=task,
                model=model,
                datasets=train_datasets,
                )
        train_runner._wrapped_model = _wrapped_model
        train_runner.start_epoch = 0

        # TODO train model with lora tuning. 
        train_runner.train(prune_retrain=True)

        # TODO merge weights. 
        for name, module in model.named_modules():
            if isinstance(module, (Linear, LoraLayer, Linear8bitLt)):
                module.merge()

        _wrapped_model = train_runner._wrapped_model
        
        torch.cuda.empty_cache()
        ## TODO set weights to zero.
        if args.sparse:
            for name, module in model.named_modules():
                if isinstance(module, (Linear, LoraLayer, Linear8bitLt)):
                    module.weight.data[~module.mask] = 0  
        
        distilled_total_size = sum(
            (param != 0).float().sum() for name, param in model.named_parameters() if "lora" not in name
        )

        print(f"{orig_total_size / 10 ** 9 :.3f} B, {distilled_total_size / 10 ** 9 :.3f} B")
        print(f"After Retraining, Remaining Proportion: {distilled_total_size / orig_total_size * 100}%")
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2) / 1000
        print(f"peak_memory: {peak_memory}")
        getattr(model, args.t5_model_prefix).config.use_cache = use_cache

    if args.evaluate:
        eval_args = copy.deepcopy(args)
        eval_args.cfg_path = eval_args.eval_cfg_path
        eval_cfg = Config(eval_args)
        eval_cfg.job_id = eval_job_id
        eval_cfg.run_cfg.output_dir = eval_cfg.run_cfg.output_dir + "_eval"
        if eval_task is None:
            eval_task = tasks.setup_task(eval_cfg)
        if eval_datasets is None:
            eval_datasets = eval_task.build_datasets(eval_cfg)

        if eval_runner is None:
            eval_runner = RunnerBase(
                cfg=eval_cfg,
                job_id=eval_job_id,
                task=eval_task,
                model=model,
                datasets=eval_datasets,
            )

        eval_runner._wrapped_model = _wrapped_model
        eval_runner.orig_total_size = orig_total_size
        eval_runner.distilled_total_size = distilled_total_size
        eval_runner.evaluate(skip_reload=True)
        torch.cuda.empty_cache()
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2) / 1000
        print(f"peak_memory: {peak_memory}")

    if args.save_pruned_model:
        saved_folder = os.path.join("pruned_checkpoint/V+L", args.pruning_method)
        os.makedirs(saved_folder, exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(saved_folder, job_id + ".pth")
        )

        # TODO save sparsity dict
        if sparsity_dict is not None and isinstance(sparsity_dict, dict):
            saved_folder = "sparsity_dict"
            os.makedirs(saved_folder, exist_ok=True)

            import yaml
            with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
                yaml.dump(sparsity_dict, f)

        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2) / 1000
        processing_time = time.time() - start
        training_dict = {
            "memory": peak_memory,
            "time": processing_time
        }

        saved_folder = "training_statistics"
        os.makedirs(saved_folder, exist_ok=True)

        import yaml
        with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
            yaml.dump(training_dict, f)

        saved_folder = "importance_scores"
        os.makedirs(saved_folder, exist_ok=True)
        torch.save(
            {k: v.importance_score for k, v in model.named_parameters() if getattr(v, "importance_score", None) is not None},
            os.path.join(saved_folder, job_id + ".pth")
        )


if __name__ == "__main__":
    main()
