import torch
import torch.nn as nn
import transformers
from copy import deepcopy
from functools import partial

from lavis.common.registry import registry
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)


from lavis.compression.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity
from lavis.peft.src.peft.tuners.lora import Linear, LoraLayer, Linear8bitLt


def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def find_layers(module, layers=[nn.Linear, Linear, LoraLayer, Linear8bitLt], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, initial_method="wanda", layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0

        self.initial_method = initial_method
        if self.initial_method == "sparsegpt":
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.sum_metric_row = torch.zeros((self.columns), device=self.dev)
        
        self.mean = torch.zeros((self.columns), device=self.dev)
        self.var = torch.zeros((self.columns), device=self.dev)
        self.ntokens = 0
        
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.type(torch.float32)

        mean_inp = torch.mean(inp, dim=1, keepdim=True)
        var_inp = torch.var(inp, dim=1, unbiased=False, keepdim=True)
        num_inp = inp.shape[1]
        self.var = var_inp if self.ntokens == 0 else (self.var * self.ntokens + var_inp * num_inp) / (self.ntokens + num_inp)
        self.mean = mean_inp if self.ntokens == 0 else (self.mean * self.ntokens + mean_inp * num_inp) / (self.ntokens + num_inp)
        self.ntokens += num_inp
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.sum_metric_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.sum_metric_row += torch.sum(inp, dim=1) / self.nsamples
      
    def free(self):
        self.H = None
        torch.cuda.empty_cache()  

@registry.register_pruner("t5_dsnot_pruner")
class T5LayerDSnoTPruner(LayerWiseBasePruner):
    pruner_name = "t5_dsnot_pruner"
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="t5_model",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        skip_layer=None, 
        skip_sub_layer=None, 
        pow_of_var_regrowing=1.,
        max_cycle_time=1e2, 
        update_threshold=0.1, 
        without_same_sign=True, 
        without_DSnoT=False, 
        prune_n=0,
        prune_m=0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m,
        )
        
        self.pow_of_var_regrowing = pow_of_var_regrowing
        self.without_same_sign = without_same_sign
        self.without_DSnoT = without_DSnoT
        self.update_threshold = update_threshold       
        self.skip_layer = skip_layer
        self.skip_sub_layer = skip_sub_layer
        self.max_cycle_time = max_cycle_time
        self.loss_func = loss_language
        self.ignore_layers = [
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        ] # not used but may be used in the future
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and lm_head
            if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, transformer, prune_spec):
        side_config = deepcopy(transformer.config)

        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(prune_spec)
        
        side_config.num_decoder_layers = num_layers
        side_config.num_layers = num_layers

        if self.is_strct_pruning:
            # structural
            side_config.d_model = get_pruned_dim(side_config.d_model, res_keep_ratio)
            side_config.d_ff = get_pruned_dim(side_config.d_ff, ffn_keep_ratio)
            side_config.d_kv = get_pruned_dim(side_config.d_kv, attn_keep_ratio)
        else:
            # unstructural
            side_config.d_model = side_config.d_model
            side_config.d_ff = side_config.d_ff
            side_config.d_kv = side_config.d_kv
            
        pruned_transformer = transformer.__class__(side_config)

        return pruned_transformer
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith("t5_model"):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}. Use shared.weight for it.")
                    scores[k] = scores["t5_model.shared.weight"]

        return scores
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        use_cache = getattr(model, self.model_prefix).config.use_cache 
        getattr(model, self.model_prefix).config.use_cache = False 

        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

        getattr(model, self.model_prefix).config.use_cache = use_cache 
        return float(count)/total_params 
    
    # def forward_to_cache(self, model, batch):
    #     return model(batch)
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block", lora_model=False):
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False
        # layers = model.encoder.block
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
        inps = []
        # caches = {'i': 0}
        
        caches = []
        if "t5_model" in self.model_prefix:
            keys_to_cache = [
                "attention_mask", "position_bias", "encoder_attention_mask", "encoder_decoder_position_bias",
                "layer_head_mask", "cross_attn_layer_head_mask", "encoder_hidden_states",
            ]
        elif "opt_model" in self.model_prefix:
            keys_to_cache = [
                "attention_mask", "layer_head_mask", 
            ]
        elif any(prefix in self.model_prefix for prefix in ["llm_model",]):
            keys_to_cache = [
                "attention_mask", "position_ids", 
            ]
            
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                
            def forward(self, inp, dense=True, **kwargs):
                inps.append(inp)
                inps[-1].requires_grad = False
                
                cache = {}
                for k in keys_to_cache:
                    cache[k] = kwargs[k]
                if lora_model:
                    cache["dense"] = dense
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if total_samples >= n_samples:
                break
            if "image" in batch:
                total_samples += batch["image"].shape[0]
            else: 
                total_samples += len(batch["text_input"])
            try:
                self.forward_to_cache(model, batch, lora_model)
            except ValueError:
                pass 
        layers[0] = layers[0].module
        outs = [None] * len(inps)

        getattr(model, model_prefix).config.use_cache = use_cache
            
        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5, lora_model=False):
        use_cache = getattr(model, model_prefix).config.use_cache 
        getattr(model, model_prefix).config.use_cache = False 

        import time
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process, lora_model)

        n_samples = min(n_samples, len(inps))

        total_time = 0
        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if (f"model.layers.{i}" in model.hf_device_map):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = (
            #         inps.to(dev),
            #         outs.to(dev),
            #         attention_mask.to(dev),
            #         position_ids.to(dev),
            #     )
                
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
                        
            for h in handles:
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                
                start_time = time.time()

                # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                # setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())                
                # signed weight * activation
                
                DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1))

                if self.initial_method == "wanda":
                    # unsigned weight * activation^2
                    initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1))
                    )
                elif self.initial_method == "magnitude":
                    initial_metric = torch.abs(subset[name].weight.data)
                
                elif self.initial_method == "sparsegpt":
                    W = subset[name].weight.data.clone()
                    if isinstance(subset[name], nn.Conv2d):
                        W = W.flatten(1)
                    if isinstance(subset[name], transformers.Conv1D):
                        W = W.t()
                    W = W.float()

                    H = wrapped_layers[name].H
                    # del wrapped_layers[name].H
                    dead = torch.diag(H) == 0
                    H[dead, dead] = 1
                    W[:, dead] = 0

                    percdamp = 0.01
                    damp = percdamp * torch.mean(torch.diag(H))
                    diag = torch.arange(
                        wrapped_layers[name].columns, device=wrapped_layers[name].dev
                    )
                    H[diag, diag] += damp
                    H = torch.linalg.cholesky(H)
                    H = torch.cholesky_inverse(H)
                    H = torch.linalg.cholesky(H, upper=True)
                    Hinv = H

                    initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1)))
                # W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                weight_mask = torch.zeros_like(initial_metric) == 1

                if self.prune_n != 0:
                    # structured n:m sparsity
                    print(f"pruning {model_prefix} layer {i} {name} at structured {self.prune_n}:{self.prune_m} sparsity")
                    # print(f"name: {name}")
                    # if (name.split(".")[0] == self.skip_layer or name.split(".")[1] == self.skip_sub_layer):
                    #     for ii in range(initial_metric.shape[1]):
                    #         if ii % self.prune_m == 0:
                    #             tmp = initial_metric[:,ii:(ii+self.prune_m)].float()
                    #             weight_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                    # else:
                    initial_prune_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)
                    initial_res_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)

                    for ii in range(initial_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + self.prune_m)].float()
                            _, tmp_all_indices = torch.sort(tmp, dim=1)
                            tmp_all_indices += ii
                            res_prune_n = self.prune_m - self.prune_n
                            tmp_indices, tmp_res_indices = torch.split(
                                tmp_all_indices,
                                split_size_or_sections=[self.prune_n, res_prune_n],
                                dim=1,
                            )

                            initial_prune_indices = torch.cat(
                                (initial_prune_indices, tmp_indices), dim=1
                            )
                            initial_res_indices = torch.cat(
                                (initial_res_indices, tmp_res_indices), dim=1
                            )
                            weight_mask.scatter_(1, tmp_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    
                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)

                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if self.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            self.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one
                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    initial_metric.scatter_(1, initial_prune_indices, float("inf"))
                    W_metric_max_value = (torch.max(initial_metric, dim=1, keepdim=True)[0] + 1)

                    cycle_time = 1
                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    while not (
                        torch.all(update_mask == False)
                        or cycle_time > self.max_cycle_time
                    ):
                        cycle_time += 1

                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )
                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        recover_block_start_indice = (
                            regrowing_indice - regrowing_indice % self.prune_m
                        )

                        recover_block_indices = (
                            torch.arange(
                                0, self.prune_m, device=recover_block_start_indice.device
                            ).repeat(recover_block_start_indice.shape[1], 1)
                            + recover_block_start_indice
                        )

                        pruning_block = torch.gather(
                            initial_metric, 1, recover_block_indices.to(torch.int64)
                        )

                        _, pruning_indice = torch.topk(
                            pruning_block, 1, dim=1, largest=False
                        )

                        pruning_indice += recover_block_start_indice

                        pruning_metric = DSnoT_metric.gather( 1, pruning_indice.to(torch.int64) )
                        
                        reconstruction_error_after = ( reconstruction_error + pruning_metric - regrowing_metric )

                        update_mask = (update_mask & ( initialize_error_sign == torch.sign(reconstruction_error_after) ) & ( abs(reconstruction_error) > self.update_threshold))

                        initial_metric.scatter_(1, pruning_indice, W_metric_max_value)

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)
                        
                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )
                else:
                    # unstructured pruning
                    _, sorted_initial_indice = torch.sort(
                        initial_metric, dim=-1, stable=True
                    )
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
    
                    if sparsity_ratio[sparsity_key] == 0.: 
                        continue
                    sparsity_num = round(initial_metric.shape[1] * sparsity_ratio[sparsity_key])
                    res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                    initial_prune_indices, initial_res_indices = torch.split(
                        sorted_initial_indice,
                        split_size_or_sections=[sparsity_num, res_sparsity_num],
                        dim=1,
                    )
                    # print(f"name: {name}")

                    # if (
                    #     name.split(".")[0] == self.skip_layer
                    #     or name.split(".")[1] == self.skip_sub_layer
                    #     or self.without_DSnoT
                    # ):
                    if self.without_DSnoT:
                        weight_mask.scatter_(1, initial_prune_indices, True)    # unstructured

                    else:
                        weight_mask.scatter_(1, initial_prune_indices, True)    # unstructured
                                            
                        wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        )
                        wanda_metric.scatter_(1, initial_prune_indices, float("inf"))        # unstructured, exclude pruned columns. 
                        wanda_res_indices, _ = torch.split(
                            torch.sort(wanda_metric, dim=1, stable=True)[1],
                            split_size_or_sections=[res_sparsity_num, sparsity_num],
                            dim=1,
                        )
                        reorder_indice_of_pruning_indice = return_reorder_indice(
                            torch.gather(DSnoT_metric, 1, wanda_res_indices)
                        )
                        pruning_indices_block = torch.gather(
                            wanda_res_indices, 1, reorder_indice_of_pruning_indice
                        )                                                                   # TODO Equation (3)
                        
                        metric_for_regrowing = DSnoT_metric.clone()
                        metric_for_regrowing.scatter_(1, initial_res_indices, 0)            # unstructured, only for pruned columns. 
                        reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                        initialize_error_sign = torch.sign(reconstruction_error)

                        if self.pow_of_var_regrowing:
                            metric_for_regrowing /= torch.pow(
                                wrapped_layers[name].var.reshape((1, -1)),
                                self.pow_of_var_regrowing,
                            )

                        _, regrowing_indices_block = torch.sort(                            # [dimension, dimension]
                            metric_for_regrowing, dim=1, stable=True
                        )                                           
                        
                        # TODO initialize indices for regrowing. 
                        indice_indice_list_for_regrowing = torch.zeros(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        last_one = regrowing_indices_block.shape[-1] - 1
                        indice_indice_list_for_regrowing[:, 1] = last_one    # [0, n - 1] * n

                        update_num_for_regrowing = torch.ones(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )                                                    
                        update_num_for_regrowing[:, 1] = -1                  # [1, -1] * n
                        
                        # TODO initialize indices for pruning. 
                        indice_indice_list_for_pruning = torch.zeros(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        last_one = pruning_indices_block.shape[-1] - 1      # [0, n - 1] * n
                        indice_indice_list_for_pruning[:, 1] = last_one

                        update_num_for_pruning = torch.ones(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        update_num_for_pruning[:, 1] = -1                   # [1, -1] * n

                        update_mask = torch.ones_like(
                            reconstruction_error, dtype=torch.bool
                        )
                        cycle_time = 0
                        while not ( torch.all(update_mask == False) or cycle_time >= self.max_cycle_time ):
                            cycle_time += 1
                            
                            # regrowing, reconstruction_error > 0 --> [0, 1] \in R^m
                            indice_of_indice_indice_list_for_regrowing = (
                                (reconstruction_error > 0).int().to(torch.int64)
                            )
                            # indices of reconstruction_error > 0 --> [head, tail]
                            indice_indice_for_regrowing = torch.gather(
                                indice_indice_list_for_regrowing,
                                1,
                                indice_of_indice_indice_list_for_regrowing,
                            )

                            regrowing_indice = torch.gather(
                                regrowing_indices_block,
                                1,
                                indice_indice_for_regrowing.to(torch.int64),
                            )
                            regrowing_metric = DSnoT_metric.gather(
                                1, regrowing_indice.to(torch.int64)
                            )

                            indice_indice_list_for_regrowing.scatter_(
                                1,
                                indice_of_indice_indice_list_for_regrowing,
                                indice_indice_for_regrowing
                                + update_num_for_regrowing.gather(
                                    1, indice_of_indice_indice_list_for_regrowing
                                ),
                            )
 
                            # TODO pruning
                            indice_of_indice_indice_list_for_pruning = (
                                (reconstruction_error < 0).int().to(torch.int64)
                            )
                            # select < 0 
                            indice_indice_for_pruning = torch.gather(
                                indice_indice_list_for_pruning,
                                1,
                                indice_of_indice_indice_list_for_pruning,
                            )

                            pruning_indice = torch.gather(
                                pruning_indices_block,
                                1,
                                indice_indice_for_pruning.to(torch.int64),
                            )

                            pruning_metric = DSnoT_metric.gather(               
                                1, pruning_indice.to(torch.int64)
                            )

                            indice_indice_list_for_pruning.scatter_(
                                1,
                                indice_of_indice_indice_list_for_pruning, 
                                indice_indice_for_pruning
                                + update_num_for_pruning.gather(
                                    1, indice_of_indice_indice_list_for_pruning
                                ),
                            )

                            # change mask
                            reconstruction_error_after = (
                                reconstruction_error + pruning_metric - regrowing_metric
                            )

                            if self.without_same_sign:
                                update_mask = update_mask & (
                                    abs(reconstruction_error) > self.update_threshold
                                )
                            else:
                                update_mask = (
                                    update_mask
                                    & (abs(reconstruction_error) > self.update_threshold)
                                    & (
                                        initialize_error_sign
                                        == torch.sign(reconstruction_error_after)
                                    )
                                )
                                
                            weight_mask.scatter_(1, pruning_indice, update_mask)
                            weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                            sub_mask_prune = torch.gather(weight_mask, 1, pruning_indice)
                            sub_mask_regrow = torch.gather(weight_mask, 1, regrowing_indice)
                            prune_values = False * update_mask + sub_mask_prune * ~update_mask
                            regrow_values = True * update_mask + sub_mask_regrow * ~update_mask
                            
                            weight_mask.scatter_(1, pruning_indice, prune_values)
                            weight_mask.scatter_(1, regrowing_indice, regrow_values)
                        
                            reconstruction_error += torch.where(
                                update_mask,
                                pruning_metric,
                                torch.zeros_like(pruning_metric),
                            )
                            reconstruction_error -= torch.where(
                                update_mask,
                                regrowing_metric,
                                torch.zeros_like(regrowing_metric),
                            )
                            
                setattr(subset[name], "mask", ~weight_mask.bool())
                if lora_model == False:
                    subset[name].weight.data[weight_mask] = 0  ## set weights to zero 
                
            end_time = time.time()
            total_time += end_time - start_time

            # print(f"pruning time: {total_time:.3f}")
            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
            inps, outs = outs, inps

        getattr(model, model_prefix).config.use_cache = use_cache 
        torch.cuda.empty_cache()
        
        return model
    
    # def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
    #     # original_sparsity = 0.5 * (t5_sparsity + vit_sparsity)
    #     if self.sparsity_dict is not None:
    #         import yaml
    #         with open(self.sparsity_dict, "r") as f:
    #             return yaml.load(f, Loader=yaml.FullLoader)

    #     if sparsity_ratio_granularity == None:
    #         layer_to_group_mapping = {}
        
    #     else:
    #         def check(name, v):
    #             if len(v.shape) == 2 and \
    #                     ".block" in name and \
    #                         "relative_attention_bias.weight" not in name and \
    #                             name.startswith(self.model_prefix):
    #                 return True
    #             return False
    #         parameters_to_prune = [
    #             k for k, v in self.model.named_parameters() if check(k, v)
    #         ]

    #         if sparsity_ratio_granularity == "layer":
    #             layer_to_group_mapping = {
    #                 k: k
    #                 for k in parameters_to_prune
    #             }
    #         elif sparsity_ratio_granularity == "block":
    #             layer_to_group_mapping = {
    #                 k: ".".join(k.split(".")[:4])
    #                 for k in parameters_to_prune
    #             }
    #         else:
    #             raise NotImplementedError
        
    #     sparsity_module = LayerSparsity(
    #         self.model, 
    #         self.data_loader, 
    #         loss_language, 
    #         self.num_data_first_stage,
    #         original_sparsity,
    #         self.max_sparsity_per_layer,
    #         self.score_method,
    #         self.num_noise,
    #         self.noise_eps,
    #         layer_to_group_mapping,
    #         # prune_per_model=self.prune_per_model,
    #         # per_model_sparsity=[t5_sparsity, vit_sparsity],
    #     )
        
    #     return sparsity_module.return_sparsity()
        
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None, lora_model=False):
        
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.encoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            lora_model=lora_model, 
        )
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.decoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            lora_model=lora_model, 
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("vit_dsnot_pruner")
class VITLayerDSnoTPruner(LayerWiseBasePruner):
    pruner_name = "vit_dsnot_pruner"
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="visual",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        prune_n=0,
        prune_m=0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n,
            prune_m=prune_m,
        )
        
        self.loss_func = loss_vision
        
        self.ignore_layers = []
        
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and output layers
            if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, vit, vit_prune_spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(vit_prune_spec)
        
        if self.is_strct_pruning:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=int(vit.embed_dim * res_keep_ratio),
                attn_dim=int(vit.attn_dim * attn_keep_ratio),
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        else:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=vit.embed_dim,
                attn_dim=vit.attn_dim,
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        
        return pruned_vit
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith(self.model_prefix):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}")

        return scores
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            # print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        return float(count)/total_params 
    
    # def forward_to_cache(self, model, batch):
    #     return model.encode_image(batch["image"])
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block", lora_model=False):
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
        inps = []
        # caches = {'i': 0}
        
        # print(f"model.parameters.dtype: {dtype}")
        
        caches = []
        
        keys_to_cache = [
            "rel_pos_bias"
        ]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, rel_pos_bias, dense=True):
                inps.append(inp)
                inps[-1].requires_grad = False
                
                cache = {}
                cache["rel_pos_bias"] = rel_pos_bias
                if lora_model:
                    cache["dense"] = dense 
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if total_samples >= n_samples:
                break
            total_samples += batch["image"].shape[0]
            try:
                self.forward_to_cache(model, batch, lora_model)
            except ValueError:
                pass 
        layers[0] = layers[0].module

        outs = [None] * len(inps)

        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5, lora_model=False):
        import time
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process, lora_model)

        n_samples = min(n_samples, len(inps))

        total_time = 0
        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if (f"model.layers.{i}" in model.hf_device_map):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = (
            #         inps.to(dev),
            #         outs.to(dev),
            #         attention_mask.to(dev),
            #         position_ids.to(dev),
            #     )
                
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])
                        
            for h in handles:
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                
                start_time = time.time()

                # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                # setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())                
                # signed weight * activation
                
                DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1))

                if self.initial_method == "wanda":
                    # unsigned weight * activation^2
                    initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1))
                    )
                elif self.initial_method == "magnitude":
                    initial_metric = torch.abs(subset[name].weight.data)
                
                elif self.initial_method == "sparsegpt":
                    W = subset[name].weight.data.clone()
                    if isinstance(subset[name], nn.Conv2d):
                        W = W.flatten(1)
                    if isinstance(subset[name], transformers.Conv1D):
                        W = W.t()
                    W = W.float()

                    H = wrapped_layers[name].H
                    # del wrapped_layers[name].H
                    dead = torch.diag(H) == 0
                    H[dead, dead] = 1
                    W[:, dead] = 0

                    percdamp = 0.01
                    damp = percdamp * torch.mean(torch.diag(H))
                    diag = torch.arange(
                        wrapped_layers[name].columns, device=wrapped_layers[name].dev
                    )
                    H[diag, diag] += damp
                    H = torch.linalg.cholesky(H)
                    H = torch.cholesky_inverse(H)
                    H = torch.linalg.cholesky(H, upper=True)
                    Hinv = H

                    initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1)))
                # W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                weight_mask = torch.zeros_like(initial_metric) == 1

                if self.prune_n != 0:
                    # structured n:m sparsity
                    print(f"pruning {model_prefix} layer {i} {name} at structured {self.prune_n}:{self.prune_m} sparsity")
                    # if (name.split(".")[0] == self.skip_layer or name.split(".")[1] == self.skip_sub_layer):
                    #     for ii in range(initial_metric.shape[1]):
                    #         if ii % self.prune_m == 0:
                    #             tmp = initial_metric[:,ii:(ii+self.prune_m)].float()
                    #             weight_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                    # else:
                    initial_prune_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)
                    initial_res_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)

                    for ii in range(initial_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + self.prune_m)].float()
                            _, tmp_all_indices = torch.sort(tmp, dim=1)
                            tmp_all_indices += ii
                            res_prune_n = self.prune_m - self.prune_n
                            tmp_indices, tmp_res_indices = torch.split(
                                tmp_all_indices,
                                split_size_or_sections=[self.prune_n, res_prune_n],
                                dim=1,
                            )

                            initial_prune_indices = torch.cat(
                                (initial_prune_indices, tmp_indices), dim=1
                            )
                            initial_res_indices = torch.cat(
                                (initial_res_indices, tmp_res_indices), dim=1
                            )
                            weight_mask.scatter_(1, tmp_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    
                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)

                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if self.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            self.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one
                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    initial_metric.scatter_(1, initial_prune_indices, float("inf"))
                    W_metric_max_value = (torch.max(initial_metric, dim=1, keepdim=True)[0] + 1)

                    cycle_time = 1
                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    while not (
                        torch.all(update_mask == False)
                        or cycle_time > self.max_cycle_time
                    ):
                        cycle_time += 1

                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )
                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        recover_block_start_indice = (
                            regrowing_indice - regrowing_indice % self.prune_m
                        )

                        recover_block_indices = (
                            torch.arange(
                                0, self.prune_m, device=recover_block_start_indice.device
                            ).repeat(recover_block_start_indice.shape[1], 1)
                            + recover_block_start_indice
                        )

                        pruning_block = torch.gather(
                            initial_metric, 1, recover_block_indices.to(torch.int64)
                        )

                        _, pruning_indice = torch.topk(
                            pruning_block, 1, dim=1, largest=False
                        )

                        pruning_indice += recover_block_start_indice

                        pruning_metric = DSnoT_metric.gather( 1, pruning_indice.to(torch.int64) )
                        
                        reconstruction_error_after = ( reconstruction_error + pruning_metric - regrowing_metric )

                        update_mask = (update_mask & ( initialize_error_sign == torch.sign(reconstruction_error_after) ) & ( abs(reconstruction_error) > self.update_threshold))

                        initial_metric.scatter_(1, pruning_indice, W_metric_max_value)

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)
                        
                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )
                else:
                    # unstructured pruning
                    _, sorted_initial_indice = torch.sort(
                        initial_metric, dim=-1, stable=True
                    )
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
    
                    if sparsity_ratio[sparsity_key] == 0.: 
                        continue
                    sparsity_num = round(initial_metric.shape[1] * sparsity_ratio[sparsity_key])
                    res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                    initial_prune_indices, initial_res_indices = torch.split(
                        sorted_initial_indice,
                        split_size_or_sections=[sparsity_num, res_sparsity_num],
                        dim=1,
                    )
                    
                    # if (
                    #     name.split(".")[0] == self.skip_layer
                    #     or name.split(".")[1] == self.skip_sub_layer
                    #     or self.without_DSnoT
                    # ):
                    if self.without_DSnoT:
                        weight_mask.scatter_(1, initial_prune_indices, True)    # unstructured

                    else:
                        weight_mask.scatter_(1, initial_prune_indices, True)    # unstructured
                                            
                        wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                            wrapped_layers[name].scaler_row.reshape((1, -1))
                        )
                        wanda_metric.scatter_(1, initial_prune_indices, float("inf"))        # unstructured, exclude pruned columns. 
                        wanda_res_indices, _ = torch.split(
                            torch.sort(wanda_metric, dim=1, stable=True)[1],
                            split_size_or_sections=[res_sparsity_num, sparsity_num],
                            dim=1,
                        )
                        reorder_indice_of_pruning_indice = return_reorder_indice(
                            torch.gather(DSnoT_metric, 1, wanda_res_indices)
                        )
                        pruning_indices_block = torch.gather(
                            wanda_res_indices, 1, reorder_indice_of_pruning_indice
                        )                                                                   # TODO Equation (3)
                        
                        metric_for_regrowing = DSnoT_metric.clone()
                        metric_for_regrowing.scatter_(1, initial_res_indices, 0)            # unstructured, only for pruned columns. 
                        reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                        initialize_error_sign = torch.sign(reconstruction_error)

                        if self.pow_of_var_regrowing:
                            metric_for_regrowing /= torch.pow(
                                wrapped_layers[name].var.reshape((1, -1)),
                                self.pow_of_var_regrowing,
                            )

                        _, regrowing_indices_block = torch.sort(                            # [dimension, dimension]
                            metric_for_regrowing, dim=1, stable=True
                        )                                           
                        
                        # TODO initialize indices for regrowing. 
                        indice_indice_list_for_regrowing = torch.zeros(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        last_one = regrowing_indices_block.shape[-1] - 1
                        indice_indice_list_for_regrowing[:, 1] = last_one    # [0, n - 1] * n

                        update_num_for_regrowing = torch.ones(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )                                                    
                        update_num_for_regrowing[:, 1] = -1                  # [1, -1] * n
                        
                        # TODO initialize indices for pruning. 
                        indice_indice_list_for_pruning = torch.zeros(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        last_one = pruning_indices_block.shape[-1] - 1      # [0, n - 1] * n
                        indice_indice_list_for_pruning[:, 1] = last_one

                        update_num_for_pruning = torch.ones(
                            (reconstruction_error.shape[0], 2),
                            device=reconstruction_error.device,
                            dtype=torch.long,
                        )
                        update_num_for_pruning[:, 1] = -1                   # [1, -1] * n

                        update_mask = torch.ones_like(
                            reconstruction_error, dtype=torch.bool
                        )
                        cycle_time = 0
                        while not ( torch.all(update_mask == False) or cycle_time >= self.max_cycle_time ):
                            cycle_time += 1
                            
                            # regrowing, reconstruction_error > 0 --> [0, 1] \in R^m
                            indice_of_indice_indice_list_for_regrowing = (
                                (reconstruction_error > 0).int().to(torch.int64)
                            )
                            # indices of reconstruction_error > 0 --> [head, tail]
                            indice_indice_for_regrowing = torch.gather(
                                indice_indice_list_for_regrowing,
                                1,
                                indice_of_indice_indice_list_for_regrowing,
                            )

                            regrowing_indice = torch.gather(
                                regrowing_indices_block,
                                1,
                                indice_indice_for_regrowing.to(torch.int64),
                            )
                            regrowing_metric = DSnoT_metric.gather(
                                1, regrowing_indice.to(torch.int64)
                            )

                            indice_indice_list_for_regrowing.scatter_(
                                1,
                                indice_of_indice_indice_list_for_regrowing,
                                indice_indice_for_regrowing
                                + update_num_for_regrowing.gather(
                                    1, indice_of_indice_indice_list_for_regrowing
                                ),
                            )
 
                            # TODO pruning
                            indice_of_indice_indice_list_for_pruning = (
                                (reconstruction_error < 0).int().to(torch.int64)
                            )
                            # select < 0 
                            indice_indice_for_pruning = torch.gather(
                                indice_indice_list_for_pruning,
                                1,
                                indice_of_indice_indice_list_for_pruning,
                            )

                            pruning_indice = torch.gather(
                                pruning_indices_block,
                                1,
                                indice_indice_for_pruning.to(torch.int64),
                            )

                            pruning_metric = DSnoT_metric.gather(               
                                1, pruning_indice.to(torch.int64)
                            )

                            indice_indice_list_for_pruning.scatter_(
                                1,
                                indice_of_indice_indice_list_for_pruning, 
                                indice_indice_for_pruning
                                + update_num_for_pruning.gather(
                                    1, indice_of_indice_indice_list_for_pruning
                                ),
                            )

                            # change mask
                            reconstruction_error_after = (
                                reconstruction_error + pruning_metric - regrowing_metric
                            )

                            if self.without_same_sign:
                                update_mask = update_mask & (
                                    abs(reconstruction_error) > self.update_threshold
                                )
                            else:
                                update_mask = (
                                    update_mask
                                    & (abs(reconstruction_error) > self.update_threshold)
                                    & (
                                        initialize_error_sign
                                        == torch.sign(reconstruction_error_after)
                                    )
                                )
                                
                            weight_mask.scatter_(1, pruning_indice, update_mask)
                            weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                            sub_mask_prune = torch.gather(weight_mask, 1, pruning_indice)
                            sub_mask_regrow = torch.gather(weight_mask, 1, regrowing_indice)
                            prune_values = False * update_mask + sub_mask_prune * ~update_mask
                            regrow_values = True * update_mask + sub_mask_regrow * ~update_mask
                            
                            weight_mask.scatter_(1, pruning_indice, prune_values)
                            weight_mask.scatter_(1, regrowing_indice, regrow_values)
                        
                            reconstruction_error += torch.where(
                                update_mask,
                                pruning_metric,
                                torch.zeros_like(pruning_metric),
                            )
                            reconstruction_error -= torch.where(
                                update_mask,
                                regrowing_metric,
                                torch.zeros_like(regrowing_metric),
                            )
                setattr(subset[name], "mask", ~weight_mask.bool())
                if lora_model == False:
                    subset[name].weight.data[weight_mask] = 0  ## set weights to zero 
                
            end_time = time.time()
            total_time += end_time - start_time

            # print(f"pruning time: {total_time:.3f}")
            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])
            inps, outs = outs, inps

        torch.cuda.empty_cache()
        
        return model
    
    # def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
    #     # original_sparsity = 0.5 * (t5_sparsity + vit_sparsity)
    #     # print(f"self.sparsity_dict: {self.sparsity_dict}, sparsity_ratio_granularity: {sparsity_ratio_granularity}")
    #     if self.sparsity_dict is not None:
    #         import yaml
    #         with open(self.sparsity_dict, "r") as f:
    #             sparsity_dict = yaml.load(f, Loader=yaml.FullLoader)
                
    #         sparsity_dict = {k.replace("visual_encoder.", "visual."): v for k, v in sparsity_dict.items()}
            
    #         if "visual.blocks.39.attn.qkv.weight" not in sparsity_dict:
    #             # get from multi-modal pruning
    #             sparsity_dict["visual.blocks.39.attn.qkv.weight"] = 0
    #             sparsity_dict["visual.blocks.39.attn.proj.weight"] = 0
    #             sparsity_dict["visual.blocks.39.mlp.fc1.weight"] = 0
    #             sparsity_dict["visual.blocks.39.mlp.fc2.weight"] = 0
            
    #         return sparsity_dict

    #     if sparsity_ratio_granularity == None:
    #         layer_to_group_mapping = {}
        
    #     else:
    #         def check(name, v):
    #             if len(v.shape) == 2 and \
    #                     ".blocks" in name and \
    #                         name.startswith(self.model_prefix):
    #                 return True
    #             return False
    #         parameters_to_prune = [
    #             k for k, v in self.model.named_parameters() if check(k, v)
    #         ]

    #         if sparsity_ratio_granularity == "layer":
    #             layer_to_group_mapping = {
    #                 k: k
    #                 for k in parameters_to_prune
    #             }
    #         elif sparsity_ratio_granularity == "block":
    #             layer_to_group_mapping = {
    #                 k: ".".join(k.split(".")[:3])
    #                 for k in parameters_to_prune
    #             }
    #         else:
    #             raise NotImplementedError
        
    #     sparsity_module = LayerSparsity(
    #         self.model, 
    #         self.data_loader, 
    #         loss_vision, 
    #         self.num_data_first_stage,
    #         original_sparsity,
    #         self.max_sparsity_per_layer,
    #         self.score_method,
    #         self.num_noise,
    #         self.noise_eps,
    #         layer_to_group_mapping,
    #         # prune_per_model=self.prune_per_model,
    #         # per_model_group=["t5_model", "visual"],
    #         # prune_per_sparsity=[t5_sparsity, vit_sparsity],
            
    #     )
        
    #     return sparsity_module.return_sparsity()

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None, lora_model=False):
        # print("In: ", self.pruner_name)
        # print(f"sparsity_ratio_granularity: {self.sparsity_ratio_granularity}")

        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.blocks",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            lora_model=lora_model, 
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("blipt5_dsnot_pruner")
class BLIPT5LayerDSnoTPruner(LayerWiseBasePruner):
    pruner_name = "blipt5_dsnot_pruner"
    def __init__(
        self,
        model,
        data_loader,
        t5_prune_spec=None,
        vit_prune_spec=None,
        t5_pruning_method=None,
        vit_pruning_method=None,
        t5_importance_scores_cache=None,
        t5_keep_indices_or_masks_cache=None,
        vit_importance_scores_cache=None,
        vit_keep_indices_or_masks_cache=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        t5_model_prefix="t5_model",
        vit_model_prefix="visual_encoder",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        initial_method="wanda",
        skip_layer=None, 
        skip_sub_layer=None, 
        pow_of_var_regrowing=1.,
        max_cycle_time=1e2, 
        update_threshold=0.1, 
        without_same_sign=True, 
        without_DSnoT=False,
        peft_postfix="",
        prune_n=0,
        prune_m=0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=None,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=f"{vit_model_prefix}+{t5_model_prefix}",
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
            prune_n=prune_n, 
            prune_m=prune_m, 
        )
        self.pow_of_var_regrowing = pow_of_var_regrowing
        self.without_same_sign = without_same_sign
        self.without_DSnoT = without_DSnoT
        self.update_threshold = update_threshold       
        self.skip_layer = skip_layer
        self.skip_sub_layer = skip_sub_layer
        self.max_cycle_time = max_cycle_time
        
        self.t5_prune_spec = t5_prune_spec
        self.vit_prune_spec = vit_prune_spec
        
        self.peft_postfix = peft_postfix
        self.initial_method = initial_method
        
        assert t5_pruning_method is not None
        assert vit_pruning_method is not None
        
        self.t5_model_prefix = t5_model_prefix
        self.vit_model_prefix = vit_model_prefix
        
    def get_sparsity(self, t5_sparsity, vit_sparsity, sparsity_ratio_granularity=None):
        original_sparsity = 0.5 * (t5_sparsity + vit_sparsity)
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        if sparsity_ratio_granularity == None or sparsity_ratio_granularity == "none":
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                    ".block" in name and \
                        "relative_attention_bias.weight" not in name and \
                        (name.startswith(self.t5_model_prefix) or \
                            name.startswith(self.vit_model_prefix)):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "model":
                
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return self.t5_model_prefix
                    elif name.startswith(self.vit_model_prefix):
                        return self.vit_model_prefix
                    else:
                        return "other"
                
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
                
            elif sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return ".".join(name.split(".")[:4])
                    elif name.startswith(self.vit_model_prefix):
                        return ".".join(name.split(".")[:3])
                    else:
                        return "other"
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_vision_language, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
            per_model_group=[self.t5_model_prefix, self.vit_model_prefix],
            per_model_sparsity=[t5_sparsity, vit_sparsity],
        )
        
        return sparsity_module.return_sparsity()
        
    def forward_to_cache(self, model, batch, lora_model=False):
        if lora_model:
            return model(batch, vit_dense=True, llm_dense=True)
        else:
            return model(batch)
        
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None, lora_model=False):

        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
        
        global_sparsity_dict = None
        _, vit_keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
        _, t5_keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec) 

        if self.sparsity_ratio_granularity not in [None, "none"]: 

            # assert vit_keep_ratio == t5_keep_ratio

            global_sparsity_dict = self.get_sparsity(
                1 - t5_keep_ratio, 
                1 - vit_keep_ratio, # same as 1 - t5_keep_ratio
                sparsity_ratio_granularity=self.sparsity_ratio_granularity
            )

        if self.vit_prune_spec is not None and float(vit_keep_ratio) < 1.:

            sparsity_ratio = 1 - vit_keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio,
                    sparsity_ratio_granularity=None
                )
            
            # print(f"vit sparsity dict: {sparsity_dict}")
            _vit_prune = partial(VITLayerDSnoTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerDSnoTPruner.prepare_calibration_input_encoder,
                self,
                )
            
            self.model = _vit_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.vit_model_prefix,
                module_to_process=f"{self.vit_model_prefix}.blocks",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                lora_model=lora_model, 
            )
            
        if self.t5_prune_spec is not None and float(t5_keep_ratio) < 1.:

            sparsity_ratio = 1 - t5_keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
        #    elif self.prune_per_model: 
        #         sparsity_dict = self.get_sparsity(
        #             sparsity_ratio,
        #             sparsity_ratio_granularity=self.sparsity_ratio_granularity
        #         )
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio, 
                    sparsity_ratio_granularity=None
                )
            # print(f"t5 sparsity dict: {sparsity_dict}")
            _t5_prune = partial(T5LayerDSnoTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                T5LayerDSnoTPruner.prepare_calibration_input_encoder,
                self,
                )
            if "t5_model" in self.t5_model_prefix:
                self.model = _t5_prune(
                    self.model, self.data_loader, device, 
                    model_prefix=self.t5_model_prefix,
                    module_to_process=f"{self.t5_model_prefix}.encoder.block",
                    n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                    lora_model=lora_model, 
                )
                
                self.model = _t5_prune(
                    self.model, self.data_loader, device, 
                    model_prefix=self.t5_model_prefix,
                    module_to_process=f"{self.t5_model_prefix}.decoder.block",
                    n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                    lora_model=lora_model, 
                )
            else:
                self.model = _t5_prune(
                    self.model, self.data_loader, device, 
                    model_prefix=self.t5_model_prefix,
                    module_to_process=f"{self.t5_model_prefix}{self.peft_postfix}.model.layers", 
                    n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                    lora_model=lora_model, 
                )
                
        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, global_sparsity_dict
    
    def check(self, name, v, model_prefix):
        if len(v.shape) == 2 and \
                ".block" in name and \
                    "relative_attention_bias.weight" not in name and \
                        name.startswith(model_prefix):
            return True
        return False
    
    def trans_sparsity(self, vit_params, t5_params, vit_keep_ratio, t5_keep_ratio): 
        vit_keep_params = (vit_params + t5_params) * (vit_keep_ratio) / 2
        t5_keep_params = (vit_params + t5_params) * (t5_keep_ratio) / 2
        vit_keep_ratio = min(vit_keep_params / vit_params, 1.0)
        t5_keep_ratio = min(t5_keep_params / t5_params, 1.0)
        return vit_keep_ratio, t5_keep_ratio
    

def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the negative number remains unchanged, and the relative order in the positive number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice