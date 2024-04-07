"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import numpy as np
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger, SmoothedValue
from torch.nn import KLDivLoss
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

@registry.register_task("image_text_retrain")
class ImageTextRetrainTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.kl_weight = 0.01
        self.T = 2. 
        
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.append(eval_output.cpu().numpy())

        print(f"eval results: {np.mean(results)}")
        return results
    
    def train_step(self, model, samples, vit_dense=False, llm_dense=False):
        outputs = model(samples, vit_dense=vit_dense, llm_dense=llm_dense)
        loss, logits = outputs["loss"], outputs["logits"]
        return loss, logits
    
    def valid_step(self, model, samples):
        loss = model(samples, vit_dense=False)["loss"]
        return loss
    
    def get_data_derivative(self, model, data_loader, num_data=128, power=2, num_logits=1, vision_weight=0.0, cuda_enabled=False):
        gradients_dict = {}

        if power == 1:
            grad_method = torch.abs
        elif power == 2:
            grad_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        for name, param in model.named_parameters():
            gradients_dict[name] = 0

        idx = 0

        no_grad_list = set()

        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            loss_dict = model.forward_with_vision_auxloss(samples)
            loss = loss_dict["loss"] + loss_dict["vision_auxloss"] * vision_weight
            loss.backward()

            for name, param in model.named_parameters():

                if param.grad is not None:
                    gradients_dict[name] += grad_method(param.grad.cpu().data) / num_data
                else:
                    no_grad_list.add(name)
            
            model.zero_grad()

            idx += 1

            if idx >= num_data:
                break

        for k in no_grad_list:
            print(f"{k} has no grad")

        return gradients_dict

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        kl_fnt = KLDivLoss(reduction="batchmean", log_target=True)
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, logits_DD = self.train_step(model=model, samples=samples, vit_dense=True, llm_dense=True)
                    # _, logits_DS = self.train_step(model=model, samples=samples, vit_dense=True, llm_dense=False)
                    # _, logits_SD = self.train_step(model=model, samples=samples, vit_dense=False, llm_dense=True)
                    
            model.train()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, logits_SS = self.train_step(model=model, samples=samples, vit_dense=False, llm_dense=False)

            kl_loss = kl_fnt(F.log_softmax(logits_SS / self.T, -1), F.log_softmax(logits_DD / self.T, -1)) 

            # kl_loss = kl_fnt(F.log_softmax(logits_SS / self.T, -1), F.log_softmax(logits_DS / self.T, -1)) + \
            #             + kl_fnt(F.log_softmax(logits_SS / self.T, -1), F.log_softmax(logits_SD / self.T, -1))
        
            # print(f"kl_loss: {kl_loss}")
            loss = (1 - self.kl_weight) * loss +  self.kl_weight * kl_loss
            # loss = kl_loss
            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
                        
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }