# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from utils import multi_class_auc

import utils


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    all_outputs = []
    all_targets = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with amp_autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        # Calculate accuracy for each batch
        acc1 = accuracy(outputs, targets)[0]
        batch_size = outputs.shape[0]

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters[f'train_acc1'].update(acc1.item(), n=batch_size)

        # Store outputs and targets for global AUC calculation
        all_outputs.append(outputs.detach())
        all_targets.append(targets.detach())

        # Concatenate all batches
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # Calculate global AUC
    train_auc = multi_class_auc(all_outputs, all_targets) * 100
    metric_logger.update(train_auc=train_auc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(args)
    # print("Averaged stats:", metric_logger)
    return {k.split('_')[-1]: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, args, split: str):
    assert split in ['val', 'test'], "Evaluation Split must be either 'val' or 'test'"
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"{split.capitalize()}:"

    # switch to evaluation mode
    model.eval()

    # These will store all outputs and targets to compute AUC
    all_outputs = []
    all_targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output, target)[0]
        batch_size = images.shape[0]
        if split == 'val':
            metric_logger.update(val_loss=loss.item())
        elif split == 'test':
            metric_logger.update(test_loss=loss.item())
        metric_logger.meters[f'{split}_acc1'].update(acc1.item(), n=batch_size)

        # Save outputs and targets to calculate AUC later
        all_outputs.append(output)
        all_targets.append(target)

    # Concatenate all batches for global AUC computation
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # Calculate global AUC
    auc_score = multi_class_auc(all_outputs, all_targets) * 100

    # gather the stats from all processes
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} AUC {auc:.3f}'
    #       .format(top1=metric_logger.meters[f'{split}_acc1'], losses=metric_logger.meters[f'{split}_loss'],
    #               auc=auc_score))

    # Add AUC to results
    metric_logger.meters[f'{split}_auc'].update(auc_score)
    metric_logger.synchronize_between_processes(args)
    results = {k.split('_')[-1]: meter.global_avg for k, meter in metric_logger.meters.items()}

    return results
