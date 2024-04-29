import datetime
import json
import time
from pathlib import Path

import torch
from timm.utils import get_state_dict
import wandb

from argparser import get_args_parser
from engine import train_one_epoch, evaluate
import utils
from data import get_data_loaders
from train import init_training, init_model, init_criterion


def main():
    args = get_args_parser().parse_known_args()[0]
    if not args.disable_wandb and args.local_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BU-Vim",
            # track hyperparameters and run metadata
            config=args
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set seeds for reproducibility
    utils.set_seed(args.seed)

    # Prepare data loaders
    train_loaders, val_loaders = get_data_loaders(args)

    all_folds_max_accuracy = []
    for fold_index in range(len(train_loaders)):
        train_loader = train_loaders[fold_index]
        val_loader = val_loaders[fold_index]
        # Initialize and load model
        model = init_model(args, device)

        # start from here
        output_dir = Path(args.output_dir) / f"fold_{fold_index}"
        output_dir.mkdir(parents=True, exist_ok=True)

        optimizer, lr_scheduler, amp_autocast, loss_scaler, model_ema = init_training(model, args)
        criterion, mixup_fn = init_criterion(args)

        print(f"Start training for fold {fold_index + 1}/{len(train_loaders)} for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of params: {n_parameters}')

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, train_loader,
                optimizer, device, epoch, loss_scaler, amp_autocast,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode,
                # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                args=args,
            )

            lr_scheduler.step(epoch)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'fold': fold_index,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)

            test_stats = evaluate(val_loader, model, device, amp_autocast, args)
            print(f"Accuracy of the network for fold {fold_index + 1} on test images: {test_stats['acc1']:.1f}%")

            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'fold': fold_index,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                            'args': args,
                        }, checkpoint_path)

            print(f'Max accuracy for fold {fold_index + 1}: {max_accuracy:.2f}% Epoch {epoch+1}/{args.epochs}')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'fold': fold_index + 1}

            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        print(f'Max accuracy for fold {fold_index + 1}: {max_accuracy:.2f}%')
        all_folds_max_accuracy.append(max_accuracy)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'Training time for fold {fold_index + 1}: {total_time_str} \n\n\n')

    average_max_accuracy = sum(all_folds_max_accuracy) / len(all_folds_max_accuracy)
    print(f'Average maximum accuracy across all folds: {average_max_accuracy:.2f}%')

if __name__ == '__main__':
    main()
