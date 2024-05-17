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
import os 

def main():
    args = get_args_parser().parse_known_args()[0]
    data_name = os.path.basename(os.path.normpath(args.data_path))
    if not args.disable_wandb and args.local_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BU-Mamba",
            # track hyperparameters and run metadata
            config=args,
            name=f"{args.arch}-{data_name}"
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set seeds for reproducibility
    utils.set_seed(args.seed)

    # Prepare data loaders
    # train_loaders, val_loaders = get_data_loaders(args)

    all_folds_max_accuracy = []
    all_folds_max_auc = []
    all_folds_corresponding_test_stats = []

    for fold_index in range(args.k_folds):
        # train_loader = train_loaders[fold_index]
        # val_loader = val_loaders[fold_index]
        train_loader, val_loader, test_loader = get_data_loaders(args, seed=fold_index + args.seed)

        # Initialize and load model
        model = init_model(args, device)

        # start from here
        output_dir = Path(args.output_dir) / f"fold_{fold_index}"
        output_dir.mkdir(parents=True, exist_ok=True)

        optimizer, lr_scheduler, amp_autocast, loss_scaler, model_ema = init_training(model, args)
        criterion, mixup_fn = init_criterion(args)

        print(f"Start training for fold {fold_index + 1}/{args.k_folds} for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        max_auc = 0.0
        corresponding_test_stats = None

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of params: {n_parameters}')
        if not args.disable_wandb:
            wandb.run.summary["n_parameters"] = n_parameters
            
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler,
                                          amp_autocast, args.clip_grad, model_ema, mixup_fn,
                                          set_training_mode=args.train_mode, args=args)
            lr_scheduler.step(epoch)
            val_stats = evaluate(val_loader, model, device, amp_autocast, args, split='val')
            test_stats = evaluate(test_loader, model, device, amp_autocast, args, split='test')

            # Always print epoch results
            print(f"\nEpoch {epoch + 1} - Fold {fold_index + 1}: Training Loss: {train_stats['loss']:.4f}, ",
                  f"Training Acc: {train_stats['acc1']:.2f}%, Training AUC: {train_stats['auc']:.2f}%")

            print(f"Epoch {epoch + 1} - Fold {fold_index + 1}: Validation Loss: {val_stats['loss']:.4f}, ",
                  f"Validation Acc: {val_stats['acc1']:.2f}%, Validation AUC: {val_stats['auc']:.2f}%")

            print(f"Epoch {epoch + 1} - Fold {fold_index + 1}: Test Loss: {test_stats['loss']:.4f}, ",
                  f"Test Acc: {test_stats['acc1']:.2f}%, Test AUC: {test_stats['auc']:.2f}%\n")


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

            if max_accuracy <= val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                max_auc = val_stats["auc"]
                corresponding_test_stats = {
                    'test_acc': test_stats['acc1'],
                    'test_auc': test_stats['auc']
                }
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


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'fold': fold_index + 1}

            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            print(f'Max Val Statistics for fold {fold_index + 1}: Accuracy - {max_accuracy:.2f}%, AUC - {max_auc:.3f}')
            print(f'Corresponding Test Statistics: Accuracy - {corresponding_test_stats["test_acc"]:.2f}%, AUC - {corresponding_test_stats["test_auc"]:.3f}\n')


            # update W&B
            if not args.disable_wandb and args.local_rank == 0:
                wandb.log(log_stats)

        all_folds_max_accuracy.append(max_accuracy)
        all_folds_max_auc.append(max_auc)  # Append max AUC per fold
        all_folds_corresponding_test_stats.append(corresponding_test_stats)

        # print(f'Max Val Statistics for fold {fold_index + 1}: Accuracy - {max_accuracy:.2f}%, AUC - {max_auc:.3f}')
        # print(f'Corresponding Test Statistics: Accuracy - {corresponding_test_stats["test_acc"]:.2f}%, AUC - {corresponding_test_stats["test_auc"]:.3f}')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'Training time for fold {fold_index + 1}: {total_time_str} \n')

    average_max_accuracy = sum(all_folds_max_accuracy) / len(all_folds_max_accuracy)
    average_max_auc = sum(all_folds_max_auc) / len(all_folds_max_auc)
    print(f'Average maximum accuracy across all folds: {average_max_accuracy:.2f}%')
    print(f'Average maximum AUC across all folds: {average_max_auc:.3f}')

    average_test_acc = sum(stat['test_acc'] for stat in all_folds_corresponding_test_stats) / len(all_folds_corresponding_test_stats)
    average_test_auc = sum(stat['test_auc'] for stat in all_folds_corresponding_test_stats) / len(all_folds_corresponding_test_stats)
    print(f'Average maximum accuracy across all folds: {average_max_accuracy:.2f}%')
    print(f'Average maximum AUC across all folds: {average_max_auc:.3f}')
    print(f'Average corresponding Test Accuracy: {average_test_acc:.2f}%, AUC: {average_test_auc:.3f}\n')


    if not args.disable_wandb:
        wandb.run.summary["mean_val_acc"] = average_max_accuracy
        wandb.run.summary["mean_val_auc"] = average_max_auc
        wandb.run.summary["mean_test_acc"] = average_test_acc
        wandb.run.summary["mean_test_auc"] = average_test_auc
        wandb.finish()

if __name__ == '__main__':
    main()
