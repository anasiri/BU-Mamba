from functools import partial

import torch
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vim.main import *
from sklearn.model_selection import KFold
from config import configurations
from vim.models_mamba import VisionMamba

from timm.models import create_model

def collate_fn(batch):
    # Apply transformations and separate images and labels
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images)
    # Convert labels into a tensor
    labels = torch.tensor(labels)

    return images, labels



if __name__ == '__main__':
    args = get_args_parser().parse_known_args()[0]
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.02, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load your dataset without any transformations applied
    dataset = ImageFolder('dataset/Combined/')

    # Number of folds
    k_folds = 5
    batch_size = 32
    num_workers = 1
    # KFold provides train/test indices to split data in train/test sets.
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Start the k-fold cross-validation run
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Create a Subset for each fold based on indices for training and validation, then apply respective transformations
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        # Define data loaders for training and validation using the subsets
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn([(train_transform(item[0]), item[1]) for item in x])
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn([(val_transform(item[0]), item[1]) for item in x])
        )
    if args.arch in configurations:
        config = configurations[args.arch]
        if args.arch.startswith('vim'):
            model = VisionMamba(num_classes=3, **config)
            state_dict = torch.load('vim_s_midclstok_80p5acc.pth')['model']
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            model.load_state_dict(state_dict, strict=False)
        elif args.arch.startswith('vit'):
            config = configurations['vit-s']
            model = create_model(
                args.model,
                pretrained=True,
                num_classes=3,
                drop_rate=config['drop_rate'],
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=224
            )
        embed_dim = model.embed_dim
        print('EMBEDDED DIM:', embed_dim)
    else:
        print(f"Unknown architecture: {args.arch}")

    model.to(device)
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active and False:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=3)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp:  # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
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
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(val_loader, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(val_subset)} test images: {test_stats['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))