from contextlib import suppress

import torch
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma

from config import configurations
import utils
from vim.models_mamba import VisionMamba

def init_model(args, device):
    """Initialize the model based on the provided arguments."""
    if args.arch in configurations:
        config = configurations[args.arch]
        if args.arch == 'vim-s':
            model = VisionMamba(num_classes=3, **config)
            state_dict = torch.load('checkpoints/vim_s_midclstok_80p5acc.pth')['model']
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            model.load_state_dict(state_dict, strict=False)
        elif args.arch == 'vit-s':
            # config = configurations['vit-s']
            model = create_model('vit_small_patch16_224', pretrained=True, img_size=224, num_classes=3)
            #     create_model(
            #     args.arch,
            #     pretrained=True,
            #     num_classes=3,
            #     drop_rate=config['drop_rate'],
            #     drop_path_rate=args.drop_path,
            #     drop_block_rate=None,
            #     img_size=224
            # ))
        model.to(device)
        return model
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")


def init_criterion(args):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active and False:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=3)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()


    return criterion, mixup_fn


def init_training(model, args):
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')


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

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
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

    return optimizer, lr_scheduler, amp_autocast, loss_scaler, model_ema