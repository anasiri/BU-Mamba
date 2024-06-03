from contextlib import suppress

import torch
from torch import nn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma

from config import configurations
import utils
from vim.models_mamba import VisionMamba
from VMamba.classification.models.vmamba import VSSM
from torchvision import models

def init_model(args, device, num_classes):
    """Initialize the model based on the provided arguments."""
    if args.arch == 'resnet50':
        model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, num_classes)
    elif args.arch == 'vgg16':
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )
    elif args.arch == 'vit-ti16':
        model = create_model('vit_tiny_patch16_224', pretrained=True, img_size=224, num_classes=num_classes)
    elif args.arch == 'vit-s16':
        model = create_model('vit_small_patch16_224', pretrained=True, img_size=224, num_classes=num_classes)
    elif args.arch == 'vit-s32':
        model = create_model('vit_small_patch32_224', pretrained=True, img_size=224, num_classes=num_classes)
    elif args.arch == 'vit-b16':
        model = create_model('vit_base_patch16_224', pretrained=True, img_size=224, num_classes=num_classes)
    elif args.arch == 'vit-b32':
        model = create_model('vit_base_patch32_224', pretrained=True, img_size=224, num_classes=num_classes)
    elif args.arch == 'vim-s':
        config = configurations[args.arch]
        model = VisionMamba(num_classes=num_classes, **config)
        # state_dict = torch.load('checkpoints/pretrained/vim_s_midclstok_80p5acc.pth')['model']
        # state_dict.pop('head.weight')
        # state_dict.pop('head.bias')
        # model.load_state_dict(state_dict, strict=False)
    elif args.arch == 'vssm-ti':
        config = configurations[args.arch]
        config['num_classes'] = num_classes
        model = VSSM(**config)
        state_dict = torch.load('checkpoints/pretrained/vssm_tiny_0230_ckpt_epoch_262.pth')['model']
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith('classifier')}
        model.load_state_dict(state_dict, strict=False)
    elif args.arch == 'vssm-s':
        config = configurations[args.arch]
        config['num_classes'] = num_classes
        model = VSSM(**config)
        state_dict = torch.load('checkpoints/pretrained/vssm_small_0229_ckpt_epoch_222.pth')['model']
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith('classifier')}
        model.load_state_dict(state_dict, strict=False)
    elif args.arch == 'vssm-b':
        config = configurations[args.arch]
        config['num_classes'] = num_classes
        model = VSSM(**config)
        state_dict = torch.load('checkpoints/pretrained/vssm_base_0229_ckpt_epoch_237.pth')['model']
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith('classifier')}
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    model.to(device)
    return model

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
