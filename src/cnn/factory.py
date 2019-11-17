import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensor
import pretrainedmodels

from .dataset.custom_dataset import CustomDataset
from .transforms.transforms import RandomResizedCrop
from .utils.logger import log

from .MODELS import res2fg
from .OPTIMIZERS.ranger import Ranger
from .d_li_GITHUB.octconv_pytorch_master.oct_resnet import oct_resnet101
from .d_li_GITHUB.octconv_pytorch_master.oct_resnet import oct_resnet50
from .d_li_GITHUB.octconv_pytorch_master.oct_resnet_mish import oct_resnet50 as oct_resnet50_mish
from .d_li_GITHUB.octconv_pytorch_master.oct_resnet_mish import oct_resnet101 as oct_resnet101_mish
# from .MODELS.Effnet import DenseNet121

from ww import f

def get_loss(cfg):
    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss


def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    print("-------------------------------")
    print(folds)
    print("-------------------------------")
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


def get_model(cfg):

    log(f('model: {cfg.model.name}'))
    log(f('pretrained: {cfg.model.pretrained}'))

    if cfg.model.name in ['resnext101_32x8d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        return model

    if cfg.model.name in ['res2next50']:
        model = res2fg.res2next(depth=50, num_classes=cfg.model.n_output, width_per_group=4, scale=4, pretrained=False, progress=True)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ res2next50 will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return model

    if cfg.model.name in ['OCTresnet101']:
        model = oct_resnet101()
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, cfg.model.n_output))

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ OCT_RESNET101 will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return model


    if cfg.model.name in ['OCTresnet101_mish']:
        model = oct_resnet101_mish()
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, cfg.model.n_output))

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ OCT_RESNET101_MISH will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return model



    if cfg.model.name in ['OCTresnet50_mish']:
        model = oct_resnet50_mish()
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, cfg.model.n_output))

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ OCT_RESNET50_MISH will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return model

    if cfg.model.name in ['Effnet_b5']:
        model = DenseNet121(cfg.model.n_output)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ EFFNET_B5 will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return model


    if (cfg.model.name in ['se_resnext101_32x4d']):
        try:
            model_func = pretrainedmodels.__dict__[cfg.model.name]
        except KeyError as e:
            model_func = eval(cfg.model.name)

        model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Linear(
            model.last_linear.in_features,
            cfg.model.n_output,)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ SE_RESNEXT101 will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return model


    try:
        model_func = pretrainedmodels.__dict__[cfg.model.name]
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ SE_RESNEXT50 will be used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    except KeyError as e:
        model_func = eval(cfg.model.name)

    model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        cfg.model.n_output,
    )
    return model


def get_optim(cfg, parameters):
    if (cfg.optim.name =='Adam'):
        optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
        log(f('optim: {cfg.optim.name}'))
        return optim

    elif(cfg.optim.name == 'Ranger'):
        optim = Ranger(parameters,**cfg.optim.params)
        log(f('optim: {cfg.optim.name}'))
        return optim
    else:
        print("SPECIFY CORRECT OPTIMIZER")


def get_scheduler(cfg, optim, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f('last_epoch: {last_epoch}'))
    return scheduler

