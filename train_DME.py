import os
import argparse
import logging
import math
import time
import collections
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from parse_config import ConfigParser
from config.EncoderConfig import CTX_ADDING_FUSION_CFG, CTX_CONCAT_FUSION_CFG, QUERY_ADDING_FUSION_CFG, QUERY_CONCAT_FUSION_CFG, MM_FUSION_CFG
from models.model import AQVSR_Bert
from models.loss import NormSoftmaxLoss
from datasets.Datasets import AQVSR_query, AQVSR_segment, collate_for_concat_fusion, collate_for_adding_fusion, collate_concat_segment
from AQVSR.Trainer import Trainer_DME


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/home/stan/ai_assistant/code_release/AQVSR/config/modelv1.json', type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='4', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--ctx', '--ctx_layers'], type=int, target=('arch','ctx_layers')),
        CustomArgs(['--q','--q_layers'],type=int, target=('arch','q_layers')),
    ]
    config = ConfigParser(args, options)
    logger = config.get_logger('train')
    
    # dataset
    trainset = AQVSR_query(dset_name='train')
    valset = AQVSR_query(dset_name='valid')
    val_seg = AQVSR_segment(dset_name='valid')
    testset = AQVSR_query(dset_name='test')
    test_seg = AQVSR_segment(dset_name='test')


    f_concat_fusion = partial(
        collate_for_concat_fusion,
        mask_query_text=config['modal']['mask_query_text'],
        mask_query_img=config['modal']['mask_query_vis'],
        mask_ctx_text=config['modal']['mask_ctx_text'],
        mask_ctx_vis=config['modal']['mask_ctx_vis']
    )
    f_concat_segment = partial(
        collate_concat_segment,
        mask_ctx_text=config['modal']['mask_ctx_text'], 
        mask_ctx_vis=config['modal']['mask_ctx_vis']
    )
    # dataloader
    train_loader = DataLoader(dataset=trainset, batch_size=16, shuffle=True, collate_fn=f_concat_fusion)

    val_loader = DataLoader(dataset=valset, batch_size=16, shuffle=False, collate_fn=f_concat_fusion)
    val_seg_loader = DataLoader(dataset=val_seg, batch_size=16, shuffle=False, collate_fn=f_concat_segment)

    test_loader = DataLoader(dataset=testset, batch_size=16, shuffle=False, collate_fn=f_concat_fusion)
    test_seg_loader = DataLoader(dataset=test_seg, batch_size=16, shuffle=False, collate_fn=f_concat_segment)

    # model
    QUERY_CONCAT_FUSION_CFG.num_hidden_layers = config['arch']["q_layers"]
    CTX_CONCAT_FUSION_CFG.num_hidden_layers = config['arch']["ctx_layers"]

    
    model = AQVSR_Bert(
        QueryEncCfg=QUERY_CONCAT_FUSION_CFG,
        CtxEncCfg=CTX_CONCAT_FUSION_CFG,
        loss_type=None
    )
    logger.info(model)

    # get loss fn
    loss = NormSoftmaxLoss(temperature=0.05)

    # get metric
    metrics = []
    
    # optim
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None

    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    print(lr_scheduler)
    # get writer
    writer = SummaryWriter(config.log_dir)

    # trainer
    trainer = Trainer_DME(
        model=model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer, 
        config=config, 
        data_loader=train_loader, 
        valid_data_loader=val_loader, 
        val_seg_loader=val_seg_loader, 
        test_data_loader=test_loader, 
        test_seg_loader=test_seg_loader,
        lr_scheduler=lr_scheduler, 
        writer=writer
    )

    trainer.train()
