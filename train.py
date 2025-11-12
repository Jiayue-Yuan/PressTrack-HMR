import sys
import os

import os.path as osp

import time
import yaml
import smplx
import torch
import pickle
import logging
import numpy as np

from dataset.CarpetPressureDataset import CarpetPressureDataset
from model.processor import InputProcessor
from model.multi_person import MultiPersonProcessor
from model.single_person import SinglePersonHMR

from core.MPloss import MPHMRLoss
from core.trainer import Trainer

from utils.optimizers.optim_factory_mae import LayerDecayValueAssigner, create_optimizer
from utils.others.utils import NoteGeneration, setup_seed
from utils.others.loss_record import updateLoss
from config.cmd_train_parser import parser_train_config


from torch.utils.data import DataLoader

def main(args):

    setup_seed(42)

    logging_path = os.path.join(args.logging_path, 'PressTrack-HMR', NoteGeneration(args))
    checkpoints_path = os.path.join(args.checkpoints_path, 'PressTrack-HMR', NoteGeneration(args))

    os.makedirs(logging_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=os.path.join(logging_path, 'logging.log'),
                        filemode='w')
    logger = logging.getLogger(__name__)

    logging.info(f"Start training for {args.epochs} epochs.")

    loss_record = updateLoss(logging_path)
    loss_record.start()

    cfgs = {
        'dataset_path': args.dataset_path,
        'seqlen': args.seqlen,
        'overlap': args.overlap,
        'dataset_mode': args.exp_mode,
        'normalize': True,
        'img_size': args.img_size,
        'curr_fold': args.curr_fold,
    }

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create Dataset from db file if available else create from folders
    train_set = CarpetPressureDataset(
        cfgs,
        mode='train'
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    g = torch.Generator()
    g.manual_seed(42)

    val_set = CarpetPressureDataset(
        cfgs,
        mode='eval'
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
        num_workers=0
    )

    # model
    if args.temp_encoder == 'gru':
        TEncoderConfigs = dict(
            input_size=args.tem_feature_len,
            hidden_size=args.gru_hidden_layers,
            bidirectional=False,
            num_layers=args.gru_layers
        )
    elif args.temp_encoder == 'trans':
        TEncoderConfigs = dict(
            input_feature_len=args.tem_feature_len,
            heads=8,
            mlp_hidden_dim=512,
            depth=args.trans_depth,
            drop_path_rate=0.2,
            drop_rate=0.1,
            seqlen=args.seqlen
        )
    elif args.temp_encoder == '1dconv':
        TEncoderConfigs = dict(
            input_feature_len=args.tem_feature_len,
            kernel_size=3,
        )
    elif args.temp_encoder == 'fc':
        TEncoderConfigs = None
    elif args.temp_encoder == 'rnn':
        TEncoderConfigs = dict(
            input_feature_len=args.tem_feature_len,
            output_feature_len=args.tem_feature_len,
            hidden_size=args.gru_hidden_layers,
            num_layers=args.gru_layers
        )

    hmr_model = SinglePersonHMR(
        seqlen=args.seqlen, 
        TEncoderConfigs=TEncoderConfigs,
        feature_len=args.feature_len,
        tem_feature_len=args.tem_feature_len,
        encoder_model=args.encoder, 
        tem_encoder_model=args.temp_encoder, 
    )

    model = MultiPersonProcessor(hmr_model)


    # regressore loss
    single_person_loss_cfg = {
        'e_3d_loss_weight': args.e_3d_loss_weight,
        'e_pose_loss_weight': args.e_pose_loss_weight,
        'e_shape_loss_weight': args.e_shape_loss_weight,
        'e_trans_loss_weight': args.e_trans_loss_weight,
        'device': 'cuda'
    }
    loss = MPHMRLoss(single_person_loss_cfg)

    # data

    # optimizer and schedule
    optimizer = create_optimizer(args, model)
    # print(optimizer)


    # ===================== Start Training ===================
    Trainer(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=loss,
        loss_record=loss_record,
        lr_scheduler=None,
        writer=logging,
        checkpoints_path=checkpoints_path,
        exp_mode=args.exp_mode,
        curr_fold=args.curr_fold,
        test_loader=None,
        len_val_set=val_set.get_data_len(),
        len_test_set=0,
        val_segments=val_set.get_segments(),
        test_segments=None,
        device=device
    ).fit()

    loss_record.end()

if __name__ == '__main__':

    args = parser_train_config()
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    main(args)