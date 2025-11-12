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

    logging_path = os.path.join(args.logging_path, 'test', NoteGeneration(args))
    checkpoints_path = os.path.join(args.checkpoints_path, 'test', NoteGeneration(args))
    test_save_path = os.path.join(args.test_save_path, NoteGeneration(args))

    os.makedirs(logging_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

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


    g = torch.Generator()
    g.manual_seed(42)

    if args.exp_mode == 'unseen_group':
        val_loader = None
        val_segments = None
        val_data_len = 0
    else:
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
        val_data_len = val_set.get_data_len()
        val_segments = val_set.get_segments()

    if args.exp_mode == 'unseen_subject':
        test_loader = None
        test_segments = None
        test_data_len = 0
    else:
        test_set = CarpetPressureDataset(
            cfgs,
            mode='test'
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            generator=g,
            num_workers=0
        )
        test_data_len = test_set.get_data_len()
        test_segments = test_set.get_segments()

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
        seqlen=args.seqlen,  # 16
        TEncoderConfigs=TEncoderConfigs,
        feature_len=args.feature_len,  # 768
        tem_feature_len=args.tem_feature_len,  # 768 + 6
        encoder_model=args.encoder,  # resnet18
        tem_encoder_model=args.temp_encoder,  # gru
    )

    model = MultiPersonProcessor(hmr_model)

    ckpt = torch.load(args.test_best_checkpoints, map_location=device)['state_dict']
    model.load_state_dict(ckpt)

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


    # ===================== Start Training ===================
    Trainer(
        args=args,
        train_loader=None,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=loss,
        loss_record=loss_record,
        writer=logging,
        checkpoints_path=checkpoints_path,
        exp_mode=args.exp_mode,
        curr_fold=args.curr_fold,
        test_loader=test_loader,
        len_val_set=val_data_len,
        len_test_set=test_data_len,
        val_segments=val_segments,
        test_segments=test_segments,
        device=device,
        test_save_path=test_save_path
    ).test()

    loss_record.end()

if __name__ == '__main__':
    import glob
    # from model.track_processor import MultiPersonProcessor
    from model.multi_person import MultiPersonProcessor
    args = parser_train_config()

    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    args.note = 'test'

    args.curr_fold = 1
    args.lr = args.lr
    args.cosine = 1
    args.batch_size = 64
    args.seqlen = 16
    args.overlap = 0.95

    args.test_best_checkpoints = '/workspace/MCarpet/checkpoints/20250612/MPCarpet_unseen_group_resnet18_gru_spin_1e-05_16_16_0.9_768_1024/hps_60_losses_101.11.pth'
    configs = args.test_best_checkpoints.split('/')[5].split('_')
    args.exp_mode = 'unseen_' + configs[2]
    args.encoder = configs[3]
    args.temp_encoder = configs[4]
    args.feature_len = int(configs[-2])
    args.tem_feature_len = int(configs[-1])

    print(args.exp_mode, args.encoder, args.temp_encoder)
    main(args)