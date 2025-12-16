import os
import torch
import os.path as osp
import torch.nn as nn

from torchvision import models
from functools import partial
import torch.nn.functional as F
from .spin import Regressor
from .transBlock import Block
from .base_encoder import *
from .temp_encoder import *

class SinglePersonHMR(nn.Module):
    def __init__(
            self,
            seqlen,
            TEncoderConfigs,
            feature_len=768,
            tem_feature_len=768,
            encoder_model='resnet18',
            tem_encoder_model='gru',
    ):
        super().__init__()
        self.seqlen = seqlen
        self.encoder_model = encoder_model

        if self.encoder_model[:6] == 'resnet':
            self.encoder = RESNETEncoder(encoder_model, feature_len)

        self.layer = nn.Sequential(
            nn.Linear(feature_len + 10, tem_feature_len),
            nn.ReLU(),
        )

        if tem_encoder_model == 'gru':
            self.temp_encoder = TemporalGRUEncoder(
                TEncoderConfigs,
                add_linear=True,
                use_residual=True,
                feature_len=tem_feature_len,
                model_type=tem_encoder_model
            )
        elif tem_encoder_model == 'fc':
            self.temp_encoder = TemporalFCEncoder(
                seqlen=self.seqlen,
                feature_len=tem_feature_len
            )
        elif tem_encoder_model == 'trans':
            self.temp_encoder = TemporalTransEncoder(
                TEncoderConfigs
            )
        elif tem_encoder_model == '1dconv':
            self.temp_encoder = Temporal1DConvEncoder(
                TEncoderConfigs
            )
        elif tem_encoder_model == 'rnn':
            self.temp_encoder = TemporalRNNEncoder(
                TEncoderConfigs,
                add_linear=True,
                use_residual=True
            )

        self.regressor = Regressor(tem_feature_len)

        self.apply(self.init_weights)

        self.tem_feature_len = tem_feature_len

    def forward(self, x, bbox_info):

        batch_size, seqlen, h, w = x.shape

        x = self.encoder(x)  # [batch_size, seqlen, feature_len]

        x = torch.cat((x, bbox_info), dim=-1)

        x = self.layer(x)  # [batch_size, seqlen, tem_feature_len]

        x = self.temp_encoder(x).mean(1)  # [batch_size, tem_feature_len]

        x = x.reshape(-1, self.tem_feature_len) # [batch_size, tem_feature_len]

        smpl_output = self.regressor(x)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, -1) # torch.Size([16, 85])
            s['verts'] = s['verts'].reshape(batch_size, -1, 3) # torch.Size([16, 6890, 3])
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3) # torch.Size([16, 25, 3])
            s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3) # torch.Size([16, 24, 3, 3])

        return smpl_output

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
