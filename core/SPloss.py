import torch
import torch.nn as nn

import numpy as np

from utils.geometry.geometry import batch_rodrigues

import matplotlib.pyplot as plt
import smplx

class SPHMRLoss(nn.Module):
    def __init__(self,
                 e_3d_loss_weight=300.,
                 e_pose_loss_weight=1.,
                 e_shape_loss_weight=0.01,
                 e_trans_loss_weight=100.,
                 device='cuda'
                 ):
        super(SPHMRLoss, self).__init__()
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.e_trans_loss_weight = e_trans_loss_weight

        self.device = device

        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)

    def forward(self,
                outputs,
                gt_trans,
                gt_pose,
                gt_shape,
                gt_keypoints_3d
                ):

        reduce = lambda x: x.reshape((x.shape[0],) + x.shape[1:])

        gt_keypoints_3d = reduce(gt_keypoints_3d)
        gt_trans = reduce(gt_trans)
        gt_pose = reduce(gt_pose)
        gt_shape = reduce(gt_shape)

        for key in outputs:
            outputs[key] = reduce(outputs[key])

        loss_kp_3d = self.keypoints_3d(outputs['kp_3d'], gt_keypoints_3d)
        loss_smpl_pose = self.loss_smpl_pose(outputs['theta'][:, 3:75], gt_pose)
        loss_smpl_shape = self.loss_smpl_shape(outputs['theta'][:, 75:], gt_shape)
        loss_smpl_trans = self.loss_smpl_trans(outputs['theta'][:, :3], gt_trans)

        loss = self.e_pose_loss_weight * loss_smpl_pose + \
               self.e_shape_loss_weight * loss_smpl_shape + \
               self.e_trans_loss_weight * loss_smpl_trans + \
               self.e_3d_loss_weight * loss_kp_3d

        mpjpe_loss = self.mpjpe(outputs['kp_3d'], gt_keypoints_3d)

        loss_dict = {
            't_mpjpe': mpjpe_loss.item(),
            'kp3d': loss_kp_3d.item(),
            'trans': loss_smpl_trans.item(),
            'pose': loss_smpl_pose.item(),
            'shape': loss_smpl_shape.item(),
        }

        return loss, loss_dict

    def keypoints_3d(self, pred_keypoints_3d, gt_keypoints_3d):
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()

    def loss_smpl_pose(self, pred_pose, gt_pose):
        pred_rotmat_valid = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        return self.criterion_shape(pred_rotmat_valid, gt_rotmat_valid)

    def loss_smpl_shape(self, pred_shape, gt_shape):
        return self.criterion_shape(pred_shape, gt_shape)

    def loss_smpl_trans(self, pred_trans, gt_trans):
        return self.criterion_shape(pred_trans, gt_trans)

    def batch_smooth_pose_loss(self, pred_pose):
        pose_diff = pred_pose[:, 1:] - pred_pose[:, :-1]
        return torch.mean(pose_diff.abs())

    def batch_smooth_joint_loss(self, pred_joint):
        pose_diff = pred_joint[:, 1:] - pred_joint[:, :-1]
        return torch.mean(pose_diff.abs())

    def batch_smooth_shape_loss(self, pred_betas):
        shape_diff = pred_betas[:, 1:] - pred_betas[:, :-1]
        return torch.mean(shape_diff.abs())

    def batch_smooth_trans_loss(self, pred_trans):
        trans_diff = pred_trans[:, 1:] - pred_trans[:, :-1]
        return torch.mean(trans_diff.abs())

    def mpjpe(self, pred_keypoints_3d, gt_keypoints_3d):
        return ((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(-1).sqrt().mean()