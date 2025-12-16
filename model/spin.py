# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import math
import smplx
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt

from utils.geometry.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

from .smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS

GENDER = 'neutral'

class Regressor(nn.Module):
    def __init__(self, feature_len, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(feature_len + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = smplx.create('/workspace/3d_pose_estimation/smpl/SMPL_NEUTRAL.pkl', model_type='smpl')

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_trans = torch.tensor([0., 0., 0.])
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_trans)

    def forward(self, x, init_pose=None, init_shape=None, init_trans=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_trans is None:
            init_trans = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_trans = init_trans
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_trans], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_trans = self.deccam(xc) + pred_trans

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3) # 每个关节相对于全局坐标系的旋转矩阵
        # import pdb; pdb.set_trace()

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            transl=pred_trans,
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72) # 每个关节相对于全局坐标系的轴角表示


        # pred_kp = pred_joints[:, :25][0].detach().cpu().numpy()
        # pred_verts = pred_vertices[0].detach().cpu().numpy()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pred_kp[:, 0], pred_kp[:, 1], pred_kp[:, 2], c='b', marker='o', label='keypoints')
        # ax.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], c='b', marker='o', label='vertices', s=0.5)
        # ax.set_zlim([0, -4])
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_box_aspect([3, 3, 3])
        # ax.set_title('kp and verts')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.savefig('pred_kp_verts.png')
        # plt.close(fig)
        # import pdb; pdb.set_trace()


        output = [{
            'theta': torch.cat([pred_trans, pose, pred_shape], dim=1),
            'verts': pred_vertices,             # torch.Size([16, 6890, 3])
            'kp_3d': pred_joints[:, :25],   # torch.Size([16, 25, 3])
            'rotmat': pred_rotmat               # torch.Size([16, 24, 3, 3])
        }]
        return output

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]