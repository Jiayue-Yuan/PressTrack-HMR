import torch
import torch.nn as nn

import numpy as np

from utils.geometry.geometry import batch_rodrigues

import matplotlib.pyplot as plt
import smplx

class SPHMRLoss(nn.Module):
    def __init__(self,
                 e_3d_loss_weight=300.,
                 e_pose_loss_weight=20.,
                 e_shape_loss_weight=1,
                 e_trans_loss_weight=60.,
                 device='cuda'
                 ):
        super(SPHMRLoss, self).__init__()
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_root_pose_loss_weight = e_pose_loss_weight * 5  # 新增根关节权重
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
        # loss_smpl_pose = self.loss_smpl_pose(outputs['theta'][:, 3:75], gt_pose)
        loss_smpl_pose, loss_root_pose = self.loss_smpl_pose(outputs['theta'][:, 3:75], gt_pose)
        loss_smpl_shape = self.loss_smpl_shape(outputs['theta'][:, 75:], gt_shape)
        loss_smpl_trans = self.loss_smpl_trans(outputs['theta'][:, :3], gt_trans)

        loss = self.e_pose_loss_weight * loss_smpl_pose + \
               self.e_root_pose_loss_weight * loss_root_pose + \
               self.e_shape_loss_weight * loss_smpl_shape + \
               self.e_trans_loss_weight * loss_smpl_trans + \
               self.e_3d_loss_weight * loss_kp_3d

        mpjpe_loss = self.mpjpe(outputs['kp_3d'], gt_keypoints_3d)

        plot = False
        if plot:
            # 绘制两者的对比图，整个时间窗口
            smpl_model = smplx.create(
                model_path='/workspace/3d_pose_estimation/smpl/SMPL_NEUTRAL.pkl',
                model_type='smpl',
            )
            smpl_model.to(self.device)

            for t in range(gt_shape.shape[0]):
                poses = gt_pose[t].reshape(1, -1)  # (1, 72)
                shapes = gt_shape[t].reshape(1, -1)  # (1, 10)

                smpl_output = smpl_model(
                    betas=shapes,
                    body_pose=poses[:, 3:],
                    global_orient=poses[:, :3],  # 加入全局旋转
                    return_verts=True
                )

                # 加入全局平移
                gt_verts = smpl_output.vertices[0].detach().cpu().numpy() + gt_trans[t].detach().cpu().numpy()  # (6890, 3)
                gt_kp = smpl_output.joints[0, :25].detach().cpu().numpy() + gt_trans[t].detach().cpu().numpy()

                pred_kp = outputs['kp_3d'][t].detach().cpu().numpy()
                pred_verts = outputs['verts'][t].detach().cpu().numpy()

                fig = plt.figure(figsize=(15, 8))  # 设置 figure 的大小，调整宽度以容纳三个子图

                # 第一个子图: GT 和 Pred
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], c='b', marker='o', label='Predicted', s=0.01)
                ax1.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], c='r', marker='o', label='Ground Truth', s=0.01)
                ax1.set_title('GT & Pred')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.set_zlim([0, -1.8])
                ax1.set_xlim([-0.1, 1.2]) # 1.3
                ax1.set_ylim([-0.57, 1.83])  # 2.4
                ax1.set_box_aspect([1.3, 2.4, 1.8])
                ax1.legend()

                # 第二个子图: Pred Only
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], c='b', marker='o', label='Predicted', s=0.02)
                ax2.scatter(pred_kp[:, 0], pred_kp[:, 1], pred_kp[:, 2], c='black', marker='o', label='Predicted', s=1)
                ax2.set_title('Predicted Only')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                ax2.set_zlim([0, -1.8])
                ax2.set_xlim([-0.1, 1.2]) # 1.3
                ax2.set_ylim([-0.57, 1.83])  # 2.4
                ax2.set_box_aspect([1.3, 2.4, 1.8])
                ax2.legend()

                # 第三个子图: GT Only
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], c='r', marker='o', label='Ground Truth', s=0.02)
                ax3.scatter(gt_kp[:, 0], gt_kp[:, 1], gt_kp[:, 2], c='black', marker='o', label='Ground Truth', s=1)
                ax3.set_title('Ground Truth Only')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Z')
                ax3.set_zlim([0, -1.8])
                ax3.set_xlim([-0.1, 1.2]) # 1.3
                ax3.set_ylim([-0.57, 1.83])  # 2.4
                ax3.set_box_aspect([1.3, 2.4, 1.8])
                ax3.legend()

                plt.tight_layout()  # 调整子图布局
                plt.savefig(f'T_{t:02d}.png')
                plt.close(fig)

            import pdb; pdb.set_trace()

        loss_dict = {
            't_mpjpe': mpjpe_loss.item(),
            'kp3d': loss_kp_3d.item(),
            'trans': loss_smpl_trans.item(),
            'pose': loss_smpl_pose.item(),
            'root_pose': loss_root_pose.item(),
            'shape': loss_smpl_shape.item(),
        }

        return loss, loss_dict

    def keypoints_3d(self, pred_keypoints_3d, gt_keypoints_3d):
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()

    # def loss_smpl_pose(self, pred_pose, gt_pose):
    #     pred_rotmat_valid = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
    #     gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
    #     return self.criterion_shape(pred_rotmat_valid, gt_rotmat_valid)

    def loss_smpl_pose(self, pred_pose, gt_pose):
        # 将姿态参数转换为旋转矩阵
        pred_rotmat = batch_rodrigues(pred_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)

        # 分离根关节(第0个)和其他关节的旋转损失
        loss_root = self.criterion_shape(pred_rotmat[:, 0], gt_rotmat[:, 0])  # 根关节损失
        loss_other = self.criterion_shape(pred_rotmat[:, 1:], gt_rotmat[:, 1:])  # 其他关节损失

        return loss_other, loss_root  # 返回其他关节损失和根关节损失

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