import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from .SPloss import SPHMRLoss
import numpy as np

class MPHMRLoss(nn.Module):
    def __init__(self, single_person_loss_cfg):
        super().__init__()
        # 初始化单人损失函数
        self.single_person_loss = SPHMRLoss(**single_person_loss_cfg)

    def forward(self, outputs, gt_keypoints_3d, gt_trans, gt_pose, gt_shape):
        """
        Args:
            outputs: List[List[Dict]], 每个元素是 [batch][person_id] = {'theta': [1, 85], 'kp_3d': [1, 25, 3], 'verts', 'rotmat']}
            gt_trans: [B, max_N, 3]
            gt_pose: [B, max_N, 72]
            gt_shape: [B, max_N, 10]
            gt_keypoints_3d: [B, max_N, 25, 3]
        """
        batch_size, max_N, _= gt_trans.shape
        total_loss = 0.0
        total_mpjpe = 0.0
        total_matched_pairs = 0


        loss_dict = {
            't_mpjpe': 0.0,
            'kp3d': 0.0,
            'trans': 0.0,
            'pose': 0.0,
            'shape': 0.0,
        }

        for b in range(batch_size):
            # 获取当前序列的预测和真实数据，gt取中间帧
            pred_people = outputs[b]  # List[Dict], 每个Dict是一个人的预测
            gt_trans_b = gt_trans[b]  # [max_N, 3]
            gt_pose_b = gt_pose[b]  # [max_N, 72]
            gt_shape_b = gt_shape[b]  # [max_N, 10]
            gt_kp3d_b = gt_keypoints_3d[b]  # [max_N, 25, 3]

            # 找出当前序列的有效真实人数
            valid_gt_mask = gt_shape_b.abs().sum(dim=-1) > 0  # [max_N]
            N_gt = int(valid_gt_mask.sum())
            if N_gt == 0:
                continue

            # 提取真实人的数据
            gt_trans_valid = gt_trans_b[valid_gt_mask]  # [N_gt, 3]
            gt_pose_valid = gt_pose_b[valid_gt_mask]  # [N_gt, 72]
            gt_shape_valid = gt_shape_b[valid_gt_mask]  # [N_gt, 10]
            gt_kp3d_valid = gt_kp3d_b[valid_gt_mask]  # [N_gt, 25, 3]

            # 收集有效预测
            valid_preds = [p for p in pred_people if 'theta' in p and 'kp_3d' in p] # [{person1},{person2}]
            N_pred = len(valid_preds)
            if N_pred == 0:
                continue

            if N_pred == N_gt == 1:
                row_ind = np.array([0])
                col_ind = np.array([0])
            else:
                # 构建代价矩阵（基于3D关键点的距离）
                cost_matrix = torch.zeros((N_pred, N_gt), device=gt_kp3d_b.device)
                for i, pred in enumerate(valid_preds):
                    for j in range(N_gt):
                        pred_kp = pred['kp_3d'][0]  # [25, 3]
                        gt_kp = gt_kp3d_valid[j]  # [25, 3]
                        mpjpe = torch.norm(pred_kp - gt_kp, p=2, dim=-1).mean()
                        cost_matrix[i, j] = mpjpe

                # 匈牙利匹配（最小化总距离）
                cost_matrix_np = cost_matrix.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            # 计算匹配对的损失
            for i, j in zip(row_ind, col_ind):
                # 获取匹配的预测和真实数据
                pred_data = valid_preds[i]
                gt_trans_j = gt_trans_valid[j]  # [3]
                gt_pose_j = gt_pose_valid[j]  # [72]
                gt_shape_j = gt_shape_valid[j]  # [10]
                gt_kp3d_j = gt_kp3d_valid[j]  # [25, 3]

                # 计算单人损失
                # import pdb; pdb.set_trace()
                loss, loss_dict_single = self.single_person_loss(
                    outputs=pred_data,
                    gt_trans=gt_trans_j.unsqueeze(0),  # 增加batch维度
                    gt_pose=gt_pose_j.unsqueeze(0),
                    gt_shape=gt_shape_j.unsqueeze(0),
                    gt_keypoints_3d=gt_kp3d_j.unsqueeze(0)
                )

                total_loss += loss
                total_mpjpe += loss_dict_single['t_mpjpe']
                total_matched_pairs += 1

                # 累加其他损失项
                for key in loss_dict:
                    if key in loss_dict_single:
                        loss_dict[key] += loss_dict_single[key]

        # 平均损失
        if total_matched_pairs > 0:
            total_loss /= total_matched_pairs
            total_mpjpe /= total_matched_pairs
            for key in loss_dict:
                loss_dict[key] /= total_matched_pairs

        return total_loss, loss_dict