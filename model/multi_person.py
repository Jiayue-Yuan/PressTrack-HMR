import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import time

# 地毯四个角在世界坐标系下的坐标
corners = [(1.17, 1.83), (-0.03, 1.818), (1.135, -0.567), (-0.06, -0.565)]
carpet_coords = torch.tensor(corners, dtype=torch.float32).flatten()
carpet_coords = carpet_coords.unsqueeze(0).unsqueeze(0)

def stabilize_boxes(boxes_seq, eps=1e-6):
    """
    Args:
        boxes_seq: [B, T, max_N, 4], padded with [0,0,0,0] if missing boxes
    Returns:
        stabilized_boxes: [B, T, max_N, 4], where missing boxes are filled via temporal interpolation
    """
    B, T, max_N, _ = boxes_seq.shape
    device = boxes_seq.device
    stabilized_boxes = torch.zeros_like(boxes_seq)  # 保持原始形状

    for b in range(B):
        # 记录每个时间步实际有效的检测框数量
        counts = []
        for t in range(T):
            cnt = (boxes_seq[b, t].abs().sum(dim=1) > eps).sum().item()
            counts.append(cnt)

        # 找出该Batch中最常见的检测框数量n
        n = Counter(counts).most_common(1)[0][0] if counts else 0

        # 记录有效时间步（检测框数量==n的帧）
        valid_indices = [t for t, cnt in enumerate(counts) if cnt == n]

        for t in range(T):
            if counts[t] == n:
                # 直接复制有效帧
                stabilized_boxes[b, t] = boxes_seq[b, t]
            else:
                # 寻找最近的有效帧t1和t2（检测框数量==n的帧）
                t1, t2 = find_nearest_valid_frames(t, valid_indices)

                # 加权平均（按时间距离）
                if t1 == t2:
                    stabilized_boxes[b, t] = boxes_seq[b, t1]
                else:
                    weight1 = (t2 - t) / (t2 - t1)
                    weight2 = (t - t1) / (t2 - t1)
                    stabilized_boxes[b, t] = weight1 * boxes_seq[b, t1] + weight2 * boxes_seq[b, t2]

    return stabilized_boxes

def find_nearest_valid_frames(t, valid_indices):
    """找到距离t最近的前后有效帧索引"""
    valid_arr = np.array(valid_indices)
    prev_valid = valid_arr[valid_arr <= t]
    t1 = prev_valid[-1] if len(prev_valid) > 0 else valid_arr[0]

    next_valid = valid_arr[valid_arr >= t]
    t2 = next_valid[0] if len(next_valid) > 0 else valid_arr[-1]

    return t1, t2

def extract_person_pressures(pressure_seq: torch.Tensor, boxes_seq: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    从压力序列中提取每个人的压力图像，并将其放入 128x128 的中心
    参数:
        pressure_seq: 形状为[B, T, H, W]的压力序列
        boxes_seq: 形状为[B, T, max_N, 4]的边界框序列(格式为x1,y1,x2,y2)
    返回:
        形状为[max_N, B, T, 128, 128]的单人压力序列
        形状为[max_N, B, T, 3]的 128x128 的中心点在原压力图像中的坐标
    """
    B, T, H, W = pressure_seq.shape
    max_N = boxes_seq.shape[2]
    img_H, img_W = 128, 128

    # 初始化输出张量
    person_pressures = torch.zeros((max_N, B, T, img_H, img_W),
                                   device=pressure_seq.device,
                                   dtype=pressure_seq.dtype)
    centers = torch.zeros((max_N, B, T, 2),
                          device=pressure_seq.device,
                          dtype=torch.float32)

    # Convert boxes to long and get valid masks
    boxes_long = boxes_seq.long()
    valid_mask = (boxes_seq.sum(dim=-1) != 0)  # [B, T, max_N]

    # Compute centers with correct dimension ordering
    boxes_permuted = boxes_long.permute(2, 0, 1, 3)  # [max_N, B, T, 4]
    centers[..., 0] = torch.div(boxes_permuted[..., 0] + boxes_permuted[..., 2], 2, rounding_mode='trunc')  # center_x
    centers[..., 1] = torch.div(boxes_permuted[..., 1] + boxes_permuted[..., 3], 2, rounding_mode='trunc')  # center_y

    # Process all frames at once
    for n in range(max_N):
        # Get boxes for this person across all batches and frames
        person_boxes = boxes_long[:, :, n]  # [B, T, 4]

        # Get valid frames for this person
        person_valid = valid_mask[:, :, n]  # [B, T]

        # Get coordinates
        x1 = person_boxes[..., 0]
        y1 = person_boxes[..., 1]
        x2 = person_boxes[..., 2]
        y2 = person_boxes[..., 3]

        # Compute sizes
        person_h = y2 - y1
        person_w = x2 - x1

        # Compute start positions in 128x128 image
        start_y = torch.div(img_H - person_h, 2, rounding_mode='trunc')
        start_x = torch.div(img_W - person_w, 2, rounding_mode='trunc')

        # Process each batch and frame (vectorized where possible)
        for b in range(B):
            for t in range(T):
                if person_valid[b, t]:
                    # Extract pressure region
                    person_frame = pressure_seq[b, t, y1[b, t]:y2[b, t], x1[b, t]:x2[b, t]]

                    # Place in output tensor
                    sy = start_y[b, t]
                    sx = start_x[b, t]
                    ph = person_h[b, t]
                    pw = person_w[b, t]

                    person_pressures[n, b, t, sy:sy + ph, sx:sx + pw] = person_frame

    return person_pressures, centers



class MultiPersonProcessor(nn.Module):
    def __init__(self, hmr_model):
        super().__init__()
        self.hmr = hmr_model

    def forward(self, pressure_seq: torch.Tensor, boxes_seq: torch.Tensor):
        """
        Args:
            pressure_seq: [B, T, H, W] 原始压力序列
            boxes_seq: [B, T, max_N, 4] 足迹框位置
        Returns:
            List[List[Dict]: 每个样本每个人的SMPL参数
        """
        boxes_seq = stabilize_boxes(boxes_seq) # 0.02s
        person_pressures, centers = extract_person_pressures(pressure_seq, boxes_seq) # 0.05 s

        B, T, H, W = pressure_seq.shape
        max_N = boxes_seq.shape[2]
        # 初始化
        MP_smpl_inner = [{} for _ in range(max_N)]
        MP_smpl = [MP_smpl_inner.copy() for _ in range(B)]
        valid_person_pressures = []
        valid_bbox = []
        original_indices = []

        for b in range(B):
            for n in range(max_N):
                if not torch.all(boxes_seq[b, :, n] == 0).item():
                    person_pressure = person_pressures[n, b]
                    bbox_center = centers[n, b]
                    valid_person_pressures.append(person_pressure)
                    valid_bbox.append(bbox_center)
                    original_indices.append((b, n))

        if valid_person_pressures:
            # 将收集到的压力序列和中心偏移堆叠起来，增加一个批次维度
            batched_person_pressure = torch.stack(valid_person_pressures, dim=0) # [num_valid_persons, T, H, W]
            bbox_info = torch.stack(valid_bbox, dim=0) # [num_valid_persons, T, 2]
            carpet_coords_expanded = carpet_coords.expand(bbox_info.shape[0], bbox_info.shape[1], -1).to(bbox_info.device)
            combined_info = torch.cat((bbox_info, carpet_coords_expanded), dim=-1) # [num_valid_persons, T, 10]
            smpl_output = self.hmr(batched_person_pressure, combined_info)  # List[Dict]

            for i, (b_idx, n_idx) in enumerate(original_indices):
                MP_smpl[b_idx][n_idx] = {
                    'theta': smpl_output[0]['theta'][i:i+1],
                    'verts': smpl_output[0]['verts'][i:i+1],
                    'kp_3d': smpl_output[0]['kp_3d'][i:i+1],
                    'rotmat': smpl_output[0]['rotmat'][i:i+1]
                }
                if torch.all(boxes_seq[b_idx, :, n_idx] == 0).item():
                    MP_smpl[b_idx][n_idx] = {}
        return MP_smpl
