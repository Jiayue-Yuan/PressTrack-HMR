from turtledemo.penrose import start
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from model.multi_person import *
import sys
from pathlib import Path
base_dir = Path(__file__).parent.parent
yolo_path = str(base_dir / "yolo" / "ultralytics-main")
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

from ultralytics import YOLO

yolo_path = './yolo/ultralytics-main/runs/detect/train3/weights/best.pt'
tracker_config = "./yolo/ultralytics-main/ultralytics/cfg/trackers/footprint.yaml"
cmap = plt.get_cmap('viridis')
interpolation = 'nearest'
yolo_model = YOLO(yolo_path)

def get_person_color(person_id):
  # 预定义一些鲜明的颜色
  colors = [
    '#00FF00',  # 绿色
    '#FF0000',  # 红色
    '#00BFFF',  # 蓝色
  ]
  return colors[person_id % len(colors)]

def pressure_to_bgr(pressure_frame: np.ndarray) -> np.ndarray:
    """将压力数据转换为BGR图像"""
    colored = cmap(pressure_frame)[..., :3]  # 忽略alpha通道
    rgb_array = (colored * 255).astype(np.uint8)
    bgr_array = rgb_array[..., ::-1]
    return bgr_array

class MultiPersonProcessor(nn.Module):
    def __init__(self, hmr_model):
        super().__init__()
        self.hmr = hmr_model

    def tracking_by_detection(self, pressure_sequence: torch.Tensor, max_N: int, plot: bool = False):
        """
        处理时序压力数据，返回跟踪结果以及区分后的单人压力
        Args:
            pressure_sequence: [B, T, H, W] 批量的时序压力数据
            max_N: int
            plot: bool 是否开启画图，结果保存在 "tracking_visualization"
        Returns:
            boxes_seq: [B, T, max_N, 4] 跟踪得到的的足迹框位置
        """
        batch_size, seqlen = pressure_sequence.shape[:2]
        boxes_seq = torch.zeros(batch_size, seqlen, max_N, 4)  # 初始化输出张量

        for b in range(batch_size):
            if plot:
                track_ids = []
                fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(31, 16)) # figsize 设置整个图的大小

            for t in range(seqlen):
                frame = pressure_sequence[b, t].cpu().numpy()
                bgr_frame = pressure_to_bgr(frame)
                # 执行跟踪
                tracked_objects = yolo_model.track(
                    bgr_frame,
                    persist=True,
                    tracker=tracker_config,
                    verbose=False,
                    conf=0.7
                )
                # 解析跟踪结果
                if tracked_objects[0].boxes.id is not None:
                    bboxes = tracked_objects[0].boxes.xyxy.cpu().numpy()
                    ids = tracked_objects[0].boxes.id.int().cpu().numpy()
                    sorted_indices = sorted(range(len(ids)), key=lambda i: ids[i])
                    for i, box_idx in enumerate(sorted_indices):
                        if i+1 > max_N:  # 超出最大目标数则忽略
                            break
                        boxes_seq[b, t, i] = torch.from_numpy(bboxes[box_idx])

                if plot:
                    row, col = t // 8, t % 8
                    axes[row, col].imshow(frame, interpolation='nearest', cmap='viridis')
                    axes[row, col].axis('off')

                    # 绘制每个目标的框和ID
                    if tracked_objects[0].boxes.id is not None:
                        print(bboxes)
                        print(ids)
                        track_ids.append(ids.tolist())

                        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
                            rect = patches.Rectangle(
                                (x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=get_person_color(ids[idx]), linewidth=3
                            )
                            axes[row, col].add_patch(rect)
                            axes[row, col].text(
                                x1, y1 - 1, f"ID: {ids[idx]}",
                                color=get_person_color(ids[idx]), fontsize=18,
                                # bbox=dict(facecolor='white', alpha=0.8, pad=1)
                            )
            if plot:
                plt.tight_layout()
                save_path = f"tracking_visualization/batch_{b}_8x2_tracking.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"batch {b} 的4x4跟踪可视化已保存至：{save_path}")
                print(track_ids)
                # import pdb; pdb.set_trace()

        return boxes_seq


    def forward(self, pressure_seq: torch.Tensor, boxes_seq: torch.Tensor):
        """
        Args:
            pressure_seq: [B, T, H, W] 原始压力序列
            boxes_seq: [B, T, max_N, 4] 真实的足迹框位置，这里不使用
        Returns:
            List[List[Dict]: 每个样本每个人的SMPL参数
        """
        B, T, H, W = pressure_seq.shape
        max_N = boxes_seq.shape[2]
        boxes_seq = self.tracking_by_detection(pressure_seq, max_N)
        person_pressures, centers = extract_person_pressures(pressure_seq, boxes_seq)

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