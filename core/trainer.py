import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from utils.others.utils import step_learning_rate, cosine_learning_rate, translate_state_dict
from utils.others.loss_record import print_loss
from core.evaluate import *

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

import smplx

class Trainer():
    def __init__(self,
                 args,
                 model,
                 optimizer,
                 criterion,
                 loss_record,
                 lr_scheduler=None,
                 writer=None,
                 checkpoints_path='',
                 exp_mode='unseen_subject',
                 curr_fold=0,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 len_val_set=0,
                 val_segments=None,
                 len_test_set=0,
                 test_segments=None,
                 test_save_path=None,
                 device='cuda'):

        self.max_N = 3
        self.args = args
        self.epoch = 0
        self.epochs = args.epochs
        self.mode = exp_mode
        self.curr_fold = curr_fold
        self.loss_record = loss_record
        self.seqlen = args.seqlen
        self.mid_frame = args.mid_frame

        self.len_train_loader = 0
        self.len_val_set = len_val_set
        self.len_test_set = len_test_set

        self.train_loader = train_loader
        if self.train_loader is not None:
            self.len_train_loader = len(self.train_loader)

        self.gt_joints_record = np.zeros((max(self.len_test_set, self.len_val_set), self.max_N, 25, 3))

        self.val_loader = val_loader
        self.val_segments = val_segments
        self.val_results_record = np.zeros((self.len_val_set, 6))
        self.val_joints_record = np.zeros((self.len_val_set, self.max_N, 25, 3))
        if self.mode == 'unseen_sequence':
            self.test_loader = test_loader
            self.test_segments = test_segments
            self.test_results_record = np.zeros((self.len_test_set, 6))
            self.test_joints_record = np.zeros((self.len_test_set, self.max_N, 25, 3))

        self.batch_iter = 0
        self.total_batch = self.len_train_loader * self.epochs

        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion.to(device)

        self.writer = writer
        self.device = device

        self.min_val_loss = 1e5
        self.min_loss_dict = {}
        self.checkpoints_path = checkpoints_path
        self.test_save_path = test_save_path

        self.scaler = GradScaler()

        if self.train_loader is None:
            self.test_time_list = []
            self.accel_record = []
            self.w_mpjpe_record = []
            self.wa_mpjpe_record = []
            self.rte = []
            self.jitter = []
            self.foot_sliding = []
            self.gt_trans_record = np.zeros((max(self.len_test_set, self.len_val_set), self.max_N, 3))
            self.gt_verts_record = np.zeros((max(self.len_test_set, self.len_val_set), self.max_N, 6890, 3))
            self.val_trans_record = np.zeros((self.len_val_set, self.max_N, 3))
            self.val_verts_record = np.zeros((self.len_val_set, self.max_N, 6890, 3))
            if self.mode == 'unseen_group':
                self.test_trans_record = np.zeros((self.len_test_set, self.max_N, 3))
                self.test_verts_record = np.zeros((self.len_test_set, self.max_N, 6890, 3))

        self.smpl_model = smplx.create(
            model_path='/workspace/3d_pose_estimation/smpl/SMPL_NEUTRAL.pkl',
            model_type='smpl',
        )
        self.smpl_model.to(self.device)


    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.train()
            val_loss = self.val('eval')

            torch.cuda.empty_cache()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss['t_mpjpe'])

            if val_loss['t_mpjpe'] < self.min_val_loss:
                self.min_val_loss = val_loss['t_mpjpe']
                self.min_loss_dict = val_loss
                state_dict = translate_state_dict(self.model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(
                    state_dict,
                    self.checkpoints_path + '/' 'hps_' + f'{epoch}_losses_{np.round(self.min_val_loss, 3)}' + '.pth'
                )
                self.loss_record.plot(epoch)

        print_loss(0, loss_dict=self.min_loss_dict)


    def train(self):
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            pressure_seq = batch['images'].to(self.device)  # [B, T, H, W]
            boxes_seq = batch['boxes'].to(self.device)  # [B, T, max_N, 4]

            outputs  = self.model(pressure_seq, boxes_seq) #  [[{}, {}, {}] * B]  0.2s

            kp_3d = batch['gt_keypoints_3d'].to(self.device)
            trans = batch['trans'].to(self.device)
            pose = batch['pose'].to(self.device)
            betas = batch['betas'].to(self.device)


            if self.args.cosine:
                lr = cosine_learning_rate(
                    self.args, self.epoch, self.batch_iter, self.optimizer, self.len_train_loader
                )
            else:
                lr = step_learning_rate(
                    self.args, self.epoch, self.batch_iter, self.optimizer, self.len_train_loader
                )

            losses, loss_dict = self.criterion(outputs, kp_3d, trans, pose, betas)  # 0.03s

            self.optimizer.zero_grad() # 0.04s

            losses.backward() # 0.05s

            self.optimizer.step()

            batch_time = time.time() - batch_start

            self.batch_iter += 1

            self.loss_record.update(losses.item(), loss_dict, 'train')

            print(
                "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] {} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    batch_idx + 1,
                    self.len_train_loader,
                    self.batch_iter,
                    self.total_batch,
                    print_loss(losses.item(), loss_dict),
                    lr,
                    batch_time
                ))

            if batch_idx % 500 == 0:
                self.writer.info(
                    "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] {} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.epoch,
                        self.epochs,
                        batch_idx + 1,
                        self.len_train_loader,
                        self.batch_iter,
                        self.total_batch,
                        print_loss(losses.item(), loss_dict),
                        lr,
                        batch_time,
                    ))
            self.loss_record.update(losses.item(), loss_dict, 'train')


    def val(self, type='eval'):
        self.model.eval()
        loader = self.val_loader if type == 'eval' else self.test_loader

        total_val_dict = {
            't_mpjpe': 0.0,
            'kp3d': 0.0,
            'trans': 0.0,
            'pose': 0.0,
            'shape': 0.0,
        }
        valid_batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.type(torch.float32).detach().to(self.device) for k, v in batch.items()}
                pressure_seq = batch['images'].to(self.device)
                boxes_seq = batch['boxes'].to(self.device)
                outputs = self.model(pressure_seq, boxes_seq)

                kp_3d = batch['gt_keypoints_3d'].to(self.device)
                trans = batch['trans'].to(self.device)
                pose = batch['pose'].to(self.device)
                betas = batch['betas'].to(self.device)

                losses, batch_val_dict = self.criterion(outputs, kp_3d, trans, pose, betas)

                # 累加批次指标（仅当 batch_val_dict 有效时）
                if batch_val_dict is not None:
                    for k in total_val_dict.keys():
                        # 确保 batch_val_dict 包含目标键
                        if k in batch_val_dict:
                            total_val_dict[k] += batch_val_dict[k]
                    valid_batch_count += 1

        # 计算全验证集平均指标
        if valid_batch_count == 0:
            avg_val_dict = {
                't_mpjpe': 1e5,
                'kp3d': 1e5,
                'trans': 1e5,
                'pose': 1e5,
                'shape': 1e5 ,
            }
        else:
            avg_val_dict = {
                k: v / valid_batch_count
                for k, v in total_val_dict.items()
            }

            self.writer.info(
                "[Validating] Time: {}] {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    print_loss(losses.item(), avg_val_dict),
                ))

        return avg_val_dict


    def test(self):
        self.model.eval()

        if self.mode == 'unseen_subject':
            loader = self.val_loader
            mode = 'eval'
        else:
            loader = self.test_loader
            mode = 'test'

        length = loader.dataset.get_data_len()

        results = {
            'theta': np.zeros((length, self.max_N, 85)),
            'kp_3d': np.zeros((length, self.max_N, 25, 3))
        }

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader)):
                batch = {k: v.type(torch.float32).detach().to(self.device) for k, v in batch.items()}

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                pressure_seq = batch['images'].to(self.device)
                boxes_seq = batch['boxes'].to(self.device)
                outputs  = self.model(pressure_seq, boxes_seq)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                self.test_time_list.append(elapsed)

                batch['gt_verts'] = self.calculate_verts(batch['pose'], batch['betas'], batch['trans'])

                self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], batch['gt_verts'], batch['trans'], outputs, mode)

                for b in range(len(outputs)):
                    for id in range(len(outputs[0])):
                        if outputs[b][id]:
                            for key in results:
                                index = batch['curr_frame_idx'][b, self.mid_frame].cpu().reshape(-1).type(torch.long).numpy()
                                results[key][index] = outputs[b][id][key].cpu().detach().numpy()

            self.accel_evaluate(mode)
            self.global_evaluate(mode)

            if mode == 'eval':
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segments': loader.dataset.db_segments,
                    'mpjpe_wo_align': self.val_results_record[:, 0],
                    'mpjpe': self.test_results_record[:, 1],
                    'lower_mpjpe': self.test_results_record[:, 2],
                    'upper_mpjpe': self.test_results_record[:, 3],
                    'mpjpe_pa': self.test_results_record[:, 4],
                    'mpve': self.test_results_record[:, 5],
                })
            else:
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segments': loader.dataset.db_segments,
                    'mpjpe_wo_align': self.test_results_record[:, 0],
                    'mpjpe': self.test_results_record[:, 1],
                    'lower_mpjpe': self.test_results_record[:, 2],
                    'upper_mpjpe': self.test_results_record[:, 3],
                    'mpjpe_pa': self.test_results_record[:, 4],
                    'mpve': self.test_results_record[:, 5],
                })
            np.savez(self.test_save_path + '/test.npz', **results)

            self.print_metric(mode)

        self.print_test_fps()


    def demo(self, viz):

        self.model.eval()

        if self.mode == 'unseen_subject':
            loader = self.val_loader
            mode = 'eval'
        else:
            loader = self.test_loader
            mode = 'test'

        length = loader.dataset.get_data_len()

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader)):
                batch = {k: v.type(torch.float32).detach().to(self.device) for k, v in batch.items()}

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                pressure_seq = batch['images'].to(self.device)
                boxes_seq = batch['boxes'].to(self.device)
                outputs  = self.model(pressure_seq, boxes_seq) #  [[{}, {}, {}] * B]

                if viz:
                    # batch_size = len(outputs)
                    # results = {
                    #     'frame_idx': batch['curr_frame_idx'][:, self.mid_frame].cpu().numpy().astype(int),  # [B]
                    #     'pressure': pressure_seq[:, self.mid_frame].cpu().numpy(), # [B, H, W]
                    #     'shape': np.zeros((batch_size, self.max_N, 10), dtype=np.float32),
                    #     'pose': np.zeros((batch_size, self.max_N, 72), dtype=np.float32),
                    #     'trans': np.zeros((batch_size, self.max_N, 3), dtype=np.float32),
                    #     'gt_shape': batch['betas'][:, self.mid_frame].cpu().numpy(),
                    #     'gt_pose': batch['pose'][:, self.mid_frame].cpu().numpy(),
                    #     'gt_trans': batch['trans'][:, self.mid_frame].cpu().numpy(),
                    # }
                    #
                    # for frame_idx, batch_output in enumerate(outputs):
                    #     for person_idx, person_data in enumerate(batch_output):
                    #         if person_data:
                    #             results['shape'][frame_idx, person_idx] = person_data['theta'][0, 75:].cpu().numpy()
                    #             results['pose'][frame_idx, person_idx] = person_data['theta'][0, 3:75].cpu().numpy()
                    #             results['trans'][frame_idx, person_idx] = person_data['theta'][0, :3].cpu().numpy()
                    # np.savez(f'viz_batch_{batch_idx}_data.npz', **results)
                    for b, pred in enumerate(outputs):
                        pressure_data = pressure_seq[b, self.mid_frame].cpu().numpy() # [H, W]
                        frame_idx = batch['curr_frame_idx'][b, self.mid_frame].cpu().numpy().astype(int)

                        # 准备绘图
                        # 创建图形和子图，1行2列布局
                        fig = plt.figure(figsize=(15, 10))
                        fig.subplots_adjust(left=0.05, right=1, bottom=0, top=1)
                        gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.01, hspace=0.01)
                        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4"])

                        # 左侧子图：压力图
                        ax1 = fig.add_subplot(gs[0])  # 1行2列的第一个子图
                        im = ax1.imshow(pressure_data, interpolation='nearest', cmap='viridis')
                        ax1.axis('off')

                        # 右侧子图：3D点云图
                        ax2 = fig.add_subplot(gs[1], projection='3d')
                        for person_data in pred:
                            if 'verts' in person_data:
                                vertices_np = person_data['verts'][0].cpu().numpy()
                                # 绘制当前人的点云（使用不同颜色区分）
                                ax2.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], s=1)

                                # 地毯四个角在世界坐标系下的坐标
                                corners = [(-0.03, 1.818), (1.17, 1.83), (-0.06, -0.565), (1.135, -0.567)]
                                x0, y0 = corners[0]
                                x1, y1 = corners[1]
                                x2, y2 = corners[2]
                                x3, y3 = corners[3]

                                # 计算地毯在世界坐标系中的边界
                                min_x = min(x0, x1, x2, x3)
                                max_x = max(x0, x1, x2, x3)
                                min_y = min(y0, y1, y2, y3)
                                max_y = max(y0, y1, y2, y3)

                                # 绘制地毯边界（在3D空间中，z=0平面）
                                ax2.plot([x0, x1, x3, x2, x0], [y0, y1, y3, y2, y0], [0, 0, 0, 0, 0], 'r-', linewidth=2,
                                         label='地毯边界')

                                # 定义一个函数，将压力图上的点映射到3D空间
                                def pressure_to_3d(u, v, z=-0.01):
                                    """
                                    将压力图上的坐标(u, v)映射到3D空间

                                    参数:
                                        u, v: 压力图上的像素坐标
                                        pressure_val: 压力值，用于可视化
                                        z: 3D空间中的高度，略高于0以避免与地面重叠

                                    返回:
                                        (x, y, z): 3D空间坐标
                                    """
                                    # 获取压力图尺寸
                                    h, w = pressure_data.shape

                                    # 将像素坐标标准化到[0,1]范围
                                    u_norm = u / (w - 1)
                                    v_norm = v / (h - 1)

                                    # 双线性插值计算对应的世界坐标
                                    # 计算左上角到右下角的插值
                                    x_left = x1 + v_norm * (x3 - x1)
                                    x_right = x0 + v_norm * (x2 - x0)
                                    x = x_left + u_norm * (x_right - x_left)

                                    y_left = y1 + v_norm * (y3 - y1)
                                    y_right = y0 + v_norm * (y2 - y0)
                                    y = y_left + u_norm * (y_right - y_left)

                                    return (x, y, z)

                                # 示例：在压力图上选择一些点并映射到3D空间
                                # 这里选择压力值较大的点作为示例
                                threshold = 0.33 * pressure_data.max()  # 阈值，选择压力较大的点
                                high_pressure_points = np.where(pressure_data > threshold)

                                # 映射并绘制这些点
                                if len(high_pressure_points[0]) > 0:
                                    # 为了避免点太多，这里只取前500个点
                                    num_points = min(500, len(high_pressure_points[0]))
                                    for i in range(num_points):
                                        v, u = high_pressure_points[0][i], high_pressure_points[1][i]
                                        pressure_val = pressure_data[v, u]
                                        x, y, z = pressure_to_3d(u, v, pressure_val, z=0.02)

                                        # 在右侧3D图上绘制这些点，使用压力值作为颜色
                                        ax2.scatter(x, y, z, c=[pressure_val], cmap='viridis', s=50, alpha=0.8)

                        ax2.set_zlim([0, -1.8])
                        ax2.set_xlim([1.2, -0.1])  # 1.3
                        ax2.set_ylim([-0.57, 1.83])  # 2.4
                        ax2.set_box_aspect([1.3, 2.4, 1.8])

                        # 保存当前帧图像
                        output_dir = '/workspace/PressTrack-HMR/viz/track/1_person'
                        os.makedirs(output_dir, exist_ok=True)
                        frame_path = os.path.join(output_dir, f'{frame_idx:06d}.png')
                        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)  # 关闭图形，释放内存


    def calculate_verts(self, pose, betas, trans):
        """
        计算batch内多个人的SMPL顶点
        Args:
            pose: torch.Tensor, 形状 [batch_size, num_persons, 72]，
            betas: torch.Tensor, 形状 [batch_size, num_persons, 10]
            trans: torch.Tensor, 形状 [batch_size, num_persons, 3]
        Returns:
            verts: torch.Tensor, 形状 [batch_size, num_persons, 6890, 3]
        """
        batch_size, num_persons, _ = pose.shape  # num_persons=3
        verts_list = []

        # 遍历每个batch内的每个人
        for person_idx in range(num_persons):
            # 提取当前人的姿态、形状、平移参数：[batch_size, 72]/[batch_size,10]/[batch_size,3]
            pose_person = pose[:, person_idx, :]  # [64,72]
            betas_person = betas[:, person_idx, :]  # [64,10]
            trans_person = trans[:, person_idx, :]  # [64,3]

            # 过滤无人的情况（pose全0表示无人）
            # 计算每个样本是否为有效人体（pose非全0）
            is_valid = torch.any(pose_person != 0, dim=-1)  # [batch_size]

            # 初始化当前人的顶点（默认全0，对应无人的情况）
            verts_person = torch.zeros(
                batch_size, 6890, 3,
                dtype=pose.dtype, device=pose.device
            )

            # 仅对有效人体计算SMPL顶点
            if torch.any(is_valid):
                # 提取有效样本的参数
                pose_valid = pose_person[is_valid]  # [num_valid,72]
                betas_valid = betas_person[is_valid]  # [num_valid,10]
                trans_valid = trans_person[is_valid]  # [num_valid,3]

                # 前向计算SMPL
                smpl_output = self.smpl_model(
                    betas=betas_valid,
                    body_pose=pose_valid[:, 3:],  # 身体姿态：[num_valid,69]
                    global_orient=pose_valid[:, :3],  # 全局旋转：[num_valid,3]
                    return_verts=True
                )

                # 加上平移，并赋值回对应位置
                verts_valid = smpl_output.vertices + trans_valid.unsqueeze(1)  # [num_valid,6890,3]
                verts_person[is_valid] = verts_valid

            verts_list.append(verts_person)  # 每个元素：[64,6890,3]

        # 拼接所有人的顶点：[batch_size, num_persons, 6890, 3]
        verts = torch.stack(verts_list, dim=1)

        return verts

    def print_metric(self, mode):
        if mode == 'eval':
            loss_dict = {
                'mpjpe_wo_align': np.ma.masked_where(self.val_results_record[:, 0] == 0,
                                                     self.val_results_record[:, 0]).mean(),
                'mpjpe': np.ma.masked_where(self.val_results_record[:, 1] == 0,
                                            self.val_results_record[:, 1]).mean(),
                'lower_mpjpe': np.ma.masked_where(self.test_results_record[:, 2] == 0,
                                               self.test_results_record[:, 2]).mean(),
                'upper_mpjpe': np.ma.masked_where(self.test_results_record[:, 3] == 0,
                                           self.test_results_record[:, 3]).mean(),
                'mpjpe_pa': np.ma.masked_where(self.test_results_record[:, 4] == 0,
                                               self.test_results_record[:, 4]).mean(),
                'mpve': np.ma.masked_where(self.test_results_record[:, 5] == 0,
                                           self.test_results_record[:, 5]).mean(),
            }
            if self.train_loader is None:
                loss_dict.update({
                    'accel': sum(self.accel_record) / len(self.accel_record),
                    'W_MPJPE_100': sum(self.w_mpjpe_record) / len(self.w_mpjpe_record),
                    'WA_MPJPE_100': sum(self.wa_mpjpe_record) / len(self.wa_mpjpe_record),
                    'RTE': sum(self.rte) / len(self.rte),
                    'Jitter': sum(self.jitter) / len(self.jitter),
                    'FS': sum(self.foot_sliding) / len(self.foot_sliding),
                })

            print(
                "[Validating] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

            self.writer.info(
                "[Validating] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))
        else:
            segments = len(self.test_segments)

            loss_dict = {
                'mpjpe_wo_align': np.ma.masked_where(self.test_results_record[:, 0] == 0,
                                                     self.test_results_record[:, 0]).mean(),
                'mpjpe': np.ma.masked_where(self.test_results_record[:, 1] == 0,
                                            self.test_results_record[:, 1]).mean(),
                'lower_mpjpe': np.ma.masked_where(self.test_results_record[:, 2] == 0,
                                               self.test_results_record[:, 2]).mean(),
                'upper_mpjpe': np.ma.masked_where(self.test_results_record[:, 3] == 0,
                                           self.test_results_record[:, 3]).mean(),
                'mpjpe_pa': np.ma.masked_where(self.test_results_record[:, 4] == 0,
                                               self.test_results_record[:, 4]).mean(),
                'mpve': np.ma.masked_where(self.test_results_record[:, 5] == 0,
                                           self.test_results_record[:, 5]).mean(),
            }
            if self.accel_record and self.w_mpjpe_record and self.wa_mpjpe_record:
                loss_dict.update({
                    'accel': sum(self.accel_record) / len(self.accel_record),
                    'W_MPJPE_100': sum(self.w_mpjpe_record) / len(self.w_mpjpe_record),
                    'WA_MPJPE_100': sum(self.wa_mpjpe_record) / len(self.wa_mpjpe_record),
                    'RTE': sum(self.rte) / len(self.rte),
                    'Jitter': sum(self.jitter) / len(self.jitter),
                    'FS': sum(self.foot_sliding) / len(self.foot_sliding),
                })

            print(
                "[Testing] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

            self.writer.info(
                "[Testing] Time: {} Epoch: [{}/{}]  {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.epoch,
                    self.epochs,
                    print_loss(0, loss_dict),
                ))

        self.loss_record.update(None, loss_dict, mode)

        return loss_dict

    def print_test_fps(self):
        fps = (len(self.test_time_list) * 16) / sum(self.test_time_list)
        print(
            f'Done image [{len(self.test_time_list) * 16}], '
            f'fps: {fps:.1f} img / s, '
            f'times per image: {1000 / fps:.1f} ms / img',
            flush=True)

    def global_evaluate(self, mode, chunk_length=100):
        gt_joints = self.gt_joints_record
        gt_trans = self.gt_trans_record
        gt_verts = self.gt_verts_record
        if mode == 'eval':
            segments = self.val_segments
            joints = self.val_joints_record
            trans = self.val_trans_record
            verts = self.val_verts_record
        else:
            segments = self.test_segments
            joints = self.test_joints_record
            trans = self.test_trans_record
            verts = self.test_verts_record

        m2mm = 1000

        for person_idx in range(self.max_N):
            for seg in segments:
                seg_start = seg[0] + self.mid_frame
                seg_end = seg[1] - (self.seqlen - self.mid_frame -1)
                pred_j3d_glob = joints[seg_start:seg_end, person_idx]  # (len, 25, 3)
                target_j3d_glob = gt_joints[seg_start:seg_end, person_idx]  # (len, 25, 3)
                is_person_present = np.all(pred_j3d_glob != 0) and np.all(target_j3d_glob != 0)

                if is_person_present:
                    num_frames = pred_j3d_glob.shape[0]
                    for start in range(0, num_frames, chunk_length):
                        end = min(num_frames, start + chunk_length)
                        target_j3d = torch.as_tensor(target_j3d_glob[start:end], dtype=torch.float32)
                        pred_j3d = torch.as_tensor(pred_j3d_glob[start:end], dtype=torch.float32)

                        w_j3d = first_align_joints(target_j3d, pred_j3d)
                        wa_j3d = global_align_joints(target_j3d, pred_j3d)

                        w_jpe = compute_jpe(target_j3d, w_j3d) * m2mm
                        wa_jpe = compute_jpe(target_j3d, wa_j3d) * m2mm

                        self.w_mpjpe_record.extend(w_jpe.tolist())
                        self.wa_mpjpe_record.extend(wa_jpe.tolist())

                    # Additional metrics
                    target_trans = torch.as_tensor(gt_trans[seg_start:seg_end, person_idx], dtype=torch.float32)
                    target_verts = torch.as_tensor(gt_verts[seg_start:seg_end, person_idx], dtype=torch.float32)
                    pred_trans = torch.as_tensor(trans[seg_start:seg_end, person_idx], dtype=torch.float32)
                    pred_verts = torch.as_tensor(verts[seg_start:seg_end, person_idx], dtype=torch.float32)

                    # (RTE in %) normalized by the actual displacement of the person.
                    rte = compute_rte(target_trans, pred_trans) * 1e2
                    rte = rte[np.isfinite(rte)] # 过滤掉可能的nan值
                    # the motion in the world coordinate system in 10m/s^3
                    jitter = compute_jitter(torch.as_tensor(pred_j3d_glob, dtype=torch.float32), fps=15)
                    # foot sliding during the contact (FS in mm)
                    foot_sliding = compute_foot_sliding(target_verts, pred_verts) * m2mm

                    self.rte.extend(rte.tolist())
                    self.jitter.extend(jitter.tolist())
                    self.foot_sliding.extend(foot_sliding.tolist())


    def accel_evaluate(self, mode, fps = 15):
        if mode == 'eval':
            segments = self.val_segments
            joints = self.val_joints_record
        else:
            segments = self.test_segments
            joints = self.test_joints_record

        for person_idx in range(self.max_N):
            for seg in segments:
                seg_start = seg[0] + self.mid_frame
                seg_end = seg[1] - (self.seqlen - self.mid_frame -1)
                person_joints = joints[seg_start:seg_end, person_idx]  # (len, 25, 3)
                is_person_present = np.all(person_joints != 0)
                if is_person_present:
                    if len(person_joints) >= 3:
                        accel = compute_accel(person_joints) * (fps ** 2)  # meters per frame^s to meters per s^2
                        self.accel_record.extend(accel.tolist())


    def batch_evaluate(self, index, gt_keypoints_3d, gt_verts, gt_trans, outputs, mode):
        index = index[:, self.mid_frame].reshape(-1).type(torch.long).cpu()  # [B]
        batch_size = len(outputs)
        all_errors_wo_align = torch.zeros([batch_size])
        all_mpjpe = torch.zeros([batch_size])
        all_lower_mpjpe = torch.zeros([batch_size])
        all_upper_mpjpe = torch.zeros([batch_size])
        all_pa_mpjpe = torch.zeros([batch_size])
        all_pve = torch.zeros([batch_size])

        pred_j3ds = torch.zeros_like(gt_keypoints_3d) # [B, max_N, 25, 3]
        pred_trans = torch.zeros_like(gt_trans)  # [B, max_N, 3]
        pred_verts = torch.zeros_like(gt_verts)  # [B, max_N, 6890, 3]

        for b in range(batch_size):
            pred_people = outputs[b]  # [{}, {person1}, {person2}]
            gt_kp3d_b = gt_keypoints_3d[b]  # [max_N, 25, 3]
            gt_verts_b = gt_verts[b]

            non_zero_per_frame = gt_kp3d_b.abs().sum(dim=-1) > 0  # [max_N, 25]
            valid_gt_mask = torch.all(non_zero_per_frame, dim=-1)  # [max_N]
            N_gt = int(valid_gt_mask.sum())
            if N_gt == 0:
                continue

            gt_verts_valid = gt_verts_b[valid_gt_mask]  # [N_gt, 6890, 3]
            gt_kp3d_valid = gt_kp3d_b[valid_gt_mask]  # [N_gt, 25, 3]

            valid_preds = [p for p in pred_people if 'verts' in p and 'kp_3d' in p] # [{person1},{person2}]
            valid_index = [i for i, p in enumerate(pred_people) if 'verts' in p and 'kp_3d' in p] # [1,2]
            N_pred = len(valid_preds)
            if N_pred == 0:
                continue

            if N_pred == N_gt == 1:
                row_ind = np.array([0])
                col_ind = np.array([0])
            else:
                cost_matrix = torch.zeros((N_pred, N_gt), device=gt_kp3d_b.device)
                for i, pred in enumerate(valid_preds):
                    for j in range(N_gt):
                        pred_kp = pred['kp_3d'][0]  # [25, 3]
                        gt_kp = gt_kp3d_valid[j]  # [25, 3]
                        mpjpe = torch.norm(pred_kp - gt_kp, p=2, dim=-1).mean()
                        cost_matrix[i, j] = mpjpe

                cost_matrix_np = cost_matrix.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            total_errors_wo_align, total_mpjpe, total_lower_mpjpe, total_upper_mpjpe, total_pa_mpjpe, total_pve = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            total_matched_pairs = 0

            for i, j in zip(row_ind, col_ind):
                pred_data = valid_preds[i]
                gt_kp3d_j = gt_kp3d_valid[j]  # [25, 3]
                gt_verts_j = gt_verts_valid[j] # [6890, 3]

                errors_wo_align, mpjpe, lower_mpjpe, upper_mpjpe, pa_mpjpe, pve = self.evaluate(gt_kp3d_j, pred_data['kp_3d'][0], gt_verts_j, pred_data['verts'][0])

                try:
                    pred_j3ds[b, valid_index[j]] = pred_data['kp_3d'][0]
                    pred_trans[b, valid_index[j]] = pred_data['theta'][0, :3]
                    pred_verts[b, valid_index[j]] = pred_data['verts'][0]
                except IndexError:
                    self.writer.info('IndexError: list index out of range')
                    return

                total_errors_wo_align += errors_wo_align
                total_mpjpe += mpjpe
                total_lower_mpjpe += lower_mpjpe
                total_upper_mpjpe += upper_mpjpe
                total_pa_mpjpe += pa_mpjpe
                total_pve += pve
                total_matched_pairs += 1

            if total_matched_pairs > 0:
                all_errors_wo_align[b] = torch.tensor([total_errors_wo_align / total_matched_pairs])
                all_mpjpe[b] = torch.tensor([total_mpjpe / total_matched_pairs])
                all_lower_mpjpe[b] = torch.tensor([total_lower_mpjpe / total_matched_pairs])
                all_upper_mpjpe[b] = torch.tensor([total_upper_mpjpe / total_matched_pairs])
                all_pa_mpjpe[b] = torch.tensor([total_pa_mpjpe / total_matched_pairs])
                all_pve[b] = torch.tensor([total_pve / total_matched_pairs])

        self.gt_joints_record[index] = gt_keypoints_3d[:].detach().cpu().numpy()
        if mode == 'eval':
            self.val_results_record[index, :6] = torch.stack([
                all_errors_wo_align, all_mpjpe, all_lower_mpjpe, all_upper_mpjpe, all_pa_mpjpe, all_pve
            ], dim=1).detach().cpu().numpy()
            self.val_joints_record[index] = pred_j3ds.detach().cpu().numpy()
        else:
            self.test_results_record[index, :6] = torch.stack([
                all_errors_wo_align, all_mpjpe, all_lower_mpjpe, all_upper_mpjpe, all_pa_mpjpe, all_pve
            ], dim=1).detach().cpu().numpy()
            self.test_joints_record[index] = pred_j3ds.detach().cpu().numpy()

        if self.train_loader is None:
            self.gt_trans_record[index] = gt_trans[:].detach().cpu().numpy()
            self.gt_verts_record[index] = gt_verts[:].detach().cpu().numpy()
            if mode == 'eval':
                self.val_trans_record[index] = pred_trans.detach().cpu().numpy()
                self.val_verts_record[index] = pred_verts.detach().cpu().numpy()
            else:
                self.test_trans_record[index] = pred_trans.detach().cpu().numpy()
                self.test_verts_record[index] = pred_verts.detach().cpu().numpy()

    def evaluate(self, target_j3ds, pred_j3ds, gt_verts, pred_verts):
        # import pdb; pdb.set_trace()
        errors_wo_align = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        pred_pelvis = (pred_j3ds[[1], :] + pred_j3ds[[2], :]) / 2.0
        target_pelvis = (target_j3ds[[1], :] + target_j3ds[[2], :]) / 2.0

        pred_j3ds_p = pred_j3ds - pred_pelvis
        target_j3ds_p = target_j3ds - target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds_p - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)
        lower_body_ids = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        upper_body_ids = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        lower_errors = torch.sqrt(((pred_j3ds_p[lower_body_ids, :] - target_j3ds_p[lower_body_ids, :]) ** 2).sum(dim=-1)).mean(dim=-1)
        upper_errors = torch.sqrt(((pred_j3ds_p[upper_body_ids, :] - target_j3ds_p[upper_body_ids, :]) ** 2).sum(dim=-1)).mean(dim=-1)

        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds_p.unsqueeze(0), target_j3ds_p.unsqueeze(0))
        errors_pa = torch.sqrt(((S1_hat[0] - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)

        m2mm = 1000

        pve = compute_error_verts(target_verts=gt_verts.unsqueeze(0), pred_verts=pred_verts.unsqueeze(0))[0] * m2mm
        mpjpe = errors * m2mm
        lower_mpjpe = lower_errors * m2mm
        upper_mpjpe = upper_errors * m2mm
        pa_mpjpe = errors_pa * m2mm

        return errors_wo_align, mpjpe, lower_mpjpe, upper_mpjpe, pa_mpjpe, pve


