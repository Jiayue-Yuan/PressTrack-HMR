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

        self.len_train_loader = 0

        self.train_loader = train_loader
        if self.train_loader is not None:
            self.len_train_loader = len(self.train_loader)

        self.val_loader = val_loader

        if self.mode == 'unseen_group':
            self.test_loader = test_loader

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

        self.val_segments = val_segments
        self.test_segments = test_segments

        self.val_results_record = np.zeros((len_val_set, 6))
        self.val_joints_record = np.zeros((len_val_set, self.max_N, 25, 3))
        if self.mode == 'unseen_group':
            self.test_results_record = np.zeros((len_test_set, 6))
            self.test_joints_record = np.zeros((len_test_set, self.max_N, 25, 3))

        self.test_time_list = []

    def fit(self):
        self.batch_iter = 0

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.train()
            val_loss = self.val('eval')

            torch.cuda.empty_cache()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss['mpjpe_wo_align'])

            if val_loss['mpjpe_wo_align'] < self.min_val_loss:
                self.min_val_loss = val_loss['mpjpe_wo_align']
                self.min_loss_dict = val_loss
                state_dict = translate_state_dict(self.model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(
                    state_dict,
                    self.checkpoints_path + '/' 'hps_' + f'{epoch}_losses_{np.round(self.min_val_loss, 2)}' + '.pth'
                )

            if epoch % 5 == 0:
                self.loss_record.plot(epoch)

        print_loss(0, loss_dict=self.min_loss_dict)


    def train(self):

        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            pressure_seq = batch['images'].to(self.device)  # [B, T, H, W]
            boxes_seq = batch['boxes'].to(self.device)  # [B, T, max_N, 4]

            outputs  = self.model(pressure_seq, boxes_seq) 

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

            losses, loss_dict = self.criterion(outputs, kp_3d, trans, pose, betas) 

            self.optimizer.zero_grad() 

            losses.backward() 

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

        if type == 'eval':
            loader = self.val_loader
            descriptor ='Validating'
        else:
            loader = self.test_loader
            descriptor = 'Testing'

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.type(torch.float32).detach().to(self.device) for k, v in batch.items()}
                pressure_seq = batch['images'].to(self.device)
                boxes_seq = batch['boxes'].to(self.device)
                outputs  = self.model(pressure_seq, boxes_seq)
                self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], batch['gt_verts'], outputs, type)

                kp_3d = batch['gt_keypoints_3d'].to(self.device)
                trans = batch['trans'].to(self.device)
                pose = batch['pose'].to(self.device)
                betas = batch['betas'].to(self.device)

                losses, loss_dict = self.criterion(outputs, kp_3d, trans, pose, betas)  
                print(print_loss(losses.item(), loss_dict))

            self.accel_evaluate(type)

        return self.print_metric(type)


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

                self.batch_evaluate(batch['curr_frame_idx'], batch['gt_keypoints_3d'], batch['gt_verts'], outputs, mode)

                index_list = batch['curr_frame_idx'][7].cpu().reshape(-1).type(torch.long).numpy()

                for b in range(len(outputs)):
                    for id in range(len(outputs[0])):
                        if outputs[b][id]:
                            for key in results:
                                results[key][index_list] = outputs[b][id][key].cpu().detach().numpy()

            self.accel_evaluate(mode)
            if mode == 'eval':
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segmemts': loader.dataset.db_segmemts,
                    'mpjpe_wo_align': self.val_results_record[:, 0],
                    'mpjpe': self.val_results_record[:, 1],
                    'mpjpe_pa': self.val_results_record[:, 2],
                    'mpve': self.val_results_record[:, 3],
                    'accel': self.val_results_record[:, 4] * 1000,
                    'accel_error': self.val_results_record[:, 5]
                })
            else:
                results.update({
                    'info': loader.dataset.info,
                    'segments': loader.dataset.segments,
                    'db_segmemts': loader.dataset.db_segmemts,
                    'mpjpe_wo_align': self.test_results_record[:, 0],
                    'mpjpe': self.test_results_record[:, 1],
                    'mpjpe_pa': self.test_results_record[:, 2],
                    'mpve': self.test_results_record[:, 3],
                    'accel': self.test_results_record[:, 4] * 1000,
                    'accel_error': self.test_results_record[:, 5]
                })
            np.savez(self.test_save_path + '/test.npz', **results)
            self.print_metric(mode)

        self.print_test_fps()


    def print_metric(self, mode):
        if mode == 'eval':
            segments = len(self.val_segments)

            loss_dict = {
                'mpjpe_wo_align': np.ma.masked_where(self.val_results_record[:, 0] == 0,
                                                     self.val_results_record[:, 0]).mean(),
                'mpjpe': np.ma.masked_where(self.val_results_record[:, 1] == 0,
                                            self.val_results_record[:, 1]).mean(),
                'mpjpe_pa': np.ma.masked_where(self.val_results_record[:, 2] == 0,
                                               self.val_results_record[:, 2]).mean(),
                'mpve': np.ma.masked_where(self.val_results_record[:, 3] == 0,
                                           self.val_results_record[:, 3]).mean(),
                'accel': np.ma.masked_where(self.val_results_record[:, 4] == 0,
                                            self.val_results_record[:, 4]).mean() * 1000,
                'accel_error': np.mean(self.val_results_record[:, 5])
            }

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
                'mpjpe_pa': np.ma.masked_where(self.test_results_record[:, 2] == 0,
                                               self.test_results_record[:, 2]).mean(),
                'mpve': np.ma.masked_where(self.test_results_record[:, 3] == 0,
                                           self.test_results_record[:, 3]).mean(),
                'accel': np.ma.masked_where(self.test_results_record[:, 4] == 0,
                                            self.test_results_record[:, 4]).mean() * 1000,
                'accel_error': np.mean(self.test_results_record[:, 5])

            }

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

    def accel_evaluate(self, mode):
        if mode == 'eval':
            segments = self.val_segments
            joints = self.val_joints_record
        else:
            segments = self.test_segments
            joints = self.test_joints_record

        for segment in segments:
            segment_joints = joints[segment[0]:segment[1]]  # (len, max_N, 25, 3)
            accel = []

            # 遍历每个人
            for person_idx in range(self.max_N):
                person_joints = segment_joints[:, person_idx, :, :]  # (len, 25, 3)
                is_person_present = np.any(person_joints != 0)
                if is_person_present:
                    non_zero_slices_mask = ~np.all(person_joints == 0, axis=(1, 2))
                    start_index = -1
                    for i in range(len(non_zero_slices_mask)):
                        if non_zero_slices_mask[i] and start_index == -1:
                            start_index = i
                        elif not non_zero_slices_mask[i] and start_index != -1:
                            end_index = i
                            if end_index - start_index >= 3:
                                accel.extend(compute_accel(person_joints[start_index:end_index]).tolist())
                            start_index = -1

            if mode == 'eval':
                self.val_results_record[segment[0]:segment[1], 4] = np.mean(accel)
            else:
                self.test_results_record[segment[0]:segment[1], 4] = np.mean(accel)

    def batch_evaluate(self, index, gt_keypoints_3d, gt_verts, outputs, mode):
        index = index[:, 7].reshape(-1).type(torch.long).cpu()  # [B]
        batch_size = len(outputs)
        all_errors_wo_align = torch.zeros([batch_size])
        all_mpjpe = torch.zeros([batch_size])
        all_pa_mpjpe = torch.zeros([batch_size])
        all_pve = torch.zeros([batch_size])
        pred_j3ds = torch.zeros_like(gt_keypoints_3d[:, 7]) # [B, max_N, 25, 3]
        for b in range(batch_size):
            # 获取当前序列的预测和真实数据，gt取中间帧
            pred_people = outputs[b]  # [{}, {person1}, {person2}]
            gt_kp3d_b = gt_keypoints_3d[b, 7]  # [max_N, 25, 3]
            gt_verts_b = gt_verts[b, 7]

            # 找出当前序列的有效真实人数
            non_zero_per_frame = gt_kp3d_b.abs().sum(dim=-1) > 0  # [max_N, 25]
            valid_gt_mask = torch.all(non_zero_per_frame, dim=-1)  # [max_N]
            N_gt = int(valid_gt_mask.sum())
            if N_gt == 0:
                continue

            # 提取真实人的数据
            gt_verts_valid = gt_verts_b[valid_gt_mask]  # [N_gt, 6890, 3]
            gt_kp3d_valid = gt_kp3d_b[valid_gt_mask]  # [N_gt, 25, 3]

            # 收集有效预测
            valid_preds = [p for p in pred_people if 'verts' in p and 'kp_3d' in p] # [{person1},{person2}]
            valid_index = [i for i, p in enumerate(pred_people) if 'verts' in p and 'kp_3d' in p] # [1,2]
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

            total_errors_wo_align, total_mpjpe, total_pa_mpjpe, total_pve = 0.0, 0.0, 0.0, 0.0
            total_matched_pairs = 0

            # 计算匹配对的metric
            for i, j in zip(row_ind, col_ind):
                # 获取匹配的预测和真实数据
                pred_data = valid_preds[i]
                gt_kp3d_j = gt_kp3d_valid[j]  # [25, 3]
                gt_verts_j = gt_verts_valid[j] # [6890, 3]

                errors_wo_align, mpjpe, pa_mpjpe, pve = self.evaluate(gt_kp3d_j, pred_data['kp_3d'][0], gt_verts_j, pred_data['verts'][0])
                pred_j3ds[b, valid_index[j]] = pred_data['kp_3d'][0] # 要把j对应回原始id

                total_errors_wo_align += errors_wo_align
                total_mpjpe += mpjpe
                total_pa_mpjpe += pa_mpjpe
                total_pve += pve
                total_matched_pairs += 1

            # 平均
            if total_matched_pairs > 0:
                all_errors_wo_align[b] = torch.tensor([total_errors_wo_align / total_matched_pairs])
                all_mpjpe[b] = torch.tensor([total_mpjpe / total_matched_pairs])
                all_pa_mpjpe[b] = torch.tensor([total_pa_mpjpe / total_matched_pairs])
                all_pve[b] = torch.tensor([total_pve / total_matched_pairs])

        if mode == 'eval':
            self.val_results_record[index, :4] = torch.stack([
                all_errors_wo_align, all_mpjpe, all_pa_mpjpe, all_pve
            ], dim=1).detach().cpu().numpy()
            self.val_joints_record[index] = pred_j3ds.detach().cpu().numpy()
        else:
            self.test_results_record[index, :4] = torch.stack([
                all_errors_wo_align, all_mpjpe, all_pa_mpjpe, all_pve
            ], dim=1).detach().cpu().numpy()
            self.test_joints_record[index] = pred_j3ds.detach().cpu().numpy()


    def evaluate(self, target_j3ds, pred_j3ds, gt_verts, pred_verts):
        errors_wo_align = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1) * 1000

        pred_pelvis = (pred_j3ds[[1], :] + pred_j3ds[[2], :]) / 2.0
        target_pelvis = (target_j3ds[[1], :] + target_j3ds[[2], :]) / 2.0

        pred_j3ds_p = pred_j3ds - pred_pelvis
        target_j3ds_p = target_j3ds - target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds_p - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds_p.unsqueeze(0), target_j3ds_p.unsqueeze(0))
        errors_pa = torch.sqrt(((S1_hat[0] - target_j3ds_p) ** 2).sum(dim=-1)).mean(dim=-1)

        m2mm = 1000

        pve = compute_error_verts(target_verts=gt_verts.unsqueeze(0), pred_verts=pred_verts.unsqueeze(0))[0] * m2mm
        mpjpe = errors * m2mm
        pa_mpjpe = errors_pa * m2mm

        return errors_wo_align, mpjpe, pa_mpjpe, pve
