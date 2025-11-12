import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv
from torch.utils.data import Dataset
import time

name_group_map = {
    '1': [1, 21, 22],
    '2': [2, 23, 24, 25, 34],
    '3': [3, 25, 26, 27, 43],
    '4': [4, 28, 42],
    '5': [5, 23, 26, 33, 35, 37, 41, 45, 47, 49, 50],
    '6': [6, 29, 30],
    '7': [7, 31, 32, 44],
    '8': [8, 29, 33],
    '9': [9, 34, 35],
    '10': [10, 28, 31, 36],
    '11': [11, 38, 39],
    '12': [12, 24, 27, 37],
    '13': [13, 40, 46, 50],
    '14': [14, 41, 49, 51, 52],
    '15': [15, 32, 36, 42],
    '16': [16, 43, 44],
    '17': [17, 45, 46, 50],
    '18': [18, 21, 30, 38, 40, 48, 49, 51, 52],
    '19': [19, 22, 47],
    '20': [20, 39, 48],
}

subject_fold = [
    [['1', '6', '8', '19'], [1, 6, 8, 19, 22, 29]],
    [['7', '11', '16', '20'], [7, 11, 16, 20, 39, 44]],
]


class CarpetPressureDataset(Dataset):
    def __init__(self, cfgs, mode='train'):

        self.cfgs = cfgs
        self.dataset_name = cfgs['dataset_path']
        self.seqlen = cfgs['seqlen']  
        self.overlap = cfgs['overlap']  
        self.mid_frame = int(self.seqlen / 2)
        self.stride = max(int(self.seqlen * (1 - self.overlap) + 0.5), 1)
        self.model_type = cfgs['dataset_mode']
        self.curr_fold = cfgs['curr_fold']
        self.mode = mode
        self.normalize = cfgs['normalize']
        self.img_size = cfgs['img_size']

        self.segments = []
        self.db_segments = []

        self.data_len = 0

        self.data = {}
        self.info = {
            'name': [],
            'idx': [],
        }

        self._all_images = []
        self._all_boxes = []
        self._all_betas = []
        self._all_betas = []
        self._all_pose = []
        self._all_trans = []
        self._all_verts = []
        self._all_keypoints_3d = []

        self._current_global_offset = 0

        if self.model_type == 'unseen_group':
            for idx in range(1, 53):
                print(f'load {self.mode} dataset: {idx}')
                self._load_db(idx, self.mode)

        elif self.model_type == 'unseen_subject':
            if self.mode == 'train':
                name_list = subject_fold[self.curr_fold][0]
                train_group_list = set(range(1, 53))
                for name in name_list:
                    train_group_list = train_group_list - set(name_group_map[name])
                for idx in train_group_list:
                    print(f'load train dataset: {idx}')
                    self._load_db(idx)

            elif self.mode == 'eval':
                val_group_list = subject_fold[self.curr_fold][1]
                for idx in val_group_list:
                    print(f'load eval dataset: {idx}')
                    self._load_db(idx)

        concat_start_time = time.time()
        self.data = {
            'images': np.concatenate(self._all_images, axis=0),
            'box': np.concatenate(self._all_boxes, axis=0),
            'betas': np.concatenate(self._all_betas, axis=0),
            'pose': np.concatenate(self._all_pose, axis=0),
            'trans': np.concatenate(self._all_trans, axis=0),
            'verts': np.concatenate(self._all_verts, axis=0),
            'keypoints_3d': np.concatenate(self._all_keypoints_3d, axis=0)
        }
        self.data_len = self.data['images'].shape[0]
        concat_end_time = time.time()
        print(f'Concatenation finished. Total time: {concat_end_time - concat_start_time:.4f} seconds')
        print(f'Total data length: {self.data_len}')

        self.video_index_list, self.sample_lens = self.video_seg_window()

        if self.normalize:
            normalize_start_time = time.time()

            self.data['images'][self.data['images'] > 1.5e7] = 1.5e7
            self.data['images'] = self.data['images'] / (1.5e7 - np.min(self.data['images']))
            normalize_end_time = time.time()
            print(f"Normalization finished. Total time: {normalize_end_time - normalize_start_time:.4f} seconds")


    def _load_db(self, idx, mode=None):
        db = dict(np.load(os.path.join(self.dataset_name, f'data_{idx}.npz'), allow_pickle=True))
        segments_for_db = split_dataset(db['segments'], mode)

        for segment in segments_for_db:
            # import pdb; pdb.set_trace()
            self.info['name'].append(db['name'])
            self.info['idx'].append(idx)

            start, end = segment[0], segment[1]
            self._all_images.append(db['pressure'][start:end])
            self._all_boxes.append(db['box'][start:end])
            self._all_betas.append(db['betas'][start:end])
            self._all_pose.append(db['pose'][start:end])
            self._all_trans.append(db['trans'][start:end])
            self._all_verts.append(db['verts'][start:end])
            self._all_keypoints_3d.append(db['keypoints_3d'][start:end, :, :25, :])

            self.segments.append(np.array([self._current_global_offset, self._current_global_offset + (end - start)]))
            self.db_segments.append(segment) 
            self._current_global_offset += (end - start) 
        # db_process_time = time.time()
        # print(f"  DB {idx} process time: {db_process_time - db_load_time:.2f}")

    def __len__(self):
        return len(self.video_index_list)

    def __getitem__(self, index):

        start_idx, end_idx, len_judge = self.video_index_list[index]

        curr_frame_idx = [i for i in range(start_idx, end_idx)]
        item = {
            'curr_frame_idx': torch.tensor(curr_frame_idx),
            'images': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['images'])
            ).float(),
            'boxes': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['box'])
            ),
            'betas': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['betas'])
            ).float(),
            'pose': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['pose'])
            ).float(),
            'trans': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['trans'])
            ).float(),
            'gt_verts': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['verts'])
            ).float(),
            'gt_keypoints_3d': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, len_judge, self.data['keypoints_3d'])
            ).float(),
        }

        return item


    def get_sequence(self, start_index, end_index, len_judge, data):
        if len_judge:
            return data[start_index: end_index]
        else:
            index_list = [i for i in range(start_index, end_index)] + [end_index - 1 for _ in
                                                                       range(self.seqlen - (end_index - start_index))]
            return data[index_list]

    def get_single_item(self, index):
        pass

    def video_seg_window(self):

        index_list = []

        for segment in self.segments:

            if segment[1] - segment[0] >= self.seqlen:
                for i in range(segment[0], segment[1], self.stride):
                    if i + self.seqlen >= segment[1]:
                        index_list.append([segment[1] - self.seqlen, segment[1], 1])
                        break
                    else:
                        index_list.append([i, i + self.seqlen, 1])

        return index_list, len(index_list)


    def save_opt_results(self, curr_frame_idx, pose, shape, trans):

        start_index, end_index, len_judge = \
            curr_frame_idx[0], curr_frame_idx[1], curr_frame_idx[2]

        self.db['pose'][start_index: end_index] = pose[: end_index - start_index]
        self.db['betas'][start_index: end_index] = shape[: end_index - start_index]
        self.db['trans'][start_index: end_index] = trans[: end_index - start_index]

    def get_segments(self):
        return self.segments

    def get_data_len(self):
        return self.data_len


def split_dataset(segments, mode):
    TRAIN_RATIO = 0.8
    EVAL_RATIO = 0.1
    TEST_RATIO = 0.1
    result = []

    train_end_point = int(segments[-1][-1] * TRAIN_RATIO)
    eval_end_point = int(segments[-1][-1] * (TRAIN_RATIO + EVAL_RATIO))

    if mode == 'train':
        for segment in segments:
            if segment[1] <= train_end_point:
                result.append(segment)
            elif segment[0] < train_end_point < segment[1]:
                result.append([segment[0], train_end_point])
    elif mode == 'eval':
        for segment in segments:
            if segment[0] < train_end_point < segment[1]:
                result.append([train_end_point, min(segment[1], eval_end_point)])
            elif train_end_point <= segment[0] and segment[1] <= eval_end_point:
                result.append(segment)
            elif segment[0] < eval_end_point < segment[1]:
                result.append([max(train_end_point, segment[0]), eval_end_point])
    elif mode == 'test':
        for segment in segments:
            if eval_end_point < segment[0]:
                result.append(segment)
            elif segment[0] < eval_end_point < segment[1]:
                result.append([eval_end_point, segment[1]])
    else:
        result = segments

    return result
