import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv
from torch.utils.data import Dataset
import time
import multiprocessing as mp
from functools import partial

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


def load_db_worker(idx, dataset_path, mode):
    db = np.load(os.path.join(dataset_path, f'data_{idx}.npz'),
                 allow_pickle=True, mmap_mode='r')

    segments_for_db = split_dataset(db['segments'], mode)
    data_parts = {
        'images': [], 'box': [], 'betas': [],
        'pose': [], 'trans': [], 'keypoints_3d': []
    }
    # 记录info和segments信息
    info_name = []
    info_idx = []
    segments = []
    db_segments = []
    current_offset = 0  # 单文件内的局部偏移

    for segment in segments_for_db:
        start, end = segment[0], segment[1]
        seg_len = end - start

        data_parts['images'].append(db['pressure'][start:end].astype(np.float32))
        data_parts['box'].append(db['box'][start:end].astype(np.float32))
        data_parts['betas'].append(db['betas'][start:end].astype(np.float32))
        data_parts['pose'].append(db['pose'][start:end].astype(np.float32))
        data_parts['trans'].append(db['trans'][start:end].astype(np.float32))
        data_parts['keypoints_3d'].append(db['keypoints_3d'][start:end, :, :25, :].astype(np.float32))

        info_name.extend([db['name']] * seg_len)
        info_idx.extend([idx] * seg_len)

        segments.append([current_offset, current_offset + seg_len])
        db_segments.append(segment)
        current_offset += seg_len

    # 拼接单文件数据
    result = {}
    total_len = 0
    for key in data_parts:
        result[key] = np.concatenate(data_parts[key], axis=0) if data_parts[key] else np.array([])
        total_len = len(result[key]) if key == 'images' else total_len

    result['info_name'] = np.array(info_name, dtype=object)
    result['info_idx'] = np.array(info_idx, dtype=np.int32)
    result['segments'] = np.array(segments, dtype=np.int32)
    result['db_segments'] = np.array(db_segments, dtype=np.int32)
    result['length'] = total_len

    return result


def split_dataset(segments, mode):
    TRAIN_RATIO = 0.8
    EVAL_RATIO = 0.1
    TEST_RATIO = 0.1
    result = []

    if not segments.size:
        return result

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


class CarpetPressureDataset(Dataset):
    def __init__(self, cfgs, mode='train'):
        self.cfgs = cfgs
        self.dataset_name = cfgs['dataset_path']
        self.seqlen = cfgs['seqlen']  # 16
        self.overlap = cfgs['overlap']  # 0.95
        self.mid_frame = cfgs['mid_frame']  # 7
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
        self._all_pose = []
        self._all_trans = []
        self._all_keypoints_3d = []
        self._current_global_offset = 0

        idx_list = self._get_idx_list()

        load_start = time.time()
        pool = mp.Pool(processes=min(8, mp.cpu_count()))
        load_func = partial(load_db_worker,
                            dataset_path=self.dataset_name,
                            mode=self.mode)
        try:
            results = pool.map(load_func, idx_list)
            pool.close()
            pool.join()
        except Exception as e:
            pool.terminate()
            raise RuntimeError(f"数据加载失败：{e}")
        load_end = time.time()
        print(f'Load dataset finished. Time: {load_end - load_start:.4f}s')

        concat_start = time.time()
        self._restore_original_structure(results)

        self.data = {
            'images': np.concatenate(self._all_images, axis=0) if self._all_images else np.array([]),
            'box': np.concatenate(self._all_boxes, axis=0) if self._all_boxes else np.array([]),
            'betas': np.concatenate(self._all_betas, axis=0) if self._all_betas else np.array([]),
            'pose': np.concatenate(self._all_pose, axis=0) if self._all_pose else np.array([]),
            'trans': np.concatenate(self._all_trans, axis=0) if self._all_trans else np.array([]),
            'keypoints_3d': np.concatenate(self._all_keypoints_3d, axis=0) if self._all_keypoints_3d else np.array([])
        }
        self.data_len = self.data['images'].shape[0] if self.data['images'].size else 0
        concat_end = time.time()
        print(f'Concatenation finished. Total time: {concat_end - concat_start:.4f} seconds')
        print(f'Total data length: {self.data_len}')

        self.video_index_list, self.sample_lens = self.video_seg_window()

        if self.normalize:
            normalize_start_time = time.time()
            min_val = np.min(self.data['images'])
            max_val = 1.5e7
            self.data['images'][self.data['images'] > max_val] = max_val
            self.data['images'] = self.data['images'] / (max_val - min_val)
            normalize_end_time = time.time()
            print(f"Normalization finished. Time: {normalize_end_time - normalize_start_time:.4f} seconds")

        self._print_memory_usage()

    def _get_idx_list(self):
        idx_list = []
        if self.model_type == 'unseen_group':
            idx_list = list(range(1, 53))
            for idx in idx_list:
                print(f'load {self.mode} dataset: {idx}')
        elif self.model_type == 'unseen_subject':
            if self.mode == 'train':
                name_list = subject_fold[self.curr_fold][0]
                train_group_list = set(range(1, 53))
                for name in name_list:
                    train_group_list = train_group_list - set(name_group_map[name])
                idx_list = list(train_group_list)
                for idx in idx_list:
                    print(f'load train dataset: {idx}')
            elif self.mode == 'eval':
                val_group_list = subject_fold[self.curr_fold][1]
                idx_list = val_group_list
                for idx in idx_list:
                    print(f'load eval dataset: {idx}')
        return idx_list

    def _restore_original_structure(self, results):
        for res in results:
            if res['length'] == 0:
                continue

            self._all_images.append(res['images'])
            self._all_boxes.append(res['box'])
            self._all_betas.append(res['betas'])
            self._all_pose.append(res['pose'])
            self._all_trans.append(res['trans'])
            self._all_keypoints_3d.append(res['keypoints_3d'])

            self.info['name'].extend(res['info_name'].tolist())
            self.info['idx'].extend(res['info_idx'].tolist())

            for seg in res['segments']:
                global_seg = [seg[0] + self._current_global_offset, seg[1] + self._current_global_offset]
                self.segments.append(np.array(global_seg))
            self.db_segments.extend(res['db_segments'].tolist())

            self._current_global_offset += res['length']


    def __len__(self):
        return len(self.video_index_list)

    def __getitem__(self, index):
        start_idx, end_idx = self.video_index_list[index]

        curr_frame_idx = [i for i in range(start_idx, end_idx)]
        item = {
            'curr_frame_idx': torch.tensor(curr_frame_idx),
            'images': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, self.data['images'])
            ).float(),
            'boxes': torch.from_numpy(
                self.get_sequence(start_idx, end_idx, self.data['box'])
            ),
            'betas': torch.from_numpy(
                self.get_mid_frame(start_idx, end_idx, self.data['betas'])
            ).float(),
            'pose': torch.from_numpy(
                self.get_mid_frame(start_idx, end_idx, self.data['pose'])
            ).float(),
            'trans': torch.from_numpy(
                self.get_mid_frame(start_idx, end_idx, self.data['trans'])
            ).float(),
            'gt_keypoints_3d': torch.from_numpy(
                self.get_mid_frame(start_idx, end_idx, self.data['keypoints_3d'])
            ).float(),
        }
        return item

    def get_sequence(self, start_index, end_index, data):
        return data[start_index: end_index]

    def get_mid_frame(self, start_index, end_index, data):
        return data[start_index: end_index][self.mid_frame]

    def video_seg_window(self):
        index_list = []
        for segment in self.segments:
            if segment[1] - segment[0] >= self.seqlen:
                for i in range(segment[0], segment[1], self.stride):
                    if i + self.seqlen >= segment[1]:
                        index_list.append([segment[1] - self.seqlen, segment[1]])
                        break
                    else:
                        index_list.append([i, i + self.seqlen])
        return index_list, len(index_list)

    def save_opt_results(self, curr_frame_idx, pose, shape, trans):
        start_index, end_index, len_judge = \
            curr_frame_idx[0], curr_frame_idx[1], curr_frame_idx[2]
        self.data['pose'][start_index: end_index] = pose[: end_index - start_index]
        self.data['betas'][start_index: end_index] = shape[: end_index - start_index]
        self.data['trans'][start_index: end_index] = trans[: end_index - start_index]

    def get_segments(self):
        return self.segments

    def get_data_len(self):
        return self.data_len

    def _print_memory_usage(self):
        print("\n" + "=" * 60)
        print("                    预加载数据内存占用统计")
        print("=" * 60)

        total_memory_mb = 0.0
        for key in self.data.keys():
            data = self.data[key]
            if data.size == 0:
                print(f"{key:<15}: {'0.00':>6} MB (0.000 GB) | 形状: empty")
                continue
            data_shape = data.shape
            data_dtype = data.dtype
            dtype_bytes = data_dtype.itemsize
            total_bytes = data.size * dtype_bytes
            memory_mb = total_bytes / (1024 ** 2)
            memory_gb = memory_mb / 1024
            total_memory_mb += memory_mb
            print(f"{key:<15}: {memory_mb:>6.2f} MB ({memory_gb:>4.3f} GB) | 形状: {data_shape}")

        total_memory_gb = total_memory_mb / 1024
        print("-" * 60)
        print(f"{'预加载总计':<15}: {total_memory_mb:>6.2f} MB ({total_memory_gb:>4.3f} GB)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    cfgs = {
        'dataset_path': '/workspace/Public_Dataset/MIP',
        'seqlen': 16,
        'mid_frame': 7,
        'overlap': 0.95,
        'dataset_mode': 'unseen_group',
        'curr_fold': 1,
        'normalize': True,
        'img_size': (238, 120)
    }
    val_set = CarpetPressureDataset(cfgs, mode='train')
    sample = val_set[1]
    print(val_set.get_data_len())
    print(len(val_set))
    print(sample['curr_frame_idx'])
    import pdb;

    pdb.set_trace()