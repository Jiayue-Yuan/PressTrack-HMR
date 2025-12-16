import numpy as np
from ultralytics.trackers.utils import matching
from .byte_tracker import BYTETracker,STrack
from .basetrack import BaseTrack, TrackState


class NoPredictKalmanFilter:
    """伪卡尔曼滤波器（禁用预测步骤，仅保留必要结构）"""

    def __init__(self):
        # 维持原始矩阵结构（但不会实际使用）
        self._motion_mat = np.eye(8, 8)
        self._update_mat = np.eye(4, 8)

    def initiate(self, measurement):
        """初始化轨迹（直接返回检测框作为状态）"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        covariance = np.eye(8) * 1e-4  # 极小协方差表示完全信任检测
        return mean, covariance

    def predict(self, mean, covariance):
        """禁用预测：直接返回输入状态"""
        return mean.copy(), covariance.copy()  # 原样返回

    def multi_predict(self, mean, covariance):
        """批量禁用预测"""
        return mean.copy(), covariance.copy()

    def update(self, mean, covariance, measurement):
        """更新状态：直接用测量值覆盖"""
        new_mean = np.r_[measurement, np.zeros(4)]  # 速度归零
        new_covariance = np.eye(8) * 1e-4  # 保持极小协方差
        return new_mean, new_covariance

    def project(self, mean, covariance):
        """投影状态：返回检测框部分"""
        return mean[:4].copy(), np.eye(4) * 1e-4

    def gating_distance(self, mean, covariance, measurements, **kwargs):
        """计算距离（基于UoE而非运动模型）"""
        return np.linalg.norm(mean[:4] - measurements, axis=1)


def count_covered_small_boxes(big_box, small_boxes):
    """统计被大框覆盖的小框数量"""
    return sum(is_covered(big_box, small_box) for small_box in small_boxes if area(small_box) < area(big_box))

def is_covered(big_box, small_box):
    """判断大框是否覆盖小框（重叠面积≥小框80%）"""
    intersection = compute_intersection(big_box, small_box)
    return intersection >= 0.8 * area(small_box)

def compute_intersection(box1, box2):
    """计算两个框的交集面积"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def is_large_box(box, size_thresh=6e5, side_thresh=1200):
    """判断是否超过尺寸阈值"""
    side_max = max(box[2] - box[0], box[3] - box[1])
    return (area(box) > size_thresh) or (side_max > side_thresh)


class FootprintTracker(BYTETracker):
    def __init__(self, args, frame_rate=15):
        """
        Initialize a FootprintTracker instance for object tracking.
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = NoPredictKalmanFilter()
        self.reset_id()

    def get_dists(self, tracks, detections):
        # 计算 UoE 距离（替代原始IoU）
        # print(self.frame_id)
        uoe_dist = self.uoe_distance(tracks, detections)  # 可选iou/diou/giou
        if self.args.fuse_score:
            uoe_dist = matching.fuse_score(uoe_dist, detections)
        return uoe_dist

    @staticmethod
    def uoe_distance(atracks: list, btracks: list) -> np.ndarray:
        if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in
                      atracks]
            btlbrs = [det.xywha if hasattr(det, 'angle') and det.angle is not None else det.xyxy for det in btracks]


        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)))

        # 转换所有框为xyxy格式
        def to_xyxy(bbox):
            if len(bbox) == 5:  # xywha格式
                from ultralytics.utils.ops import xywhrm2xyxyxyxy
                return xywhrm2xyxyxyxy(bbox[:4], bbox[4])[:4]  # 取前4个值
            return bbox

        a_boxes = np.array([to_xyxy(box) for box in atlbrs])
        b_boxes = np.array([to_xyxy(box) for box in btlbrs])
        # print('a:',a_boxes)
        # print('b:',b_boxes)

        # I、U计算
        lt = np.maximum(a_boxes[:, None, :2], b_boxes[:, :2])
        rb = np.minimum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area_a = (a_boxes[:, 2] - a_boxes[:, 0]) * (a_boxes[:, 3] - a_boxes[:, 1])
        area_b = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
        union = area_a[:, None] + area_b - inter

        # 计算最小闭包区域
        enclose_lt = np.minimum(a_boxes[:, None, :2], b_boxes[:, :2])
        enclose_rb = np.maximum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        enclose_area = (enclose_rb - enclose_lt).prod(axis=2)


        uoe = union / (enclose_area + 1e-7)

        return 1 - uoe  # 转换为距离

    @staticmethod
    def iou_distance(atracks: list, btracks: list) -> np.ndarray:
        """计算IoU距离 (1 - IoU)"""
        if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in
                      atracks]
            btlbrs = [det.xywha if hasattr(det, 'angle') and det.angle is not None else det.xyxy for det in btracks]

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)))

        # 转换所有框为xyxy格式
        def to_xyxy(bbox):
            if len(bbox) == 5:  # xywha格式
                from ultralytics.utils.ops import xywhrm2xyxyxyxy
                return xywhrm2xyxyxyxy(bbox[:4], bbox[4])[:4]  # 取前4个值
            return bbox

        a_boxes = np.array([to_xyxy(box) for box in atlbrs])
        b_boxes = np.array([to_xyxy(box) for box in btlbrs])

        # 计算交并比
        lt = np.maximum(a_boxes[:, None, :2], b_boxes[:, :2])
        rb = np.minimum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area_a = (a_boxes[:, 2] - a_boxes[:, 0]) * (a_boxes[:, 3] - a_boxes[:, 1])
        area_b = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
        union = area_a[:, None] + area_b - inter

        iou = inter / (union + 1e-7)

        return 1 - iou  # 转换为距离

    @staticmethod
    def diou_distance(atracks: list, btracks: list) -> np.ndarray:
        """计算DIoU距离 (1 - DIoU)"""
        if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in
                      atracks]
            btlbrs = [det.xywha if hasattr(det, 'angle') and det.angle is not None else det.xyxy for det in btracks]

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)))

        # 转换所有框为xyxy格式
        def to_xyxy(bbox):
            if len(bbox) == 5:  # xywha格式
                from ultralytics.utils.ops import xywhrm2xyxyxyxy
                return xywhrm2xyxyxyxy(bbox[:4], bbox[4])[:4]  # 取前4个值
            return bbox

        a_boxes = np.array([to_xyxy(box) for box in atlbrs])
        b_boxes = np.array([to_xyxy(box) for box in btlbrs])

        # 计算IoU部分
        lt = np.maximum(a_boxes[:, None, :2], b_boxes[:, :2])
        rb = np.minimum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area_a = (a_boxes[:, 2] - a_boxes[:, 0]) * (a_boxes[:, 3] - a_boxes[:, 1])
        area_b = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
        union = area_a[:, None] + area_b - inter
        iou = inter / (union + 1e-7)

        # 计算中心点距离
        a_centers = np.stack([(a_boxes[:, 0] + a_boxes[:, 2]) / 2,
                              (a_boxes[:, 1] + a_boxes[:, 3]) / 2], axis=1)
        b_centers = np.stack([(b_boxes[:, 0] + b_boxes[:, 2]) / 2,
                              (b_boxes[:, 1] + b_boxes[:, 3]) / 2], axis=1)

        center_dist = np.sqrt(np.sum((a_centers[:, None] - b_centers) ** 2, axis=2))

        # 计算最小闭包区域对角线距离
        enclose_lt = np.minimum(a_boxes[:, None, :2], b_boxes[:, :2])
        enclose_rb = np.maximum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        enclose_diag = np.sqrt(np.sum((enclose_rb - enclose_lt) ** 2, axis=2))

        # 计算DIoU
        diou = iou - (center_dist ** 2) / (enclose_diag ** 2 + 1e-7)

        return 1 - diou  # 转换为距离

    @staticmethod
    def giou_distance(atracks: list, btracks: list) -> np.ndarray:
        """计算GIoU距离 (1 - GIoU)"""
        if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.xywha if hasattr(track, 'angle') and track.angle is not None else track.xyxy for track in
                      atracks]
            btlbrs = [det.xywha if hasattr(det, 'angle') and det.angle is not None else det.xyxy for det in btracks]

        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return np.zeros((len(atlbrs), len(btlbrs)))

        # 转换所有框为xyxy格式
        def to_xyxy(bbox):
            if len(bbox) == 5:  # xywha格式
                from ultralytics.utils.ops import xywhrm2xyxyxyxy
                return xywhrm2xyxyxyxy(bbox[:4], bbox[4])[:4]  # 取前4个值
            return bbox

        a_boxes = np.array([to_xyxy(box) for box in atlbrs])
        b_boxes = np.array([to_xyxy(box) for box in btlbrs])

        # 计算IoU部分
        lt = np.maximum(a_boxes[:, None, :2], b_boxes[:, :2])
        rb = np.minimum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area_a = (a_boxes[:, 2] - a_boxes[:, 0]) * (a_boxes[:, 3] - a_boxes[:, 1])
        area_b = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
        union = area_a[:, None] + area_b - inter
        iou = inter / (union + 1e-7)

        # 计算最小闭包区域
        enclose_lt = np.minimum(a_boxes[:, None, :2], b_boxes[:, :2])
        enclose_rb = np.maximum(a_boxes[:, None, 2:], b_boxes[:, 2:])
        enclose_area = (enclose_rb[:, :, 0] - enclose_lt[:, :, 0]) * (enclose_rb[:, :, 1] - enclose_lt[:, :, 1])

        # 计算GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-7)

        return 1 - giou  # 转换为距离
