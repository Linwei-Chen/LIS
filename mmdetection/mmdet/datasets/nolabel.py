import os
import os.path as osp
from posixpath import join
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from numpy import random
from terminaltables import AsciiTable
import torch
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
from glob import glob
import json
import cv2
from tqdm import tqdm


@DATASETS.register_module()
class NoLabel(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    def __init__(self,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix=None, 
                 search_path='./*/*',
                 fake_size=None):
        self.ann_file = None
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = None
        self.proposal_file = None
        self.test_mode = False
        self.filter_empty_gt = False
        self.CLASSES = None
        self.fake_size = fake_size
        # self.img_path = img_path

        IMG_EXT = ('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG')
        img_list = []
        for ext in IMG_EXT:
            temp_list = glob(osp.join(data_root, f'{search_path}.{ext}'))
            img_list +=temp_list
        
        img_json_path = osp.join(data_root, 'train.json')
        try:
            with open(img_json_path) as f:
                self.data_infos = json.load(f)
        except Exception:
            pass
            self.data_infos = []
            for img in tqdm(img_list):
                id = osp.splitext(osp.split(img)[1])[0]
                try:
                    tmp_img = cv2.imread(img)
                    h, w, _ = tmp_img.shape
                    self.data_infos.append({'id':id, 'filename':img, 'width':w, 'height':h})
                except Exception:
                    continue
            with open(img_json_path, 'w') as f:
                json.dump(self.data_infos, f)

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.ann_file):
        #         self.ann_file = osp.join(self.data_root, self.ann_file)
        #     if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
        #         self.img_prefix = osp.join(self.data_root, self.img_prefix)
        #     if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
        #         self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
        #     if not (self.proposal_file is None
        #             or osp.isabs(self.proposal_file)):
        #         self.proposal_file = osp.join(self.data_root,
        #                                       self.proposal_file)
        # load annotations (and proposals)
        # self.data_infos = self.load_annotations(self.ann_file)

        # if self.proposal_file is not None:
            # self.proposals = self.load_proposals(self.proposal_file)
        # else:
            # self.proposals = None

        # filter images too small and containing no annotations
        # if not test_mode:
        #     valid_inds = self._filter_imgs()
        #     self.data_infos = [self.data_infos[i] for i in valid_inds]
        #     if self.proposals is not None:
        #         self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        if self.fake_size is not None:
            return self.fake_size

        return len(self.data_infos)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        # pool = np.where(self.flag == self.flag[idx])[0]
        # return np.random.choice(pool)
        return np.random.randint(low=0, high=len(self.data_infos))

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.fake_size is not None:
            idx = random.randint(low=0, high=len(self.data_infos))
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = None
        # results = dict(img_info=img_info)
        results = dict(img_info=img_info, ann_info=ann_info, 
                       gt_labels=torch.Tensor([]), 
                       gt_bboxes=torch.Tensor([[-1, -1, -1, -1]]),
                       gt_masks=torch.Tensor([[0, 0, 0, 0]]), 
                       gt_bboxes_ignore=torch.Tensor([1]))
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        # print(results.keys())
        return results
    
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')