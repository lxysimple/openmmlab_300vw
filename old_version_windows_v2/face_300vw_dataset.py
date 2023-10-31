from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from typing import Optional
import os.path as osp
import numpy as np
from mmpose.structures.bbox import bbox_cs2xyxy

@DATASETS.register_module(name='Face300VWDataset')
class Face300VWDataset(BaseDataset):

    # # 感觉就是将标签文件中的data_list进行处理后再送给某个对象（但我这里没做任何处理）
    # def parse_data_info(self, raw_data_info):

        
    #     return raw_data_info

    
    METAINFO: dict = dict(from_file='configs/_base_/datasets/300w.py')

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw Face300W annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation
        """

        data_info_list = []
        for sample in raw_data_info:

            # ann = raw_data_info['raw_ann_info']
            # img = raw_data_info['raw_img_info']

            # img_path = sample['img_path']

            # # 300w bbox scales are normalized with factor 200.
            # pixel_std = 200.

            # # center, scale in shape [1, 2] and bbox in [1, 4]
            # center = np.array([ann['center']], dtype=np.float32)
            # scale = np.array([[ann['scale'], ann['scale']]],
            #                 dtype=np.float32) * pixel_std
            # bbox = bbox_cs2xyxy(center, scale)

            # # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            # _keypoints = np.array(
            #     sample['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            # keypoints = _keypoints[..., :2]
            # keypoints_visible = np.minimum(1, _keypoints[..., 2])

            # num_keypoints = ann['num_keypoints']

            center = np.array([
                                [
                                    (sample['bbox'][0]+sample['bbox'][2])/2,
                                    (sample['bbox'][1]+sample['bbox'][3])/2
                                ]
                                
                             ])
            scale = np.array([
                                [
                                    sample['bbox'][2]/sample['width'],
                                    sample['bbox'][3]/sample['height']
                                ]
                            ])


            keypoints = np.array(
                                    
                                        sample['keypoints']
                                    
                                )
  
            keypoints = keypoints.reshape(1,68,2)

            keypoints_visible = np.array(
                                            
                                                [1.0 for i in range(68)]
                                             
                                        )
            keypoints_visible = keypoints_visible.reshape(1,68)
            data_info = {
                # 'img_id': ann['image_id'],
                # 'img_path': img_path,
                # 'bbox': bbox,
                # 'bbox_center': center,
                # 'bbox_scale': scale,
                # 'bbox_score': np.ones(1, dtype=np.float32),
                # 'num_keypoints': num_keypoints,
                # 'keypoints': keypoints,
                # 'keypoints_visible': keypoints_visible,
                # 'iscrowd': ann['iscrowd'],
                # 'id': ann['id'],

                'img_id': sample['img_id'],
                'img_path': sample['img_path'],

                'bbox': np.array([sample['bbox']]),
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox_score': np.ones(1, dtype=np.float32),
                'num_keypoints': 68,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'iscrowd': 0,
                'id': sample['id'],
                'flip_indices': [i for i in range(68)]


            }

            data_info_list.append(data_info)
        return data_info_list