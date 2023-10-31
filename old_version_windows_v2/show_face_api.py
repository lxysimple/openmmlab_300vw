
import numpy as np
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer
from PIL import Image


"""
    直接调用以下代码即可
    from .show_face_api import preprocess
    preprocess(results['img'], results['keypoints'], results['bbox'])
"""
def preprocess(image, keypoints, bbox):

    show(image, keypoints, bbox)


def show(image, keypoints, bbox):
    """
        image，是numpy数组
        keypoints，是mumpy数组，[[[1,2],[3,4],...]]，shape=(1,68,2)
        bbox，是mumpy数组，[[30,30, 300, 300]]，shape=(1,4)
    """
    pose_local_visualizer = PoseLocalVisualizer(radius=1, link_color = 'red')


    # PoseDataSample存的是所有目标关键点信息
    gt_pose_data_sample = PoseDataSample() 

    # # 将构造的真实关键点存入PoseDataSample中
    gt_instances = InstanceData() # InstanceData对象中存的是一个目标关键点信息
    gt_instances.keypoints = keypoints
    gt_instances.bboxes = bbox
    
    gt_pose_data_sample.gt_instances = gt_instances 

   
    # 传入图片、标签、预测、配置，开始画图
    pose_local_visualizer.add_datasample('image', image,
                            gt_pose_data_sample,
                            # out_file='out_file.jpg',
                            show=True,
                            draw_bbox = True )