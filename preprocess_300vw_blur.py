import argparse
import cv2
from os.path import join
import os
import numpy as np
import json
from PIL import Image
from meta300vw import dataset_info # 文件名如果是300vw.py则无法导入，因为不支持数字开头的变量
import math
from PIL import Image, ImageDraw

# test
class Preprocess300vw:
    def __init__(self):
        # In Linux:
        self.original_dir = '/home/xyli/data/Blurred-300VW'


        self.processed_file = '/home/xyli/data/annotations/300VW_blur_label_list_256_train_mmpose.json' # 转换后的主目录
        self.txt_path = '/home/xyli/data/annotations/300VW_blur_label_list_256_train.txt'

        # self.txt_path = '/home/xyli/data/annotations/300VW_blur_label_list_256_test.txt'
        # self.processed_file = '/home/xyli/data/annotations/300VW_blur_label_list_256_test_mmpose.json' # 转换后的主目录

    
    # 该函数应该在convert_jpg后执行
    def convert_annot(self, dataset):

        json_data = { 
                'images': [ 
                            # {
                            #     'file_name': '000000001268.jpg',
                            #     'height': 427, 
                            #     'width': 640, 
                            #     'id': 1268 
                            # },
                ],
                'annotations': [ # 所有目标的列表
                                    # {
                                    #     'segmentation': [],
                                    #     'num_keypoints': 68,
                                    #     'iscrowd': 0,
                                    #     'category_id': 1,

                                    #     'keypoints': [
                                    #         0,0,0,
                                    #         0,0,0,
                                    #         0,0,0],
                                    #     'area': 3894.5826, 
                                    #     'image_id': 1268, 
                                    #     'bbox': [402.34, 205.02, 65.26, 88.45],
                                    #     'id': 215218 
                                    #     'center':

                                    # },
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person"
                    }
                ]
        }

        id = 0 
        for pic in dataset: 

            keypoints = pic['keypoints']
            path = pic['path'] # 相对路径
            pic_path = join(self.original_dir, path) # 该帧的绝对路径


            annotation = {
                'segmentation': [],
                'num_keypoints': 68,
                'iscrowd': 0,
                'category_id': 1,
            }
            image = {}

            # path_list = path.split('/')
            image['file_name'] = path

            # 添加图片宽、高
            image['height'] = 256
            image['width'] = 256

            # 每个关键点坐标为x,y,c，c就是置信度，一般为1
            keypoints3 = []
            for kp_i in range(1,68*2+1):
                keypoints3.append(keypoints[kp_i-1])
                if kp_i%2==0:
                    keypoints3.append(1)
            annotation['keypoints'] = keypoints3
        

            scale = 256.0/200.0
            annotation['scale'] = scale


            # 计算人脸面积
            annotation['area'] = 256*256
            
            # 计算center
            center = [
                256/2,
                256/2
            ]
            annotation['center'] = center

            # 添加image_id与id
            image['id'] = id
            annotation['image_id'] = id
            annotation['id'] = id

            json_data['images'].append(image)
            json_data['annotations'].append(annotation)

            id += 1

        # 创建注解文件
        with open(self.processed_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        return

    
    def _keypoints_path_from_txt_(self):
        """
        提取一个.txt，114个视频，所有的帧

        return:
            [
                {
                    'keypoints':[x1, y1, x2, y2, ..., x68, y68]
                    'path': '114/0.jpg'
                },
                ...
                {
                    'keypoints':[x1, y1, x2, y2, ..., x68, y68]
                    'path': '114/2994.jpg'
                },
                ...
                {
                    'keypoints':[x1, y1, x2, y2, ..., x68, y68]
                    'path': '562/0.jpg'
                },
                ...
            ]
        """
        outputs = []

        with open(self.txt_path, 'r') as file:
            for line in file:

                # 使用 split() 方法将当前行按空格分割成字符串列表
                strings = line.split()
        
                # 检查是否有 69 个字符串
                if len(strings) != 68*2+1:
                    print("Error: 每一行应该有 69 个字符串")
                    continue
                
                keypoints = []
                path = ''

                keypoints = strings[0:68*2]
                keypoints = list(map(float, keypoints))
                path = strings[68*2]

                anno_1pic = {} 
                anno_1pic['keypoints'] = keypoints
                anno_1pic['path'] = path

                outputs.append(anno_1pic)

        return outputs

if __name__ == '__main__':
    convert300vw = Preprocess300vw()


    outputs = convert300vw._keypoints_path_from_txt_()
    # print(len(outputs))
    # print('len(outputs[0][keypoints]): ', len(outputs[0]['keypoints']))
    # print(outputs[0]['keypoints'])
    # print(outputs[0]['path'])

    convert300vw.convert_annot(outputs) # 大约30s




