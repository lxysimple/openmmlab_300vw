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
        self.original_dir = '/home/xyli/data/300VW_Dataset_2015_12_14' # 要转换的300vw数据集主目录
        self.processed_dir = '/home/xyli/data/300vw' # 转换后的主目录
        self.edges_dir = '/home/xyli/data/300vw/edges'
        self.txt_path = '/home/xyli/data/annotations/300VW_blur_label_list_256_test.txt'

        # In Windows:
        # self.original_dir = 'E:/mmpose/data/300VW_Dataset_2015_12_14'
        # self.processed_dir = 'E:/mmpose/data/300vw'

    
    # 该函数应该在convert_jpg后执行
    def convert_annot(self, dataset, filename, dataroot):

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
        for video_id in dataset: # 遍历不同数据集所包含的各视频所在目录
            annot_path = join(self.original_dir, video_id, 'annot')

            i = 1
            annots = os.listdir(annot_path)
            annots.sort() # 服务器上这个列表默认是乱的，无语
            for annot in annots: # 因为1个video的注解文件有很多，所以要遍历

                
                # if this frame is broken, skip it.
                # '000001.pts' -> '000001' -> 1
                if video_id in self.broken_frames and int(annot.split('.')[0]) in self.broken_frames[video_id]:
                    i += 1
                    continue

                if i % self.sample_rate == 0: # 在这里控制转化率
                    annotation = {
                        'segmentation': [],
                        'num_keypoints': 68,
                        'iscrowd': 0,
                        'category_id': 1,
                    }
                    image = {}

                    # print(annot,i,)

                    # 找到1个帧注解所对应图片的路径
                    pic_name = os.path.splitext(annot)[0] + ".jpg"
                    pic_path = join(video_id, pic_name)
                    image['file_name'] = pic_path

                    # 添加图片宽、高
                    pic_path = join(dataroot, pic_path)
                    image_pic = Image.open(pic_path) # 打开图片
                    pic_width, pic_height = image_pic.size
                    image['height'] = pic_height
                    image['width'] = pic_width
                    image_pic.close() # 关闭图像


                    # 找到1个帧注解中的关键点坐标
                    annot_file = join(annot_path, annot)
                    keypoints = self._keypoint_from_pts_(annot_file)
                    # 每个关键点坐标为x,y,c，c就是置信度，一般为1
                    keypoints3 = []
                    for kp_i in range(1,68*2+1):
                        keypoints3.append(keypoints[kp_i-1])
                        if kp_i%2==0:
                            keypoints3.append(1)
                    annotation['keypoints'] = keypoints3
                    
                     # 计算左上坐标、宽、高，无需计算bbox，因为Face300WDataset中会用scale+center求出bbox
                    keypoints_x = []
                    keypoints_y = []
                    for j in range(68*2):
                        if j%2 == 0:
                            keypoints_x.append(keypoints[j])
                        else:
                            keypoints_y.append(keypoints[j])
                    x_left = min(keypoints_x)  
                    x_right = max(keypoints_x) 
                    y_low = min(keypoints_y) 
                    y_high = max(keypoints_y) 
                    w = x_right - x_left 
                    h = y_high - y_low 

                    # annotation['bbox'] = [x_left, y_high, x_right, y_low]

                    scale = math.ceil(max(w,h))/200
                    annotation['scale'] = scale

                    # # 以人脸框做上角为原点计算xy
                    # keypoints3 = []
                    # for kp_i in range(1,68*2+1):
                    #     if kp_i%2 == 1:
                    #         keypoints[kp_i-1] -= x_left
                    #     else:
                    #         keypoints[kp_i-1] -= y_low

                    #     keypoints3.append(keypoints[kp_i-1])
                    #     if kp_i%2==0:
                    #         keypoints3.append(1)
                    # annotation['keypoints'] = keypoints3



                    # 计算人脸面积
                    annotation['area'] = w*h
                    
                    # 计算center
                    center = [
                        (x_left + x_right)/2,
                        (y_low + y_high)/2
                    ]
                    annotation['center'] = center

                    # center = [np.mean(keypoints_x), np.mean(keypoints_y)]
                    # annotation['center'] = center
                    # print("x_left, x_right: ", x_left, x_right)
                    # print("center[0]: ", center[0])
                    # print("y_low, y_high: ", y_low, y_high)
                    # print("center[1]: ", center[1])

                    # max_x = max(x_right-center[0], center[0]-x_left)
                    # max_y = max(y_high-center[1], center[1]-y_low)
                    # scale = max(max_x, max_y)*2 + 5
                    # scale = scale / 200.0
                    # annotation['scale'] = scale
                    # print("scale: ", scale)

                    # 添加image_id与id
                    image['id'] = id
                    annotation['image_id'] = id
                    annotation['id'] = id

                    json_data['images'].append(image)
                    json_data['annotations'].append(annotation)

                id += 1
                i += 1
            print(f'文件夹 "{annot_path}" 已经转换完毕. ')

        # 创建注解文件的目录（没有该目录，无法创建注解文件）
        file_dir = join(self.processed_dir, 'annotations')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # 创建注解文件
        filename = join(file_dir, filename)
        with open(filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        return 

    
    def _keypoints_path_from_txt_(self):
        """
        提取一个.txt中，114个视频，所有的帧

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

                keypoints = strings[0:68]
                keypoints = list(map(float, keypoints))
                path = strings[68]

                anno_1pic['keypoints'] = keypoints
                anno_1pic['path'] = path

                outputs.append(anno_1pic)

        return outputs

if __name__ == '__main__':
    convert300vw = Preprocess300vw()


    outputs = convert300vw._keypoints_path_from_txt_()
    print(outputs[0])
    print(outputs[1])




