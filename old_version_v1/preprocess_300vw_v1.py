import argparse
import cv2
from os.path import join
import os
import numpy as np
import json
from PIL import Image
from meta300vw import dataset_info # 文件名如果是300vw.py则无法导入，因为不支持数字开头的变量

class Preprocess300vw:
    def __init__(self):
        # 要转换的300vw数据集主目录
        self.original_dir = 'E:\\mmpose\\data\\300VW_Dataset_2015_12_14\\'
        # 转换后的主目录
        self.processed_dir = 'E:\\mmpose\\data\\300vw\\'

        # 300vw一共有这么多视频，每个视频都用一个文件夹装着
        self.videos_all =  ['001', '002', '003', '004', '007', '009', '010', '011', '013', '015', 
                            '016', '017', '018', '019', '020', '022', '025', '027', '028', '029', 
                            '031', '033', '034', '035', '037', '039', '041', '043', '044', '046', 
                            '047', '048', '049', '053', '057', '059', '112', '113', '114', '115', 
                            '119', '120', '123', '124', '125', '126', '138', '143', '144', '150', 
                            '158', '160', '203', '204', '205', '208', '211', '212', '213', '214', 
                            '218', '223', '224', '225', '401', '402', '403', '404', '405', '406', 
                            '407', '408', '409', '410', '411', '412', '505', '506', '507', '508', 
                            '509', '510', '511', '514', '515', '516', '517', '518', '519', '520', 
                            '521', '522', '524', '525', '526', '528', '529', '530', '531', '533', 
                            '537', '538', '540', '541', '546', '547', '548', '550', '551', '553', 
                            '557', '558', '559', '562']
        self.videos_test_1 = ['114', '124', '125', '126', '150', '158', '401', '402', '505', '506',
                              '507', '508', '509', '510', '511', '514', '515', '518', '519', '520', 
                              '521', '522', '524', '525', '537', '538', '540', '541', '546', '547', 
                              '548']
        self.videos_test_2 = ['203', '208', '211', '212', '213', '214', '218', '224', '403', '404', 
                              '405', '406', '407', '408', '409', '412', '550', '551', '553']
        self.videos_test_3 = ['410', '411', '516', '517', '526', '528', '529', '530', '531', '533', 
                              '557', '558', '559', '562']
        self.videos_train = [ i for i in self.videos_all if i not in self.videos_test_1 
                                                        and i not in self.videos_test_2 
                                                        and i not in self.videos_test_3]
        self.videos_all = self.videos_all[:2] # 测试时数据搞小点

        # Downsample FPS to `1 / sample_rate`. Default: 5.
        self.sample_rate = 40 # 约等于1fps

    # 对数据集中所有视频转换成多张图片
    # 其中self.sample_rate可控制转换率，其越小，单个视频转换的图片数量越多
    def convert_jpg(self, videos):

        for video in videos:
            video_path = join(self.original_dir, video, 'vid.avi')
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取视频的总帧数
            i = 1
            while True:
                success, img = cap.read() # 读取视频的下一帧
                if not success: # 如果读一个帧失败了，则退出读取该视频帧过程，换到其它视频
                    break
                if i % self.sample_rate == 0: # 用这种方式控制视频转化率
                    # f是格式化字符串，d表示i是整数，06代表占6个格子多余填充0
                    imgname = f'{i:06d}.jpg' # 要高精度的化.png最好

                    dest_path = join(self.processed_dir, 'images', video)
                    dest = join(dest_path, imgname)
                    if not os.path.exists(dest_path): # 需要先有目录，之后才能创建图片类型文件
                        os.makedirs(dest_path)
                    cv2.imwrite(dest, img)

                    if i == frame_count: # 如果读到最后1帧，则退出
                        break
                i += 1
            cap.release()

            print(f'视频 "{video_path}" 已经转换完毕. ')
        
        return 
    
    # 该函数应该在convert_jpg后执行
    def convert_annot(self, dataset, filename):

        json_data = { # 这个格式是模仿300w的注解文件的
                "images":  [],
                "annotations": {    'segmentation': [],
                                    'num_keypoints': 68,
                                    'iscrowd': 0,
                                    'category_id': 1,
                                
                                },
                "categories": {'id': 1, 'name': 'person'} # coco格式数据集中要调用
        }

        id = 0 
        for video_id in dataset: # 遍历不同数据集所包含的各视频所在目录
            annot_path = join(self.original_dir, video_id, 'annot')

            i = 1
            annots = os.listdir(annot_path)
            annot_merge = []
            for annot in annots: # 因为1个video的注解文件有很多，所以要遍历
                if i % self.sample_rate == 0: # 在这里控制转化率
                    annotation = {}

                    # 找到1个帧注解所对应图片的路径
                    pic_name = os.path.splitext(annot)[0] + ".jpg"
                    pic_path = join(self.processed_dir, 'images', video_id, pic_name)
                    annotation['img_path'] = pic_path

                    # 添加图片宽、高
                    image = Image.open(pic_path) # 打开图片
                    pic_width, pic_height = image.size
                    annotation['height'] = pic_height
                    annotation['width'] = pic_width
                    image.close() # 关闭图像


                    # 找到1个帧注解中的关键点坐标
                    annot_file = join(annot_path, annot)
                    keypoints = self._keypoint_from_pts_(annot_file)
                    annotation['keypoints'] = keypoints
                    
                    # 计算左上坐标、宽、高
                    keypoints_x = []
                    keypoints_y = []
                    for j in range(68):
                        if j%2 == 0:
                            keypoints_x.append(keypoints[j])
                        else:
                            keypoints_y.append(keypoints[j])

                    x_left = min(keypoints_x) - 5.0 # 5是增加一点人为误差，不让关键点就在框边上
                    x_right = max(keypoints_x) 
                    y_low = min(keypoints_y) 
                    y_high = max(keypoints_y) + 5.0
                    w = x_right - x_left + 5.0
                    h = y_high - y_low + 5.0
                    annotation['bbox'] = [x_left, y_high, w, h]

                    # 添加image_id与id
                    annotation['img_id'] = id
                    annotation['id'] = id


                    annot_merge.append(annotation)
                id += 1
                i += 1
            json_data['data_list'].append(annot_merge)
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

    # 从一个.pts格式文件中提取68个关键点到列表中，并返回该列表
    def _keypoint_from_pts_(self,file_path):
        # 创建一个列表来存储关键点坐标
        keypoints = []

        with open(file_path, 'r') as file:
            file_content = file.read()

        # 查找花括号内的数据
        start_index = file_content.find('{')  # 找到第一个左花括号的位置
        end_index = file_content.rfind('}')  # 找到最后一个右花括号的位置

        if start_index != -1 and end_index != -1:
            data_inside_braces = file_content[start_index + 1:end_index]  # 提取花括号内的数据

            # 将数据拆分成行
            lines = data_inside_braces.split('\n')
            for line in lines:
                if line.strip():  # 跳过空行
                    x, y = map(float, line.split())  # 假设坐标是空格分隔的
                    keypoints.append(x)
                    keypoints.append(y)
        else:
            print("未找到花括号内的数据")

        return keypoints

if __name__ == '__main__':
    convert300vw = Preprocess300vw()
    convert300vw.convert_jpg(convert300vw.videos_all)
    convert300vw.convert_annot(convert300vw.videos_all,'train.json')




