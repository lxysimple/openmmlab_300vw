"""
原始的crop+resize256会使得不同时间步之间不对齐
于是要用一种时间步对齐的算法：

1.先固定一个中心点(cx,cy)（由一个视频所有帧每个图的中心点累加求均值）
2.在所有帧中,求得距离中心点最大点的距离d(这个距离是在y/x轴上的投影,而不是间距)
3.crop(cx-d, cy-d, cx+d, cy+d)
4.resize(256,256)
"""

import os
from os.path import join
from PIL import Image
import numpy as np


# 300vw一共有这么多视频，每个视频都用一个文件夹装着
videos_all =  ['001', '002', '003', '004', '007', '009', '010', '011', '013', '015', 
                    '016', '017', '018', '019', '020', '022', '025', '027', '028', '029', 
                    '031', '033', '034', '035', '037', '039', '041', '043', '044', '046', 
                    '047', '048', '049', '053', '057', '059', '112', '113', '114', '115', 
                    '119', '120', '123', '124', '125', '126', '138', '143', '144', '150', 
                    '158', '160', '203', '204', '205', '208', '211', '212', '213', '214', 
                    '218', '223', '224', '225', 
                                                '401', '402', '403', '404', '405', '406', 
                    '407', '408', '409', '410', '411', '412', '505', '506', '507', '508', 
                    '509', '510', '511', '514', '515', '516', '517', '518', '519', '520', 
                    '521', '522', '524', '525', '526', '528', '529', '530', '531', '533', 
                    '537', '538', '540', '541', '546', '547', '548', '550', '551', '553', 
                    '557', '558', '559', '562']

# Category 1 in laboratory and naturalistic well-lit conditions
videos_test_1 = ['114', '124', '125', '126', '150', '158', '401', '402', '505', '506',
                        '507', '508', '509', '510', '511', '514', '515', '518', '519', '520', 
                        '521', '522', '524', '525', '537', '538', '540', '541', '546', '547', 
                        '548']
# Category 2 in real-world human-computer interaction applications
videos_test_2 = ['203', '208', '211', '212', '213', '214', '218', '224', '403', '404', 
                        '405', '406', '407', '408', '409', '412', '550', '551', '553']

# Category 3 in arbitrary conditions
# videos_test_3 = ['410', '411', '516', '517', '526', '528', '529', '530', '531', '533', 
#                         '557', '558', '559', '562']
videos_test_3 = ['528', '529', '530', '531', '533','557', '558', '559', '562']

videos_train = [ i for i in videos_all if i not in videos_test_1 
                                        and i not in videos_test_2 
                                        and i not in videos_test_3]



def _keypoint_from_pts_(file_path):
    """
    从一个.pts格式文件中提取68个关键点到列表中,并返回该列表
    return:
        [x1,y1, x2,y2, ..., x68,y68]
    """
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



def find_center_xy(avideo_annot_dir):
    """
    找到某个视频人脸的中心点

    args:
        avideo_annot_path: 某个视频所有帧注解的路径
    return:
        所有注解中心点的均值坐标
    """

    pts_list = os.listdir(avideo_annot_dir)

    cx_list = []
    cy_list = []

    for pts in pts_list:
        # 获得单个.pts的 path
        avideo_annot_path = join(avideo_annot_dir, pts)

        keypoints = _keypoint_from_pts_(avideo_annot_path)

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

        cx = (x_left+x_right)/2
        cy = (y_low+y_high)/2

        cx_list.append(cx)
        cy_list.append(cy)

    return np.mean(cx_list), np.mean(cy_list)


def find_maxd(avideo_annot_dir, cx, cy):
    """
    找到某个视频距人脸中心点最大投影距离d(在y/x轴上最大投影d)

    args:
        avideo_annot_path: 某个视频所有帧注解的路径
        cx, cy: 中心点坐标
    return:
        d
    """

    pts_list = os.listdir(avideo_annot_dir)

    d_list = []

    for pts in pts_list:

        # 获得单个.pts的 path
        avideo_annot_path = join(avideo_annot_dir, pts)

        keypoints = _keypoint_from_pts_(avideo_annot_path)

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

        dx1 = np.abs(cx - x_left)
        dx2 = np.abs(cx - x_right)
        dy1 = np.abs(cy - y_low)
        dy2 = np.abs(cy - y_high)


        d_list.append(np.max([dx1, dx2, dy1, dy2]))

    return np.max(d_list)


def crop_resize256_image(pic_path, res_path, x_left, y_low, x_right, y_high):

    image = Image.open(pic_path)
    image = image.crop(
                        (x_left, y_low, x_right, y_high)   
                    )
    image = image.resize((256, 256))

    image.save(res_path)

    return


def test_300vw():
    # videos = ['001']
    # videos = videos_train
    videos = videos_test_3 

    # pic_300vw_dir = '/media/lxy/新加卷/Ubuntu/300vw_myblur'
    # annot_300vw_dir = '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14'
    # data300vw_res_dir = '/media/lxy/新加卷/Ubuntu/300vw_fix256_myblur' 
    
    pic_300vw_dir = '/home/xyli/data/300vw'
    annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'
    data300vw_res_dir = '/home/xyli/data/300vw_fix256' 

    for video in videos: # 遍历 [001,002,...]
        # 将各路径join video
        # pngs_dir = join(pic_300vw_dir, video, 'images')
        pngs_dir = join(pic_300vw_dir, video)
        annots_dir = join(annot_300vw_dir, video, 'annot') # 用于规定如何crop
        data300vw_res_video_dir = join(data300vw_res_dir, video)

        # 如果转化结果路径不存在, 则创建
        if not os.path.exists(data300vw_res_video_dir):
            os.makedirs(data300vw_res_video_dir)

        # 获得中心点坐标
        cx, cy = find_center_xy(annots_dir)

        # 获得 d 
        d = find_maxd(annots_dir, cx, cy)

        # print('cx, cy: ',cx, cy)
        # print('2*d: ', 2*d)
        
        pngs = os.listdir(pngs_dir) 
        pngs.sort()
        # from IPython import embed
        # embed()
        for png in pngs: # 遍历 001中的[00000001.png, ...]
            # 某个帧 某个帧注解 路径
            png_path = join(pngs_dir, png)
            png_res_path = join(data300vw_res_video_dir, f"{int(png[:-4]):08d}.png")
            
            crop_resize256_image(png_path, png_res_path, cx-d, cy-d, cx+d, cy+d) 

        print(f'{video}转化结束！')


if __name__ == '__main__':

    test_300vw() # 指的是注解用.pt装的数据
