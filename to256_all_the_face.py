"""
为了还原老师的blur-300vw数据集的高清图,
把老师的图中关键点最左边的点x做crop的x_left,同理,可得y_low, x_right, y_high
然后reshape到256
把原图的最左边的点x做crop的x_left,同理,可得y_low, x_right, y_high
然后reshape到256
这样就能保证两者对齐了,从而能够拿这个样本对训练ESTRNN
"""

import os
from os.path import join
from PIL import Image

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



def findxy(avideo_annot_path):
    """
    找到某个帧人脸框的左上角坐标
    args:
        avideo_annot_path: 某个帧注解的路径, .../000001.pts
    """
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

    return x_left, y_low, x_right, y_high


def crop_image(apic_path, res_path, x_left, y_low, x_right, y_high):
    """
    args:
        apic_path: 一个要被裁剪的图片路径, 要求图片命名形式为{:08d}.png
        res_path: 裁剪后结果图所放置的路径
        max_edge: 图片将要被裁剪边的长度
    """
    image = Image.open(apic_path)
    cropped_image = image.crop(
                        (x_left, y_low, x_right, y_high)   
                    )
    # 创建注解文件的目录（没有该目录，无法创建注解文件）
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    file_name = apic_path[-12:]
    save_path = join(res_path, file_name)
    cropped_image.save(save_path)

    return 

def resize256(apic_path, pic_res_dir):
    """
    args:
        apic_path: 要resize的某个帧地址, 要求图片命名形式为{:08d}.png
        pic_res_dir: 输出的图片存放目录
    """

    # 修改图片
    image = Image.open(apic_path)
    image = image.resize((256, 256))

    if not os.path.exists(pic_res_dir):
        os.makedirs(pic_res_dir)

    save_pic = join(pic_res_dir, apic_path[-12:]) 
    image.save(save_pic)

    return 

def test1():
    max_edge = find_edge(
        '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14/001/annot'
    )

    x_left, y_low = findxy(
        '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14/001/annot/000001.pts'
    )

    print('max_edge: ', max_edge)
    
    crop_image(
        '/home/lxy/桌面/00000001.png', 
        '/home/lxy/桌面/pic/', max_edge+40, x_left-20, y_low-20
    )

    resize256(
        '/home/lxy/桌面/pic/00000001.png',
        '/home/lxy/桌面/annot/000001.pts',
        '/home/lxy/桌面/pic256/',
        '/home/lxy/桌面/annot256',
        max_edge+40
    )   

def testall():
    # videos = ['001', '002', '003', '004', '007']
    videos = ['001']

    # cilent
    # pic_300vw_dir = '/home/lxy/桌面/dest'
    pic_300vw_dir = '/home/xyli/data/300vw'
    annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'

    # server
    # pic_300vw_dir = '/home/xyli/data/dest'
    # annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'

    # dest/[001,002,...]/crop_pic
    # dest/[001,002,...]/crop_annot
    # dest/[001,002,...]/resize_pic
    # dest/[001,002,...]/resize_annot
    data300vw_crop_dir_res = '/home/xyli/data/300vw_crop'
    data300vw_resize256_dir_res = '/home/xyli/data/300vw_resize256' 
    # data300vw_dir_res = pic_300vw_dir 
    # data300vw_dir_res = '/home/lxy/桌面/dest_blur'

    for video in videos: # 遍历 [001,002,...]
        # 待转化数据路径
        # pngs_dir = join(pic_300vw_dir, video, 'images')
        pngs_dir = join(pic_300vw_dir, video)
        annots_dir = join(annot_300vw_dir, video, 'annot')
        
        # 转化结果路径
        crop_pic = join(data300vw_crop_dir_res, video)
        # crop_annot = join(data300vw_dir_res, video, 'crop_annot')

        resize_pic = join(data300vw_resize256_dir_res, video)
        # resize_annot = join(data300vw_dir_res, video, 'resize_annot')

        # 如果转化结果路径不存在, 则创建
        if not os.path.exists(crop_pic):
            os.makedirs(crop_pic)
        if not os.path.exists(resize_pic):
            os.makedirs(resize_pic)

        pngs = os.listdir(pngs_dir) 
        for png in pngs: # 遍历 001中的[00000001.png, ...]
            # 某个帧 某个帧注解 路径
            png_path = join(pngs_dir, png)
            annot_path = join(annots_dir, png[-10:-4]+'.pts')

            # 从注解中提取信息
            x_left, y_low, x_right, y_high = findxy(annot_path)
 
            crop_image( 
                png_path, 
                crop_pic, 
                x_left, y_low, x_right, y_high
            )
            
            # 对crop_image进行二次加工
            png_path = join(crop_pic, png)

            resize256( 
                png_path,
                resize_pic,
            )  

        print(f'{video}转化结束！')

if __name__ == '__main__':
    testall()
