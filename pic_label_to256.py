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

def find_edge(videos_annots_path):
    """
    用所有帧中最大的人脸框的边做所有帧的框边长,这样就保证所有帧在时间维度上对齐了
    args:
        一个视频对应所有帧的注解目录
    return:
        max_edge: 这些注解中,求得最大边长并返回
    """
    annots = os.listdir(videos_annots_path)
    annots.sort() # 服务器上这个列表默认是乱的，无语

    max_edge = 0
    for annot in annots: # 因为1个video的注解文件有很多，所以要遍历
        
        annot_path = join(videos_annots_path, annot) # .../annot/001564.pts
        keypoints = _keypoint_from_pts_(annot_path)

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

        edge = max(w, h)
        max_edge = max(max_edge, edge)

    return max_edge

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

    return x_left, y_low


def chage_annot_with_crop(anont_path, res_path, max_edge, x_left, y_low):
    """
    args: 
        anont_path: 要变化注解文件的路径, .../000001.pts
        res_path: 储存结果的目录
        max_edge: crop时的边长
        midx, midy: crop时的左上坐标
    """

    with open(anont_path, 'r') as f:
        lines = f.readlines()
    # 处理第一部分，不改变（版本号和点的数量）
    header = lines[:3]
    points = lines[3:71]
    end = lines[71]

    annot_name = anont_path[-10:]
    res_path_file = join(res_path, annot_name)
    with open(res_path_file, 'w') as f:
        f.writelines(header)

        for point in points:
            x, y = point.strip().split()
            # 相对坐标就是crop后的绝对坐标
            x_new = str(float(x) - x_left)
            y_new = str(float(y) - y_low)

            # 写入新的点坐标
            f.write(f'{x_new} {y_new}\n')

        f.write(end)

    return 

def crop_image(apic_path, res_path, max_edge, x_left, y_low):
    """
    args:
        apic_path: 一个要被裁剪的图片路径, 要求图片命名形式为{:08d}.png
        res_path: 裁剪后结果图所放置的路径
        max_edge: 图片将要被裁剪边的长度
    """
    image = Image.open(apic_path)
    cropped_image = image.crop(
                        (
                            x_left , 
                            y_low  , 
                            x_left + max_edge  ,
                            y_low + max_edge  ,
                        )   
                    )
    # 创建注解文件的目录（没有该目录，无法创建注解文件）
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    file_name = apic_path[-12:]
    save_path = join(res_path, file_name)
    cropped_image.save(save_path)

    return 

def resize256(apic_path, annot_path, pic_res_dir, annot_res_dir, max_edge):
    """
    args:
        apic_path: 要resize的某个帧地址, 要求图片命名形式为{:08d}.png
        annot_path: 其对应annot地址, .../000001.pts
        pic_res_dir: 输出的图片存放目录
        annot_res_dir: 修改后注解存放目录
        max_edge: 图片的边长
    """

    # 修改注解坐标
    # ?/x = 256/max_edge, ?=256*x/max_edge
    scale = 256.0/max_edge 

    with open(annot_path, 'r') as f:
        lines = f.readlines()
    # 处理第一部分，不改变（版本号和点的数量）
    header = lines[:3]
    points = lines[3:71]
    end = lines[71]

    annot_name = annot_path[-10:]
    res_path_file = join(annot_res_dir, annot_name)
    with open(res_path_file, 'w') as f:
        f.writelines(header)

        for point in points:
            x, y = point.strip().split()
            # 相对坐标就是crop后的绝对坐标
            x_new = str(float(x)*scale)
            y_new = str(float(y)*scale)

            # 写入新的点坐标
            f.write(f'{x_new} {y_new}\n')

        f.write(end)


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

    chage_annot_with_crop(
        '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14/001/annot/000001.pts',
        '/home/lxy/桌面/annot',
        max_edge+40, x_left-20, y_low-20
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
    pic_300vw_dir = '/home/lxy/桌面/dest_blur'
    annot_300vw_dir = '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14'

    # server
    # pic_300vw_dir = '/home/xyli/data/dest'
    # annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'

    # dest/[001,002,...]/crop_pic
    # dest/[001,002,...]/crop_annot
    # dest/[001,002,...]/resize_pic
    # dest/[001,002,...]/resize_annot
    # data300vw_dir_res = '/home/lxy/桌面/dest/' 
    data300vw_dir_res = pic_300vw_dir 
    # data300vw_dir_res = '/home/lxy/桌面/dest_blur'

    for video in videos: # 遍历 [001,002,...]
        # 待转化数据路径
        # pngs_dir = join(pic_300vw_dir, video, 'images')
        pngs_dir = join(pic_300vw_dir, video, 'images')
        annots_dir = join(annot_300vw_dir, video, 'annot')
        # 转化结果路径
        crop_pic = join(data300vw_dir_res, video, 'crop_pic')
        crop_annot = join(data300vw_dir_res, video, 'crop_annot')
        resize_pic = join(data300vw_dir_res, video, 'resize_pic')
        resize_annot = join(data300vw_dir_res, video, 'resize_annot')

        # 如果转化结果路径不存在, 则创建
        if not os.path.exists(crop_pic):
            os.makedirs(crop_pic)
        if not os.path.exists(crop_annot):
            os.makedirs(crop_annot) 
        if not os.path.exists(resize_pic):
            os.makedirs(resize_pic)
        if not os.path.exists(resize_annot):
            os.makedirs(resize_annot)

        max_edge = find_edge(annots_dir) # 获取001中的最大边长

        pngs = os.listdir(pngs_dir) 
        for png in pngs: # 遍历 001中的[00000001.png, ...]
            # 某个帧 某个帧注解 路径
            png_path = join(pngs_dir, png)
            annot_path = join(annots_dir, png[-10:-4]+'.pts')

            # 从注解中提取信息
            x_left, y_low = findxy(annot_path)
 
            crop_image( 
                png_path, 
                crop_pic, 
                max_edge+40, x_left-20, y_low-20
                # max_edge, x_left, y_low
            )

            chage_annot_with_crop(
                annot_path,
                crop_annot,
                max_edge+40, x_left-20, y_low-20
                # max_edge, x_left, y_low
            )
            
            # 对crop_image进行二次加工
            png_path = join(crop_pic, png)
            annot_path = join(crop_annot, png[-10:-4]+'.pts')

            resize256( 
                png_path,
                annot_path,
                resize_pic,
                resize_annot,
                max_edge+40,
                # max_edge
            )  

        print(f'{video}转化结束！')

if __name__ == '__main__':
    testall()
