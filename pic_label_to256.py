import os
from os.path import join
from PIL import Image
from cut256_with_fixed_box import find_center_xy, find_maxd

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
videos_test_3 = ['410', '411', '516', '517', '526', '528', '529', '530', '531', '533', 
                        '557', '558', '559', '562']
# videos_test_3 = ['533', 
#                         '557', '558', '559', '562']

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


def chage_annot_with_crop(anont_path, res_path, x_left, y_low, x_right, y_high):
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

def resize256(annot_path, x_left, y_low, x_right, y_high):
    """
    args:
        annot_path: 其对应annot地址, .../000001.pts
        max_edge: 图片的边长
    """

    # 修改注解坐标
    # ?/x = 256/max_edge, ?=256*x/max_edge
    scale_x = 256.0/(x_right-x_left)
    scale_y = 256.0/(y_high-y_low)

    # print('scale_x: ', scale_x)
    # print('scale_y: ', scale_y)

    with open(annot_path, 'r') as f:
        lines = f.readlines()
    # 处理第一部分，不改变（版本号和点的数量）
    header = lines[:3]
    points = lines[3:71]
    end = lines[71]

    # 在原文件中改动
    with open(annot_path, 'w') as f:
        f.writelines(header)

        for point in points:
            x, y = point.strip().split()
            # 相对坐标就是crop后的绝对坐标
            x_new = str(float(x)*scale_x)
            y_new = str(float(y)*scale_y)

            # print('x: ', x)
            # print('y: ', y)
            # print('x_new: ', x_new)
            # print('y_new: ', y_new)

            # 写入新的点坐标
            f.write(f'{x_new} {y_new}\n')

        f.write(end)

    return 


def testall_justpic():
    # videos = ['001', '002', '003', '004', '007']
    videos = videos_test_2

    # # cilent
    # pic_300vw_dir = '/home/lxy/桌面/dest_blur'
    # annot_300vw_dir = '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14'

    # server
    pic_300vw_dir = '/home/xyli/data/300vw'
    annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'

    res_annot_300vw_dir = '/home/xyli/data/300vw_resize256_valid_annot'

    # dest/[001,002,...]/crop_pic
    # dest/[001,002,...]/crop_annot
    # dest/[001,002,...]/resize_pic
    # dest/[001,002,...]/resize_annot
    # data300vw_dir_res = '/home/lxy/桌面/dest/' 
    # data300vw_dir_res = '/home/lxy/桌面/dest_blur'

    for video in videos: # 遍历 [001,002,...]

        annots_path = join(annot_300vw_dir, video, 'annot')
        annots = os.listdir(annots_path) 
        for annot in annots: # 遍历 001中的[00000001.png, ...]

            annot_path = join(annots_path, annot)
            annot_path_res = join(res_annot_300vw_dir, video, 'annot')

            if not os.path.exists(annot_path_res):
                os.makedirs(annot_path_res)

            # 从注解中提取信息
            x_left, y_low, x_right, y_high = findxy(annot_path)

            chage_annot_with_crop(
                annot_path,
                annot_path_res,
                x_left, y_low, x_right, y_high
            )

            resize256( 
                annot_path_res,
                x_left, y_low, x_right, y_high
            )  

        print(f'{video}转化结束！')


def testall_justannot():

     # videos = ['001', '002', '003', '004', '007']
    videos = videos_test_2

    # # cilent
    # pic_300vw_dir = '/home/lxy/桌面/dest_blur'
    # annot_300vw_dir = '/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14'

    # server
    annot_300vw_dir = '/home/xyli/data/300VW_Dataset_2015_12_14'
    res_annot_300vw_dir = '/home/xyli/data/300vw_fix256_valid_annot'

    for video in videos: # 遍历 [001,002,...]

        annots_path = join(annot_300vw_dir, video, 'annot')
        annots = os.listdir(annots_path) 

        # 获得中心点坐标
        cx, cy = find_center_xy(annots_path)
        # 获得 d 
        d = find_maxd(annots_path, cx, cy)

        for annot in annots: # 遍历 001中的[00000001.png, ...]

            annot_path = join(annots_path, annot)
            annot_path_res = join(res_annot_300vw_dir, video, 'annot')

            if not os.path.exists(annot_path_res):
                os.makedirs(annot_path_res)

            chage_annot_with_crop(
                annot_path,
                annot_path_res,
                cx-d, cy-d, cx+d, cy+d
            )

            resize256( 
                annot_path,
                cx-d, cy-d, cx+d, cy+d
            )  

        print(f'{video}转化结束！')

if __name__ == '__main__':
    # testall_justpic()

    testall_justannot()
