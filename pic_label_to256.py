import os


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

def find_edge(avideo_annots_path):
    """
    用所有帧中最大的人脸框的边做所有帧的框边长,这样就保证所有帧在时间维度上对齐了
    args:
        一个视频对应所有帧的注解目录
    return:
        这些注解中,求得最大边长并返回
    """
    annots = os.listdir(avideo_annots_path)
    annots.sort() # 服务器上这个列表默认是乱的，无语

    for annot in annots: # 因为1个video的注解文件有很多，所以要遍历
        print(annot)

    return 



if __name__ == '__main__':

    find_edge('/media/lxy/新加卷/mmpose/data/300VW_Dataset_2015_12_14/001/annot')