import cv2
import numpy as np
from os.path import join

def frames_1video(frames_dir, res_dir):
    """
    args:
        frames_dir: 生成视频的所有帧目录
        res_dir: 生成视频放的目录
    """
    # 求得帧集
    frames_list = os.listdir(frames_dir)

    # 定义视频文件名和分辨率
    output_video = 'vid.avi'
    frame_width = 256
    frame_height = 256

    # 使用 VideoWriter 创建视频编写对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))

    # 循环生成 frames_count 帧图像并写入视频
    for frame in frames_list:
        # 创建随机图像
        image = cv2.imread(join(frames_dir, frame))  # 替换为你的图片路径
        # 将图像写入视频
        out.write(image)

    # 释放资源
    out.release()

if __name__ == '__main__':
