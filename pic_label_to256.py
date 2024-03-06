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