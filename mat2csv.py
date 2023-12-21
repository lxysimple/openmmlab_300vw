import os
import scipy.io
import pandas as pd

# 替换 'input_folder' 为包含 .mat 文件的文件夹路径
input_folder = 'E:/mmpose/data/300VW_Dataset_2015_12_14'

# 替换 'output_folder' 为保存 .csv 文件的文件夹路径
output_folder = 'E:/mmpose/data/300VW_Dataset_2015_12_14/mat2csv'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有 .mat 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        # 构建完整的文件路径
        mat_file_path = os.path.join(input_folder, filename)

        # 使用 loadmat 函数读取 .mat 文件
        mat_data = scipy.io.loadmat(mat_file_path)

        # 获取 .mat 文件中的数据（这里假设只有一个变量）
        variable_name = [key for key in mat_data.keys() if not key.startswith("__")][0]
        data = mat_data[variable_name]

        # 将数据转换为 Pandas DataFrame
        df = pd.DataFrame(data, columns=[f"column_{i+1}" for i in range(data.shape[1])])

        # 替换文件扩展名为 .csv，并构建输出文件路径
        csv_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.csv')

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(csv_file_path, index=False)

print("转换完成！")
