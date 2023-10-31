import torch

# 指定.pth文件路径
checkpoint_path = 'E:\\mmpose\\checkpoint\\hrnetv2_w18_300w_256x256-eea53406_20211019.pth'
checkpoint = torch.load(checkpoint_path) # 加载.pth文件
# 查看状态字典的键（参数名称）
parameter_names = checkpoint.keys()
# # 打印参数名称
# for name in parameter_names:
#     print(name)
"""
meta
state_dict
"""
meta_data = checkpoint['state_dict']
file_path = 'C:\\Users\\xaoyang\\Desktop\\pth_meta3.txt'
with open(file_path, 'w') as file:
    # 将 meta 数据写入文件
    file.write(str(meta_data))
file.close()