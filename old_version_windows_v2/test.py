from dataset300vw import dataset300v # python文件名开头只能是小写字母，否则会导入失败
# from mmengine.dataset import BaseDataset
import cv2


class LoadImage:

    def __call__(self, results):
        results['img'] = cv2.imread(results['img_path'])
        return results

class ParseImage:

    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

pipeline = [
    LoadImage(),
    ParseImage(),
]


dataset = dataset300v(

    ann_file='E:\\mmpose\\data\\300vw\\annotations\\videos_all_annot.json',
    pipeline=pipeline
)

# print(len(dataset)) # 87
# print(dataset.metainfo) 
# print(dataset.get_data_info(0))
# print(dataset[1]) # 这个是图片本身数组+当前样本在迭代器中的索引

# sub_dataset = dataset.get_subset(10) # 取前10张图片
# print(len(sub_dataset))

# dataset.get_subset_(1) # get_subset_ 接口会对原数据集类做修改，即 inplace 的方式
# print(len(dataset))
