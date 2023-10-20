from mmengine.dataset import BaseDataset

class dataset300v(BaseDataset):

    # 感觉就是将标签文件中的data_list进行处理后再送给某个对象（但我这里没做任何处理）
    def parse_data_info(self, raw_data_info):

        return raw_data_info

    

