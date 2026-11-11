
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

def create_concat_dataset(datasets, repeats):
    """ 按照每个数据集的重复次数创建连接数据集 """
    concatenated_datasets = []
    concatenated_list_data_dict = []
    for dataset, repeat in zip(datasets, repeats):
        concatenated_datasets.extend([dataset] * repeat)  # 根据重复次数添加数据集
        concatenated_list_data_dict.extend(dataset.get_data_dict() * repeat)

    # 使用 ConcatDataset 来连接数据集
    return ConcatDataset(concatenated_datasets), concatenated_list_data_dict