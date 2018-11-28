import numpy as np


def shuffle_data(data_list):
    shuffled_index = list(range(len(data_list)))
    np.random.shuffle(shuffled_index)
    shuffled_dataset = [data_list[i] for i in shuffled_index]
    return shuffled_dataset


def split_train_val(data_set, validate_fraction):
    total = len(data_set)
    val_num = int(len(data_set)*validate_fraction)
    train_num = total - val_num
    return data_set[: train_num], data_set[train_num:]