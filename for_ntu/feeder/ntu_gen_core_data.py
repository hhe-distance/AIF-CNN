#coding:utf-8
#author:黄宏恩
import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

paris = {
    'xview': (
        (1, 2, 3),(1, 13, 3),(1, 17, 3), (2, 1, 2),(2, 21, 2),(3, 4, 2), (3, 21, 2), (4, 3, 1), (5, 6, 2), (5, 21, 2),
        (6, 5, 2), (6, 7, 2), (7, 6, 2), (7, 8, 2), (8, 7, 2), (8, 23, 2), (9,  10, 2), (9, 21, 2), (10, 9, 2), (10, 11, 2),
        (11, 10, 2), (11, 12, 2), (12, 11, 2), (12, 25, 2), (13, 1, 2),(13, 14, 2), (14, 13, 2), (14, 15, 2), (15, 14, 2), (15, 16, 2),
        (16, 15, 1), (17, 1, 2), (17, 18, 2), (18, 17, 2), (18, 19, 2), (19, 18, 2), (19, 20, 2), (20, 19, 1),
        (21, 2, 4), (21, 3, 4), (21, 5, 4), (21, 9, 4), (22, 23, 1), (23, 8, 2), (23, 22, 2), (24, 25, 1), (25, 12, 2), (25, 14, 2)
    ),
    'xsub':  (
        (1, 2, 3),(1, 13, 3),(1, 17, 3), (2, 1, 2),(2, 21, 2),(3, 4, 2), (3, 21, 2), (4, 3, 1), (5, 6, 2), (5, 21, 2),
        (6, 5, 2), (6, 7, 2), (7, 6, 2), (7, 8, 2), (8, 7, 2), (8, 23, 2), (9,  10, 2), (9, 21, 2), (10, 9, 2), (10, 11, 2),
        (11, 10, 2), (11, 12, 2), (12, 11, 2), (12, 25, 2), (13, 1, 2),(13, 14, 2), (14, 13, 2), (14, 15, 2), (15, 14, 2), (15, 16, 2),
        (16, 15, 1), (17, 1, 2), (17, 18, 2), (18, 17, 2), (18, 19, 2), (19, 18, 2), (19, 20, 2), (20, 19, 1),
        (21, 2, 4), (21, 3, 4), (21, 5, 4), (21, 9, 4), (22, 23, 1), (23, 8, 2), (23, 22, 2), (24, 25, 1), (25, 12, 2), (25, 14, 2)
    ),

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
        (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

sets = {'train', 'val'}
datasets = {'xview', 'xsub'}

def gen_core_data():
    """Generate bone data from joint data for NTU skeleton dataset"""
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load('/home/hhe/hhe_first_file/data_set/NTU-RGB-D-CV/{}/{}_data.npy'.format(dataset, set))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '/home/hhe/hhe_first_file/data_set/NTU-RGB-D-CV/{}/{}_core.npy'.format(dataset, set),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))

            ori_data = np.zeros(data.shape, dtype=data.dtype)

            # Copy the joints data to bone placeholder tensor

            fp_sp[:, :C, :, :, :] = ori_data
            for v1, v2, num_joints in tqdm(paris[dataset]):
                # Reduce class index for NTU datasets
                if dataset != 'kinetics':
                    v1 -= 1
                    v2 -= 1
                # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
                # There also happens to be 25 bones
                fp_sp[:, :, :, v1, :] = fp_sp[:, :, :, v1, :] + data[:, :, :, v2, :] / num_joints


if __name__ == '__main__':
    gen_core_data()