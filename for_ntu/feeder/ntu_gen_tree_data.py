#coding:utf-8
#author:黄宏恩
import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

paris = {
    'xview': (
        (1, 2), (1, 13), (1, 17), (2, 1), (2, 21), (3, 4), (3, 21), (4, 3), (5, 6), (5, 21),
        (6, 5), (6, 7), (7, 6), (7, 8), (8, 7), (8, 23), (9, 10), (9, 21), (10, 9), (10, 11),
        (11, 10), (11, 12), (12, 11), (12, 25), (13, 1), (13, 14), (14, 13), (14, 15), (15, 14), (15, 16),
        (16, 15), (17, 1), (17, 18), (18, 17), (18, 19), (19, 18), (19, 20), (20, 19),
        (21, 2), (21, 3), (21, 5), (21, 9), (22, 23), (23, 8), (23, 22), (24, 25), (25, 12), (25, 14)
    ),
    'xsub': (
        (1, 2), (1, 13), (1, 17), (2, 1), (2, 21), (3, 4), (3, 21), (4, 3), (5, 6), (5, 21),
        (6, 5), (6, 7), (7, 6), (7, 8), (8, 7), (8, 23), (9, 10), (9, 21), (10, 9), (10, 11),
        (11, 10), (11, 12), (12, 11), (12, 25), (13, 1), (13, 14), (14, 13), (14, 15), (15, 14), (15, 16),
        (16, 15), (17, 1), (17, 18), (18, 17), (18, 19), (19, 18), (19, 20), (20, 19),
        (21, 2), (21, 3), (21, 5), (21, 9), (22, 23), (23, 8), (23, 22), (24, 25), (25, 12), (25, 14)
    ),

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
        (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

sets = {'train', 'val'}
datasets = {'xview', 'xsub'}

def gen_neighbor_data():
    """Generate bone data from joint data for NTU skeleton dataset"""
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load('/home/hhe/hhe_first_file/data_set/NTU-RGB-D-CV/{}/{}_data.npy'.format(dataset, set))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '/home/hhe/hhe_first_file/data_set/NTU-RGB-D-CV/{}/{}_neighbor.npy'.format(dataset, set),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))

            ori_data = np.zeros(data.shape, dtype=data.dtype)

            # Copy the joints data to bone placeholder tensor

            fp_sp[:, :C, :, :, :] = ori_data
            for v1, v2 in tqdm(paris[dataset]):
                # Reduce class index for NTU datasets
                if dataset != 'kinetics':
                    v1 -= 1
                    v2 -= 1
                # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
                # There also happens to be 25 bones
                fp_sp[:, :, :, v1, :] += data[:, :, :, v1, :] - data[:, :, :, v2, :]


if __name__ == '__main__':
    gen_neighbor_data()