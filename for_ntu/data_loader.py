# encoding: utf-8


import torch
from feeder.feeder import Feeder
import numpy as np

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev
    """
    if 'NTU-RGB-D' in params.dataset_name :
        if 'CV' in params.dataset_name:
            params.train_feeder_args["data1_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/train_'+params.data_type1+'.npy'
            params.train_feeder_args["data2_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/train_' + params.data_type2 + '.npy'
            params.train_feeder_args["num_frame_path"] = params.dataset_dir+'/NTU-RGB-D-CV'+'/xview/train_num_frame.npy'
            params.train_feeder_args["label_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/train_label.pkl'
            params.test_feeder_args["data1_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/val_'+params.data_type1+'.npy'
            params.test_feeder_args["data2_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/val_' + params.data_type2 + '.npy'
            params.test_feeder_args["num_frame_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/val_num_frame.npy'
            params.test_feeder_args["label_path"] = params.dataset_dir + '/NTU-RGB-D-CV' + '/xview/val_label.pkl'

        if 'CS' in params.dataset_name:
            params.train_feeder_args["data1_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/train_'+params.data_type1+'.npy'
            params.train_feeder_args["data2_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/train_' + params.data_type2 + '.npy'
            params.train_feeder_args["num_frame_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/train_num_frame.npy'
            params.train_feeder_args["label_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/train_label.pkl'
            params.test_feeder_args["data1_path"]= params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/val_'+params.data_type1+'.npy'
            params.test_feeder_args["data2_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/val_' + params.data_type2 + '.npy'
            params.test_feeder_args["num_frame_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/val_num_frame.npy'
            params.test_feeder_args["label_path"] = params.dataset_dir + '/NTU-RGB-D-CS' + '/xsub/val_label.pkl'

    if types == 'train':
        if not hasattr(params,'batch_size_train'):
            params.batch_size_train = params.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.train_feeder_args),
            batch_size=params.batch_size_train,
            shuffle=True,
            drop_last=False,
            num_workers=params.num_workers,pin_memory=params.cuda)

    if types == 'test':
        if not hasattr(params,'batch_size_test'):
            params.batch_size_test = params.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.test_feeder_args),
            batch_size=params.batch_size_test ,
            shuffle=False,
            drop_last=False,
            num_workers=params.num_workers,pin_memory=params.cuda)

    return loader

if __name__ == '__main__':


    pass
