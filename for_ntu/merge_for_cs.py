#coding:utf-8
#author:黄宏恩
import argparse
import pickle
import  os

import numpy as np
from tqdm import tqdm

def softmax(x):
    x_min_map = x.min(axis=1,keepdims=True)
    x_max_map = x.max(axis=1,keepdims=True)
    size = x_max_map  - x_min_map
    out = (x-size/2.)*10./size
    out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
    return out

root_dir = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/hhe/hhe_first_file/data_set/NTU-RGB-D-CS', help='root directory for all the datasets')
parser.add_argument('--datasets', default='/xsub', choices={'/kinetics', '/xsub', '/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha1', default=1, help='weighted summation for feature1')
parser.add_argument('--alpha2', default=0.98, help='weighted summation for feature2')
parser.add_argument('--alpha3', default=1, help='weighted summation for feature3')
arg = parser.parse_args()

val_label = open(arg.data_dir + arg.datasets + '/val_label.pkl', 'rb')
val_label = np.array(pickle.load(val_label))
train_label = open(arg.data_dir + arg.datasets + '/train_label.pkl', 'rb')
train_label = np.array(pickle.load(train_label))

train_data_bone = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/data_bone_train.pkl', 'rb')
train_data_bone = pickle.load(train_data_bone)
val_data_bone = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/data_bone_val.pkl', 'rb')
val_data_bone = softmax(np.array(pickle.load(val_data_bone)))

print('train_data shape is:',train_data_bone.shape)
print('val_data shape is:',val_data_bone.shape)


train_bone_core = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/bone_core_train.pkl', 'rb')
train_bone_core = pickle.load(train_bone_core)
val_bone_core = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/bone_core_val.pkl', 'rb')
val_bone_core = softmax(np.array(pickle.load(val_bone_core)))

train_data_core = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/data_core_train.pkl', 'rb')
train_data_core = pickle.load(train_data_core)
val_data_core = open(root_dir+'/saved_feature/NTU-RGB-D-CS/AIF_CNN' + '/data_core_val.pkl', 'rb')
val_data_core = softmax(np.array(pickle.load(val_data_core)))


right_num = total_num = right_num_5 = 0.

for i in tqdm(range(len(val_label[0]))):
    _, l = val_label[:, i]
    val_data_bone_ele = val_data_bone[i]
    val_bone_core_ele = val_bone_core[i]
    val_data_core_ele = val_data_core[i]
    r =val_data_bone_ele * arg.alpha1   + val_bone_core_ele * arg.alpha2 + val_data_core_ele * arg.alpha3
    # r = val_data_bone_ele * arg.alpha1+val_bone_core_ele * arg.alpha2
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1.0
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('val accuracy of top1 is:',acc)
print('val accuracy of top5 is:',acc5)