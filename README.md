# A PyTorch Reimplementation of AIF-CNN

## Features
#### 1. Dataset
- [x] NTU RGB+D: Cross View (CV), Cross Subject (CS)
- [ ] Kinect-skeleton

#### 2. Tasks
- [x] Action recognition

#### 3. Visualization
- Visdom supported.

## Prerequisites
Our code is based on **Python3.5**. There are a few dependencies to run the code in the following:
- Python (>=3.5)
- PyTorch (0.4.0)
- [torchnet](https://github.com/pytorch/tnt)
- Visdom
- Other version info about some Python packages can be found in `requirements.txt`

## Usage
#### Data preparation
##### NTU RGB+D
To transform raw NTU RGB+D data into numpy array (memmap format ) by this command:
```commandline
python ./feeder/ntu_gendata.py --data_path <path for raw skeleton dataset> --out_folder <path for new dataset>
```
To generate the bone information:
```commandline
python ./feeder/ntu_gen_bone_data.py --data_path <path for raw skeleton dataset> --out_folder <path for new dataset>
```
To generate the core information:
```commandline
python ./feeder/ntu_gen_core_data.py --data_path <path for raw skeleton dataset> --out_folder <path for new dataset>
```
##### Other Datasets
Not supported now.
#### Training
Before you start the training, you have to launch [visdom](https://github.com/facebookresearch/visdom) server.
```commandline
python -m visdom
```


To train the model, you should note that:
 - ```--dataset_dir``` is the **parents path** for **all** the datasets, 
 - ``` --num ``` the number of experiments trials (type: list).
```commandline
python main.py --dataset_dir <parents path for all the datasets> --mode train --model_name AIF_CNN --dataset_name NTU-RGB-D-CV --num 01
```
To run a new trial with different parameters, you need to: 
- Firstly, run the above training command with a new trial number, e.g, ```--num 03```, thus you will got an error.
- Secondly, copy a  parameters file from the ```./experiments/NTU-RGB-D-CV/HCN01/params.json``` to the path of your new trial ```"./experiments/NTU-RGB-D-CV/HCN03/params.json"``` and modify the information(data,bone or core) as you want.
- At last, run the above training command again, it will works.

#### Testing
```commandline
python main.py --dataset_dir <parents path for all the datasets> --mode test --load True --model_name AIF_CNN --dataset_name NTU-RGB-D-CV --num 01
```

#### Load and Training 
You also can load a half trained model, and start training it from a specific checkpoint by the following command:
```commandline
python main.py --dataset_dir <parents path for all the datasets> --mode load_train --load True --model_name AIF_CNN --dataset_name NTU-RGB-D-CV --num 01 --load_model <path for  trained model>
```

## Results

#### Extracting feature
```commandline
python main.py --dataset_dir <parents path for all the datasets> --mode test --load True --model_name AIF_CNN --dataset_name NTU-RGB-D-CV --num 01
```

#### Ensemble results
Combine the generated scores with:
```commandline
python merge_for_cs.py/merge_for_cv.py
```

## Reference
[1] Chao Li, Qiaoyong Zhong, Di Xie, Shiliang Pu. Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation. IJCAI 2018. http://arxiv.org/pdf/1804.06055.pdf

[2] [huguyuehuhu/HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch).

[3] [kenziyuliu/Unofficial-DGNN-PyTorch](https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch).
