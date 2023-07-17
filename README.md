# PointCompletionSSM
Implementation of [Can point cloud networks learn statistical shape models of anatomies?](https://arxiv.org/abs/2305.05610) to be presented at MICCAI 2023.
Please cite the paper if you use this code.

SSM visualizations and analysis were completed using the [ShapeWorks](http://sciinstitute.github.io/ShapeWorks/) toolkit.

## Data
The `data/` folder contains the generated proof-of-concept ellipsoid data as well as the aligned spleen and pancreas datasets created from the public [Medical Segmentation Decathlon](http://medicaldecathlon.com/) data. The femur and left atrium datasets included in the paper are not publicly available.

## Installation

To install, call `source setup.sh`.

This will create a conda environment called `PointCompletionSSM` and compile the PyTorch 3rd-party modules ([ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), [emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion), [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch))


## Usage
+ To train a model, run `python train.py -c *.yaml`, e.g., `python train.py -c cfgs/ellipsoids/pcn.yaml`. This will save the network and training log to an output folder in the `experiments/` directory with an updated yaml file. 
+ To test a model, run `python test.py -c *.yaml`, e.g., `python train.py -c experiments/ellipsoids/pcn/pcn.yaml`. This outputs a test log and saves the predicted points.
+ The configs for each experiment reported in the paper can be found in `cfgs/`.
+ Smaller trained models are available in `experiments/`.

## Acknowledgement
We include the following PyTorch 3rd-party libraries:  
[1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
[2] [emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
[3] [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)  

We include the following point completion models:  
[1] [PCN](https://github.com/wentaoyuan/pcn)  
[2] [ECG](https://github.com/paul007pl/ECG)  
[3] [VRCNet](https://github.com/paul007pl/VRCNet)  
[4] [SnowFlakeNet](https://github.com/AllenXiangX/SnowflakeNet)  
[5] [PointAttN](https://github.com/ohhhyeahhh/PointAttN/)
