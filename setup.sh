# Anaconda environment
conda create --name PointCompletionSSM python==3.9.13
conda activate point
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install h5py
conda install munch
conda install pyyaml
conda install pyvista
python -m pip install open3d
conda install timm
conda install einops

# Install utils
export PATH=/usr/local/cuda-11.7/bin:$PATH
export CPATH=/usr/local/cuda-11.7/include:$CPATH
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../emd/
python setup.py install
cd ../Pointnet2.PyTorch/pointnet2/
python setup.py install
cd ../../gridding/
python setup.py install
cd ../cubic_feature_sampling/
python setup.py install 
