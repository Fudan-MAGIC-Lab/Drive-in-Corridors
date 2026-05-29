 # Step-by-step installation instructions

**0. Clone the corridor repo.**
```shell
git clone https://github.com/Fudan-MAGIC-Lab/Drive-in-Corridors.git
```

**1. Create the environment.**
```shell
conda create -n corridor python=3.8 -y
conda activate corridor
```

**2. Install pytorch.**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Install openmmlab dependencies.**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**4. Install other dependencies.**
```shell
pip install -r requirements.txt
```

**5. Install mmdetection3d.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout -f v0.17.1
pip install -r requirements/runtime.txt  
pip install -e .                         
```