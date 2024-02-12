# human-tracking

### Setup with Anaconda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n botsort_env python=3.8.8
conda activate botsort_env
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>

The code was tested:
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 3.** Install BoT-SORT.
```shell
pip3 install -r requirements.txt
python3 setup.py develop
```
**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step 5. Others
```shell
# Cython-bbox
pip3 install cython_bbox

# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

### Download Models
Download and store the traiend models in 'pretrained' folder as follows
```shell
<BoT-SORT_dir>/pretrained
```

**[YOLOX (BBox detection)](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file)**   
**[ByteTrack](https://github.com/ifzhang/ByteTrack?tab=readme-ov-file)**   
**[ReID model (MOT20)](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view)**   

## MMDet
```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet[tracking]
```
