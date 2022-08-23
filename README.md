This is the project page for the paper:

>[**ISTR: End-to-End Instance Segmentation via Transformers**](https://arxiv.org/abs/2105.00637).

<!-- :star:**Highlights:**
- **GPU Friendly**: Four 1080Ti/2080Ti GPUs can handle the training for R50, R101 backbones with ISTR.
- **High Performance**: On COCO test-dev, ISTR-R50-3x gets 46.8/38.6 box/mask AP, and ISTR-R101-3x gets 48.1/39.9 box/mask AP. -->

## Updates
- (2022.03.09) New codes for ISTR-PCA, ISTR-DCT, and ISTR-SMT with better performance and speed have been released.
- (2021.05.03) The project page for ISTR is avaliable.

Method   | backbone | fps | box AP | mask AP | link
---      |   :---:  |  :---:|:---:   |:---:    |:---:
ISTR-PCA | R50-FPN  | 13.0  | 46.7   | 39.8    | [7p58](https://pan.baidu.com/s/1WZsA1OBH9NPtig8kaUmpjw?pwd=7p58) || [goole drive](https://drive.google.com/drive/folders/1TAFFSu2MIEmaUIGrH5lCulp12Th0xvSK?usp=sharing)
ISTR-DCT | R50-FPN  | 12.5  | 46.9   | 40.2    | [ibi3](https://pan.baidu.com/s/1gsVq53bP1ZyPDoxIlEkamg?pwd=ibi3)
ISTR-SMT | R50-FPN  | 10.4  | 47.4   | 41.7    | [73bs](https://pan.baidu.com/s/1aGAGvqs5jcly8Ywh_KP6SA?pwd=73bs)
ISTR-PCA | R101-FPN | 10.7  | 48.0   | 41.1    | [5rcj](https://pan.baidu.com/s/11Mi_kGVUIDBz1U6jgW0GnQ?pwd=5rcj)
ISTR-DCT | R101-FPN | 10.3  | 48.3   | 41.6    | [0mdl](https://pan.baidu.com/s/1SCoT6Pc92GdHWIsaW_Elug?pwd=0mdl)
ISTR-SMT | R101-FPN | 8.9   | 48.8   | 42.9    | [qbr8](https://pan.baidu.com/s/1jXowvA5xR_U191p-CAkWOA?pwd=qbr8)
ISTR-SMT | Swin-L   | 3.5   | 55.8   | 49.2    | [nuj8](https://pan.baidu.com/s/1pcAM6jDKspqve3X_I2KRJA?pwd=nuj8)
ISTR-SMT@1088 | Swin-L | 2.9 | 56.4 | 49.7 | [9uj8](https://pan.baidu.com/s/1AjH9VyLc01tKWSzw2ee8tQ?pwd=9uj8)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/istr-end-to-end-instance-segmentation-with/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=istr-end-to-end-instance-segmentation-with)


- The inference time is evaluated with a single 2080Ti GPU.
- We use the models pre-trained on ImageNet using torchvision. The ImageNet pre-trained [ResNet-101](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) backbone is obtained from [SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN/blob/main/tools/convert-torchvision-to-d2.py).

## Installation
The codes are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN), and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).

#### Requirements
- Python=3.8
- PyTorch=1.6.0, torchvision=0.7.0, cudatoolkit=10.1
- OpenCV for visualization

#### Steps
1. Install the repository (we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda create -n ISTR python=3.8 -y
conda activate ISTR
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
or (conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
pip install opencv-python
pip install scipy
pip install shapely
git clone https://github.com/hujiecpp/ISTR.git
cd ISTR
python setup.py build develop
```

2. Link coco dataset path
```
ln -s /coco_dataset_path/coco ./datasets
```

3. Train ISTR (e.g., with ResNet50 backbone)
```
python projects/ISTR/train_net.py --num-gpus 4 --config-file projects/ISTR/configs/ISTR-R50-3x.yaml
```

4. Evaluate ISTR (e.g., with ResNet50 backbone)
```
python projects/ISTR/train_net.py --num-gpus 4 --config-file projects/ISTR/configs/ISTR-R50-3x.yaml --eval-only MODEL.WEIGHTS ./output/model_final.pth
```

5. Visualize the detection and segmentation results (e.g., with ResNet50 backbone)
```
python demo/demo.py --config-file projects/ISTR/configs/ISTR-R50-3x.yaml --input input1.jpg --output ./output --confidence-threshold 0.4 --opts MODEL.WEIGHTS ./output/model_final.pth
```

## Citation

If our paper helps your research, please cite it in your publications:

```BibTeX
@article{hu2021istr,
  title={Istr: End-to-end instance segmentation with transformers},
  author={Hu, Jie and Cao, Liujuan and Lu, Yao and Zhang, ShengChuan and Wang, Yan and Li, Ke and Huang, Feiyue and Shao, Ling and Ji, Rongrong},
  journal={arXiv preprint arXiv:2105.00637},
  year={2021}
}
```
