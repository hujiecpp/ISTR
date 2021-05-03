This is the project page of the paper:

ISTR: End-to-End Instance Segmentation via Transformers.

**Highlights:**
- **GPU Friendly**: Four 1080Ti/2080Ti GPUs can handle the training for R50, R101 backbones with ISTR.
- **High Performance**: On COCO test-dev, ISTR-R50-3x gets 46.8/38.6 box/mask AP, and ISTR-R101-3x gets 48.1/39.9 box/mask AP.

## Updates
- (2021.05.03) The project page for ISTR is avaliable.

## Models
Method | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[ISTR-R50-3x](https://github.com/hujiecpp/ISTR/blob/main/projects/ISTR/configs/ISTR-R50-3x.yaml) | 17.8 FPS | 46.8  | 38.6 | [model](https://drive.google.com/drive/folders/1LEq1I3RlH5Ufz8agNv9iDgxP85k2Fh2X?usp=sharing) \| [log](https://drive.google.com/drive/folders/1LEq1I3RlH5Ufz8agNv9iDgxP85k2Fh2X?usp=sharing)
[ISTR-R101-3x](https://github.com/hujiecpp/ISTR/blob/main/projects/ISTR/configs/ISTR-R101-3x.yaml) | 13.9 FPS | 48.1  | 39.9 | [model](https://drive.google.com/drive/folders/1LEq1I3RlH5Ufz8agNv9iDgxP85k2Fh2X?usp=sharing) \| [log](https://drive.google.com/drive/folders/1LEq1I3RlH5Ufz8agNv9iDgxP85k2Fh2X?usp=sharing)

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
paper is coming soon.
```
