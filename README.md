# Diff9D: Diffusion-Based Domain-Generalized Category-Level 9DoF Object Pose Estimation

This is the PyTorch implementation of paper **[Diff9D]()** by <a href="https://cnjliu.github.io/">J. Liu</a>, <a href="http://robotics.hnu.edu.cn/info/1071/1265.htm">W. Sun</a>, <a href="https://github.com/CNJianLiu/Diff9D">H. Yang</a>, <a href="https://github.com/CNJianLiu/Diff9D">P. Deng</a>, <a href="https://github.com/CNJianLiu/Diff9D">C. Liu</a>, <a href="https://scholar.google.com.hk/citations?user=stFCYOAAAAAJ&hl=zh-CN&oi=ao">N. Sebe</a>, <a href="https://sites.google.com/view/rahmaniatlu">H. Rahmani</a>, and <a href="https://ajmalsaeed.net/">A. Mian</a>. Diff9D is a simple yet effective prior-free domain-generalized (sim2real) category-level 9DoF object pose generator based on diffusion.

<p align="center">
<img src="image/Fig0.jpg" alt="intro" width="100%"/>
</p>

## Installation
Our code has been trained and tested with:
- Ubuntu 20.04
- Python 3.8.15
- PyTorch 1.12.0
- CUDA 11.3

Complete installation can refer to our [environment](https://github.com/CNJianLiu/Diff9D/blob/main/environment.yaml).

## Datasets
Download NOCS dataset ([CAMERA_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [Real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[gt annotations](http://download.cs.stanford.edu/orion/nocs/gts.zip),
[mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip), and [segmentation results](https://drive.google.com/file/d/1hNmNRr7YRCgg-c_qdvaIzKEd2g4Kac3w/view?usp=sharing)) and Wild6D ([testset](https://ucsdcloud-my.sharepoint.com/:u:/r/personal/yafu_ucsd_edu/Documents/Wild6D/test_set.zip)). Data processing can refer to [IST-Net](https://github.com/CVMI-Lab/IST-Net).
Unzip and organize these files in ../data as follows:
```
data
├── CAMERA
├── camera_full_depths
├── Real
├── gts
├── obj_models
├── segmentation_results
├── Wild6D
```

## Evaluation
You can download our pretrained model [epoch_1000.pth](https://drive.google.com/file/d/1TEOo7MpUAeumi1KQbGwcLg47Hd8ytoAn/view?usp=sharing) (trained solely on the synthetic CAMERA25 dataset) and put it in the '../log1/diffusion_pose' directory. Then, you can quickly evaluate the real-world REAL275 dataset using the following command:
```
python test.py --config config/diffusion_pose.yaml
```
The real-world Wild6D dataset can be evaluated using the following command:
```
bash test_wild6d.sh
```
Note that there is a small mistake in the original evaluation code of [NOCS](https://github.com/hughw19/NOCS_CVPR2019/blob/dd58dbf68feede04c3d7bbafeb9212af1a43422f/utils.py#L252) for the 3D IOU metrics. We thank [CATRE](https://github.com/THU-DA-6D-Pose-Group/CATRE) and [SSC-6D](https://github.com/swords123/SSC-6D) for pointing out this. We have revised it and recalculated the metrics of some methods. The revised evaluation code is given in our released [code](https://github.com/CNJianLiu/Diff9D/blob/1ca38c2bd3f5e896470ad76dcb3ba8e64a2aeff2/utils/evaluation_utils.py#L128).

## Training
To train the model, remember to download the synthetic CAMERA25 dataset and organize & preprocess it properly.

train.py is the main file for training. You can start training using the following command:
```
python train.py --gpus 0 --config config/diffusion_pose.yaml
```
The complete training [log](https://github.com/CNJianLiu/Diff9D/blob/main/log1/diffusion_pose/training_logger.log) has been provided.

## Citation
If you find our work useful, please consider citing:
```latex
@article{Diff9D,
  author={Liu, Jian and Sun, Wei and Yang, Hui and Deng, Pengchao and Liu, Chongpei and Sebe, Nicu and Rahmani, Hossein and Mian, Ajmal},
  title={Diff9D: Diffusion-Based Domain-Generalized Category-Level 9-DoF Object Pose Estimation},
  journal={arXiv preprint arXiv:2502.02525},
  year={2025}
}
```

## Acknowledgment
Our implementation leverages the code from [DPDN](https://github.com/JiehongLin/Self-DPDN) and [IST-Net](https://github.com/CVMI-Lab/IST-Net). We thank the authors for releasing the code.

## Licence

This project is licensed under the terms of the MIT license.
