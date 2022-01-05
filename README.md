# Image Classification with CondenseNeXt
This repository contains code for single imagle classification on NXP BlueBox 2.0 using RTMaps, which is an extension of "[CondenseNeXt: An Ultra-Efficient Deep Neural Network for Embedded Systems](https://arxiv.org/abs/2112.00698)" paper.

### Citation

If you find my work useful, please consider citing my work:

```
@inproceedings{kalgaonkar2021image,
  title={Image Classification with CondenseNeXt for ARM-Based Computing Platforms},
  author={Kalgaonkar, Priyank and El-Sharkawy, Mohamed},
  booktitle={2021 IEEE International IOT, Electronics and Mechatronics Conference (IEMTRONICS)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

DOI: 10.1109/IEMTRONICS52119.2021.9422541

## Introduction
 In this paper, we demonstrate the implementation of our ultra-efficient deep convolutional neural network architecture: CondenseNeXt on NXP BlueBox, an autonomous driving development platform developed for self-driving vehicles. We show that CondenseNeXt is remarkably efficient in terms of FLOPs, designed for ARM-based embedded computing platforms with limited computational resources and can perform image classification without the need of a CUDA enabled GPU. CondenseNeXt utilizes the state-of-the-art depthwise separable convolution and model compression techniques to achieve a remarkable computational efficiency. Extensive analyses are conducted on CIFAR-10, CIFAR-100 and ImageNet datasets to verify the performance of CondenseNeXt Convolutional Neural Network (CNN) architecture. It achieves state-of-the-art image classification performance on three benchmark datasets including CIFAR-10 (4.79% top-1 error), CIFAR-100 (21.98% top-1 error) and ImageNet (7.91% single model, single crop top-5 error). CondenseNeXt achieves final trained model size improvement of 2.9+ MB and up to 59.98% reduction in forward FLOPs compared to CondenseNet and can perform image classification on ARM-Based computing platforms without needing a CUDA enabled GPU support, with outstanding efficiency.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch ver. 1.1.0](http://pytorch.org)
- [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/)
- [RTMaps Studio ver. 4.8.0](https://intempora.com/products/rtmaps/)
- [NXP BlueBox 2.0](https://community.nxp.com/pwmxy87654/attachments/pwmxy87654/connects/258/1/AMF-AUT-T3652.pdf)

### Evaluation

To run single image classification scripts from this repo using RTMaps Studio software, utilize `rtmaps_python_v2.pck` module within RTMaps, provide python script path and filename as well as path to the sample test image you wish to utilize for image classification purposes. You will need to provide trained weights of CondenseNeXt CNN. Instructions and code to train CondenseNeXt on your desired dataset can be found in this repo: https://github.com/priyankkalgaonkar/CondenseNeXt .

## Contact
pkalgaon@purdue.edu

Any discussions or concerns are welcomed!