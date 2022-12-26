# AWT-for-CISS (WACV 2023) [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](http://arxiv.org/abs/2210.07207)
Official repository for our paper on "Attribution-aware Weight Transfer: A Warm-Start Initialization for Class-Incremental Semantic Segmentation".

<img src="https://github.com/dfki-av/AWT-for-CISS/blob/main/figs/AWT.png" width="100%" height="100%">

# Abstract
In class-incremental semantic segmentation (CISS), deep learning architectures suffer from the critical problems of catastrophic forgetting and semantic background shift. Although recent works focused on these issues, existing classifier initialization methods do not address the background shift problem and assign the same initialization weights to both background and new foreground class classifiers. We propose to address the background shift with a novel classifier initialization method which employs gradient-based attribution to identify the most relevant weights for new classes from the classifier's weights for the previous background and transfers these weights to the new classifier. This warm-start weight initialization provides a general solution applicable to several CISS methods. Furthermore, it accelerates learning of new classes while mitigating forgetting. Our experiments demonstrate significant improvement in mIoU compared to the state-of-the-art CISS methods on the Pascal-VOC 2012, ADE20K and Cityscapes datasets.

```
@article{goswami2022attribution,
  title={Attribution-aware Weight Transfer: A Warm-Start Initialization for Class-Incremental Semantic Segmentation},
  author={Goswami, Dipam and Schuster, Ren{\'e} and van de Weijer, Joost and Stricker, Didier},
  journal={arXiv preprint arXiv:2210.07207},
  year={2022}
}
```

Follow the installation requirements from [Douillard et al.'s repository](https://github.com/arthurdouillard/CVPR2021_PLOP).

## Download the pretrained weights for the ResNet backbone
```
mkdir pretrained
wget https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar -O ./pretrained/resnet101_iabn_sync.pth.tar
```
The default is to use a pretraining for the backbone used, that is in the pretrained folder of the project.
We used the pretrained model released by the authors of In-place ABN (as done by PLOP), that can be found here:
[link](https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar).

# Execution
`run.py` can be used for training individual tasks/increments. It comes with a set of command line options. The most important ones are given in the following. 

### Parameters
The default parameters will replicate our experiments.
- data folder: --data_root \<data_root\>
- dataset: --dataset voc, ade or cityscapes
- task: --task \<task\>, where tasks are
    - 15-1, 15-5, 10-1, 5-3, ..
- step (each step is run separately): --step \<N\>, where N is the step number, starting from 0
- learning rate: --lr 0.02 (for step 0) | 0.001 (for step > 0)
- batch size: --batch_size \<24/num_GPUs\>
- epochs: 60 (ADE20k) | 30 (VOC and Cityscapes)
- method: --method \<method name\>, where names are
    - FT, LWF, LWF-MC, ILT, EWC, RW, PI, MIB, PLOP
- initialization: --orig_init, to use the original initialization proposed by MiB
- threshold: --att 25, threshold for selecting most significant channels (we use 25 in our experiments)


## Training
You can use one of the provided scripts in the `scripts` folder that will launch every step of a continual training. 

For example, edit the last line of run.sh to run the corresponding script. Then, execute run.sh
```
bash run.sh
```

This repository is a modified version of [Douillard et al.'s repository](https://github.com/arthurdouillard/CVPR2021_PLOP).
