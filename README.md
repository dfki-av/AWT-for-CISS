# AWT-for-CISS (WACV 2023) [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](http://arxiv.org/abs/2210.07207)
Official repository for our paper on "Attribution-aware Weight Transfer: A Warm-Start Initialization for Class-Incremental Semantic Segmentation".

<img src="https://github.com/dfki-av/AWT-for-CISS/blob/main/figs/AWT.png" width="100%" height="100%">

# Abstract
In class-incremental semantic segmentation (CISS), deep learning architectures suffer from the critical problems of catastrophic forgetting and semantic background shift. Although recent works focused on these issues, existing classifier initialization methods do not address the background shift problem and assign the same initialization weights to both background and new foreground class classifiers. We propose to address the background shift with a novel classifier initialization method which employs gradient-based attribution to identify the most relevant weights for new classes from the classifier's weights for the previous background and transfers these weights to the new classifier. This warm-start weight initialization provides a general solution applicable to several CISS methods. Furthermore, it accelerates learning of new classes while mitigating forgetting. Our experiments demonstrate significant improvement in mIoU compared to the state-of-the-art CISS methods on the Pascal-VOC 2012, ADE20K and Cityscapes datasets.


Code will be published soon!
