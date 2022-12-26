import copy
import glob
import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset   #, group_images, filter_images

# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 0, # unlabelled + background
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 1,   # road
    8: 2,   # sidewalk
    9: 255,
    10: 255,
    11: 3,  # building
    12: 4,  # wall
    13: 5,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 6,  # pole
    18: 255,
    19: 7,  # traffic light
    20: 8,  # traffic sign
    21: 9,  # vegetation
    22: 10,  # terrain
    23: 11, # sky
    24: 12, # person
    25: 13, # rider
    26: 14, # car
    27: 15, # truck
    28: 16, # bus
    29: 255,
    30: 255,
    31: 17, # train
    32: 18, # motorcycle
    33: 19, # bicycle
    -1: 255
}


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0,255]
    if overlap:
        fil = lambda c: any(x in labels for x in c)
    else:
        fil = lambda c: any(x in labels for x in c) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        tgt = np.unique(np.array(dataset[i][1],dtype=np.int64).flatten())
        cls = [id_to_trainid.get(x,255) for x in tgt]
        if fil(cls):
            idxs.append(i)
        if i % 500 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    print('no of images in current task : ', len(idxs))
    return idxs

class CityscapesSegmentation(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        root = os.path.expanduser(root)
        annotation_folder = os.path.join(root, 'gtFine')
        image_folder = os.path.join(root, 'leftImg8bit')

        if train:
            self.images = [  # Add 18 train cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "train",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "train/*/*.png")))
            ]
            print('images ',len(self.images))
        else:
            self.images = [  # Add 3 validation cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "val",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "val/*/*.png")))
            ]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        try:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

class CityscapesSegmentationIncremental(data.Dataset):
    """Labels correspond to classes (not domain) in this case."""
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        buffer_path=None,
        masking=True,
        overlap=True,
        data_masking="current",
        ignore_test_bg=False,
        buffer_size=20,
        **kwargs
    ):
        full_data = CityscapesSegmentation(root, train)

        self.labels = []
        self.labels_old = []
        self.buffer = []

        if labels is not None:   # task labels
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels  # all labels stored in order

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                if train:
                    idxs = idxs[:train_len]
                else:
                    idxs = idxs[train_len:]

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                if data_masking == "current":
                    tmp_labels = self.labels + [255]
                elif data_masking == "current+old":
                    tmp_labels = labels_old + self.labels + [255]
                elif data_masking == "all":
                    raise NotImplementedError(
                        f"data_masking={data_masking} not yet implemented sorry not sorry."
                    )
                elif data_masking == "new":
                    tmp_labels = self.labels
                    masking_value = 255

                target_transform1 = tv.transforms.Lambda(
                            lambda t: t.apply_(lambda x: id_to_trainid.get(x, 255))
                        )

                target_transform2 = tv.transforms.Lambda(
                            lambda t: t.
                                apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                        )
            else:
                assert False
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform1, target_transform2)
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

