import copy
import os
import random

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from apex.parallel import DistributedDataParallel
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

import argparser
import tasks
import utils
from dataset import (AdeSegmentationIncremental,
                     VOCSegmentationIncremental,
                     CityscapesSegmentationIncremental, transform)
from metrics import StreamSegMetrics
from segmentation_module import make_model
from utils.logger import Logger
from run import get_dataset
from captum.attr import LayerIntegratedGradients


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if opts.crop_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    print(opts.dataset)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    elif opts.dataset == 'cityscapes_domain':
        dataset = CityscapesSegmentationIncrementalDomain
    elif opts.dataset == 'cityscapes':
        dataset = CityscapesSegmentationIncremental
    elif opts.dataset == 'bdd':
        dataset = BDDSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        buffer_path=path_base + f"/buffer.npy",
        masking=not opts.no_mask,
        overlap=opts.overlap,
        disable_background=opts.disable_background,
        data_masking=opts.data_masking,
        test_on_val=opts.test_on_val,
        step=opts.step,
        buffer_size=opts.buffer_size
    )

    if not opts.no_cross_val:  # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:  # don't use cross_val
        val_dst = dataset(
            root=opts.data_root,
            train=False,
            transform=val_transform,
            labels=list(labels),
            labels_old=list(labels_old),
            idxs_path=path_base + f"/val-{opts.step}.npy",
            masking=not opts.no_mask,
            overlap=True,
            disable_background=opts.disable_background,
            data_masking=opts.data_masking,
            step=opts.step
        )

    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(
        root=opts.data_root,
        train=opts.val_on_trainset,
        transform=val_transform,
        labels=list(labels_cum),
        idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy",
        disable_background=opts.disable_background,
        test_on_val=opts.test_on_val,
        step=opts.step,
        ignore_test_bg=opts.ignore_test_bg
    )

    return train_dst, val_dst, test_dst, len(labels_cum)


def main(opts):
    print(f"Learning for {len(opts.step)} with lrs={opts.lr}.")
    all_val_scores = []
    for i, (step, lr) in enumerate(zip(copy.deepcopy(opts.step), copy.deepcopy(opts.lr))):
        if i > 0:
            opts.step_ckpt = None

        opts.step = step
        opts.lr = lr

    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(
            logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step
        )
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    new_cls = tasks.get_task_labels_attr(opts.dataset, opts.task, opts.step)
    print('new cls for attribution ',new_cls)

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(
            opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)
        )
        [model_old] = amp.initialize(
            [model_old.to(device)], opt_level=opts.opt_level, cast_model_type=torch.float32
        )
        model_old = DistributedDataParallel(model_old.to(device))

        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"{opts.checkpoint}{task_name}_{opts.name}_{opts.step - 1}.pth"

        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            path = f"{opts.checkpoint}{task_name}_{opts.name}_{opts.step - 1}.pth"
            model_old.load_state_dict(
                 step_checkpoint['model_state'], strict=False
            )
            #model_old.to(device="cuda")
            print("Old Model weights loaded for computing attributions")
            del step_checkpoint

    def cast_running_stats(m):
        if isinstance(m, ABN):
            m.running_mean = m.running_mean.float()
            m.running_var = m.running_var.float()

    imp_c = None
    if model_old is not None and not opts.no_att and not opts.random_channels and not opts.test:
        torch.cuda.empty_cache()
        model_old.eval()

        exp_loader = data.DataLoader(
            train_dst,
            batch_size=1,
            sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
            num_workers=1,
            drop_last=True
        )

        def prev(inp):
            out = model_old(inp)[0]
            return out.sum(dim=(2,3))

        torch.cuda.empty_cache()
        lig = LayerIntegratedGradients(forward_func=prev, layer=model_old.module.cls[0])
        print('bkg cls ', model_old.module.cls[0].weight[0].data.shape)

        gc.collect()
        attr = []
        miss = 0
        imp = []
        if opts.dataset == 'ade':
            n_step = 30
        else:
            n_step = 50

        for m in new_cls:
            print('class ',str(m))

            for cur_step, (images, labels) in enumerate(exp_loader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if opts.mask_att:
                    mask = labels == m

                    mask = F.interpolate(mask.unsqueeze(0).float(), size=([32, 32]), mode="nearest")
                    mask = mask.expand([1, 256, 32, 32])
                    if mask[0][0].sum() < 1:
                        miss += 1
                        continue

                torch.cuda.empty_cache()
                del labels
                with torch.no_grad():
                    attribution = lig.attribute(images, target=0, n_steps=n_step, attribute_to_layer_input=True)
                print(cur_step, attribution[0].shape)
                attr.append(attribution[0]*mask)

                del images
                del mask
                del attribution
                torch.cuda.empty_cache()
                gc.collect()
                if cur_step > 1999 and opts.dataset == 'ade' and opts.task != "50":
                    break
                if opts.task == "50" and cur_step > 4999:
                    break

            print('number of missed images ',miss)
            att = torch.cat(attr, dim=0)
            print('att ',att.shape)

            print('Max after mean')
            att = torch.mean(att, dim=0)
            print('avg att ',att.shape)
            att = nn.MaxPool2d(32)(att)
            print('after max pool ',att.shape)

            top = int(opts.att)
            print('top weights ',top)
            if top != 0:
                if top == 10:
                    high = 26
                elif top == 25:
                    high = 64
                elif top == 50:
                    high = 128
                elif top == 75:
                    high = 192
                top_att = torch.topk(att.squeeze(),high)[1]
                imp_c = torch.zeros_like(att)
                imp_c[top_att] = 1
                imp_c = imp_c > 0
            if imp_c is not None:
                print('important channels ',imp_c.shape, imp_c.sum())
            else:
                print("imp_c is None")

            imp.append(imp_c)

    imp = torch.stack(imp)
    print("Attributions generated for ",str(len(imp))," classes")
    att_method = path.split('/')[-2]
    path = f"channels/{att_method}"
    os.makedirs(path, exist_ok=True)
    imp_name = f"channels/{att_method}/imp_{opts.method}_{opts.dataset}_{opts.task}_{opts.step}.pt"
    torch.save(imp,imp_name)
    if low_c is not None:
        low_name = f"channels/{att_method}/low_{opts.method}_{opts.dataset}_{opts.task}_{opts.step}.pt"
        torch.save(low_c,low_name)


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs(f"{opts.checkpoint}", exist_ok=True)

    main(opts)
