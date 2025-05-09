import json
import os
import pdb

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

from monai.data import (
    CacheDataset,
    Dataset,
    ThreadDataLoader,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
)
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    # pdb.set_trace()
    return tr, val


class MaskGenerator:
    def __init__(self, input_size=96, mask_patch_size=16, model_patch_size=(2, 2, 2), mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size[0]
        self.mask_ratio = mask_ratio
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size**3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        return mask


class Transform:
    def __init__(self, args):
        if args.iso_spacing:
            self.transform_ct = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                    SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                    RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                    ToTensord(keys=["image"]),
                ]
            )
            self.transform_mri = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                    SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                    RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                    ToTensord(keys=["image"]),
                ]
            )
        else:
            self.transform_ct = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                    SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                    RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                    ToTensord(keys=["image"]),
                ]
            )
            self.transform_mri = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    # Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=("bilinear")),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                    SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                    RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                    ToTensord(keys=["image"]),
                ]
            )

        model_patch_size = args.patch_size

        self.mask_generator = MaskGenerator(
            input_size=args.img_size,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=args.mask_ratio,
        )

    def __call__(self, img):
        if img["class"] == "ct":
            img = self.transform_ct(img)
        else:
            img = self.transform_mri(img)
        # img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(args):
    if args.local:
        splits2 = "/60659/dataset_0.json"
        datadir2 = "/NGC_Datasets/60659/"
        datalist = load_decathlon_datalist(splits2, False, "training", base_dir=datadir2)
        val_files = load_decathlon_datalist(splits2, False, "validation", base_dir=datadir2)
    elif args.onlycovid:
        splits1 = "/dataset_LUNA16_0.json"
        splits2 = "/dataset_TCIAcovid19_0.json"
        splits3 = "/dataset_HNSCC_0.json"
        splits4 = "/dataset_TCIAcolon_v2_0.json"
        splits5 = "/dataset_LIDC_0.json"
        list_dir = "./jsons"
        jsonlist1 = list_dir + splits1
        jsonlist2 = list_dir + splits2
        jsonlist3 = list_dir + splits3
        jsonlist4 = list_dir + splits4
        jsonlist5 = list_dir + splits5
        datadir1 = "/dataset/dataset1"
        datadir2 = "/dataset/dataset2"
        datadir3 = "/dataset/dataset3"
        datadir4 = "/dataset/dataset4"
        datadir5 = "/dataset/dataset8"

        datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
        print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))

        vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)

        datalist = datalist2
        val_files = vallist2

        for i in range(len(datalist)):
            tmp = datalist[i]
            imgname = tmp["image"]
            if "t1" in imgname:
                tmp["class"] = "t1"
            elif "t2" in imgname:
                tmp["class"] = "t2"
            elif "t1ce" in imgname:
                tmp["class"] = "t1ce"
            elif "flair" in imgname:
                tmp["class"] = "flair"
            else:
                tmp["class"] = "ct"

        for i in range(len(val_files)):
            tmp = val_files[i]
            imgname = tmp["image"]
            if "t1" in imgname:
                tmp["class"] = "t1"
            elif "t2" in imgname:
                tmp["class"] = "t2"
            elif "t1ce" in imgname:
                tmp["class"] = "t1ce"
            elif "flair" in imgname:
                tmp["class"] = "flair"
            else:
                tmp["class"] = "ct"

        # datalist = new_datalist1 + datalist2 + datalist3 + datalist4
        # val_files = vallist1 + vallist2 + vallist3 + vallist4

        print("Dataset all training: number of data: {}".format(len(datalist)))
        print("Dataset all validation: number of data: {}".format(len(val_files)))
    else:
        splits1 = "/dataset_LUNA16_0.json"
        splits2 = "/dataset_TCIAcovid19_0.json"
        splits3 = "/dataset_HNSCC_0.json"
        splits4 = "/dataset_TCIAcolon_v2_0.json"
        splits8 = "/dataset_LIDC_0.json"
        splits9 = "/brats21_pre.json"

        list_dir = "./jsons"
        jsonlist1 = list_dir + splits1
        jsonlist2 = list_dir + splits2
        jsonlist3 = list_dir + splits3
        jsonlist4 = list_dir + splits4
        jsonlist8 = list_dir + splits8
        jsonlist9 = list_dir + splits9
        # jsonlist10 = list_dir + splits10

        datadir1 = "/dataset/dataset1"
        datadir2 = "/dataset/dataset2"
        datadir3 = "/dataset/dataset3"
        datadir4 = "/dataset/dataset4"
        datadir8 = "/dataset/dataset8"
        datadir9 = "/dataset/dataset9"
        datadir10 = "/dataset/dataset10"

        datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
        print("Dataset 1 LUNA16: number of data: {}".format(len(datalist1)))
        new_datalist1 = []
        for item in datalist1:
            item_dict = {"image": item["image"]}
            new_datalist1.append(item_dict)
        # dataset 2
        datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
        print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
        # dataset 3
        datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
        print("Dataset 3 HNSCC: number of data: {}".format(len(datalist3)))
        # dataset 4
        datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
        print("Dataset 4 TCIA Colon: number of data: {}".format(len(datalist4)))

        datalist8 = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
        print("Dataset 8: number of data: {}".format(len(datalist8)))

        datalist9, vallist9 = datafold_read(datalist=jsonlist9, basedir=datadir9, fold=0)
        print("Dataset 9: number of data: {}".format(len(datalist9)))

        # datalist10 = load_decathlon_datalist(jsonlist10, False, "training", base_dir=datadir10)
        # print('Dataset 10: number of data: {}'.format(len(datalist10)))

        vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
        vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
        vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
        vallist4 = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
        vallist8 = load_decathlon_datalist(jsonlist8, False, "validation", base_dir=datadir8)
        # vallist9 = load_decathlon_datalist(jsonlist9, False, "validation", base_dir=datadir9)
        # vallist10 = load_decathlon_datalist(jsonlist10, False, "validation", base_dir=datadir10)

        datalist_ct = new_datalist1 + datalist2 + datalist3 + datalist4 + datalist8
        datalist_mri = datalist9  # + datalist10

        datalist = datalist_mri + datalist_ct
        val_files = vallist1 + vallist2 + vallist3 + vallist4 + vallist8 + vallist9  # + vallist10
        # val_files = vallist9
        # pdb.set_trace()
        # datalist = new_datalist1 + datalist2 + datalist3 + datalist4
        # val_files = vallist1 + vallist2 + vallist3 + vallist4

        # pdb.set_trace()

        for i in range(len(datalist)):
            tmp = datalist[i]
            imgname = tmp["image"]
            if "t1" in imgname:
                tmp["class"] = "t1"
            elif "t2" in imgname:
                tmp["class"] = "t2"
            elif "t1ce" in imgname:
                tmp["class"] = "t1ce"
            elif "flair" in imgname:
                tmp["class"] = "flair"
            else:
                tmp["class"] = "ct"

        for i in range(len(val_files)):
            tmp = val_files[i]
            imgname = tmp["image"]
            if "t1" in imgname:
                tmp["class"] = "t1"
            elif "t2" in imgname:
                tmp["class"] = "t2"
            elif "t1ce" in imgname:
                tmp["class"] = "t1ce"
            elif "flair" in imgname:
                tmp["class"] = "flair"
            else:
                tmp["class"] = "ct"

        print("Dataset CT training: number of data: {}".format(len(datalist_ct)))
        print("Dataset MRI training: number of data: {}".format(len(datalist_mri)))

        print("Dataset all validation: number of data: {}".format(len(val_files)))

    transform = Transform(args)
    # dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=1.0, num_workers=8, cache_num=4759)
    # dataset_val = CacheDataset(data=val_files, transform=transform, cache_rate=1.0, num_workers=8, cache_num=260)
    dataset_train = CacheDataset(data=datalist, transform=transform, cache_rate=1.0, num_workers=8)
    dataset_val = CacheDataset(data=val_files, transform=transform, cache_rate=1.0, num_workers=8)

    sampler_train = DistributedSampler(
        dataset_train, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )

    # dataloader_train = ThreadDataLoader(dataset_train, num_workers=8, sampler=sampler_train, batch_size=args.batch_size,
    #                                     collate_fn=collate_fn)
    # dataloader_val = ThreadDataLoader(dataset_val, num_workers=8,sampler=sampler_val, batch_size=1, collate_fn=collate_fn)

    dataloader_train = DataLoader(
        dataset_train,
        args.batch_size,
        sampler=sampler_train,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    dataloader_val = DataLoader(
        dataset_val, 1, sampler=sampler_val, num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    return dataloader_train, dataloader_val
