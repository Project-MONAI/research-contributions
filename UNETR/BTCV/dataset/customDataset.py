import os
from torch.utils.data import DataLoader
from monai.data import Dataset
import monai.transforms as transforms
import torch

from config import NIFTI_DATA_ROOT, NIFTI_LABEL_ROOT, PREDICT_DATA_ROOT

def _get_collate_fn(isTrain:bool):
    def collate_fn(batch):
        '''collate function'''
        images = []
        labels = []
        if isTrain:
            for p in batch: # [ {"image": (C, H, W ,D), "label": (C, H, W ,D)} , ...]
                for i in range(len(p)): # list, RandCropByPosNegLabeld will produce multiple samples
                    images.append(p[i]['image'])
                    labels.append(p[i]['label'])
        else:
            for p in batch:
                images.append(p['image'])
                labels.append(p['label'])

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        # keep images float and labels long for loss functions
        return [images.float(), labels.long()]

    return collate_fn

def getDatasetLoader(args):
    exts = (".nii", ".nii.gz")
    img_names = {f for f in os.listdir(NIFTI_DATA_ROOT) if f.endswith(exts) and os.path.isfile(os.path.join(NIFTI_DATA_ROOT, f))}
    lbl_names = {f for f in os.listdir(NIFTI_LABEL_ROOT) if f.endswith(exts) and os.path.isfile(os.path.join(NIFTI_LABEL_ROOT, f))}
    common = sorted(img_names & lbl_names)
    if not common:
        raise RuntimeError(f"No matching image/label pairs found in {NIFTI_DATA_ROOT} and {NIFTI_LABEL_ROOT}")
    dataDicts = [{"image": os.path.join(NIFTI_DATA_ROOT, f), "label": os.path.join(NIFTI_LABEL_ROOT, f)} for f in common]

    trainDicts, valDicts = _splitList(dataDicts)

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

    trainDataset = Dataset(data=trainDicts, transform=train_transform)
    valDataset = Dataset(data=valDicts, transform=val_transform)
    trainLoader = DataLoader(trainDataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers, collate_fn=_get_collate_fn(isTrain=True))
    valLoader = DataLoader(valDataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers, collate_fn=_get_collate_fn(isTrain=False))
    loader = [trainLoader, valLoader]

    return loader

def _splitList(l, trainRatio:float = 0.8):
    totalNum = len(l)
    splitIdx = int(totalNum * trainRatio)

    return l[:splitIdx], l[splitIdx :]

def getPredictLoader(args):
    dataName = [d for d in os.listdir(PREDICT_DATA_ROOT)]
    dataDicts = [{"image": f"{os.path.join(PREDICT_DATA_ROOT, d)}" } for d in dataName]

    preTransform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
                transforms.EnsureTyped(keys=["image"], track_meta=True),
            ]
        )
    valDataset = Dataset(data=dataDicts, transform=preTransform)
    valLoader = DataLoader(valDataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)

    return valLoader, preTransform
