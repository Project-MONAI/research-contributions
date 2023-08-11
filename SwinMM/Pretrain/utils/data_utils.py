# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import timm.data
from utils import dataset_in_memory

from monai.data import DataLoader, Dataset, DistributedSampler, load_decathlon_datalist
from monai.data.utils import list_data_collate
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)


def get_loader(args):
    splits0 = "/dataset00_BTCV.json"
    # splits1 = "/dataset01_BrainTumour.json"
    splits2 = "/dataset02_Heart.json"
    splits3 = "/dataset03_Liver.json"
    splits4 = "/dataset04_Hippocampus.json"
    # splits5 = "/dataset05_Prostate.json"
    splits6 = "/dataset06_Lung.json"
    splits7 = "/dataset07_Pancreas.json"
    splits8 = "/dataset08_HepaticVessel.json"
    splits9 = "/dataset09_Spleen.json"
    splits10 = "/dataset10_Colon.json"
    splits11 = "/dataset11_TCIAcovid19.json"
    splits12 = "/dataset12_WORD.json"
    splits13 = "/dataset13_AbdomenCT-1K.json"
    splits14 = "/dataset_HNSCC.json"
    splits15 = "/dataset_TCIAcolon.json"
    splits16 = "/dataset_LIDC.json"

    list_dir = "./jsons"
    jsonlist0 = list_dir + splits0
    # jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    # jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7
    jsonlist8 = list_dir + splits8
    jsonlist9 = list_dir + splits9
    jsonlist10 = list_dir + splits10
    jsonlist11 = list_dir + splits11
    jsonlist12 = list_dir + splits12
    jsonlist13 = list_dir + splits13
    jsonlist14 = list_dir + splits14
    jsonlist15 = list_dir + splits15
    jsonlist16 = list_dir + splits16

    datadir0 = "./dataset/dataset00_BTCV"
    # datadir1 = "./dataset/dataset01_BrainTumour"
    datadir2 = "./dataset/dataset02_Heart"
    datadir3 = "./dataset/dataset03_Liver"
    datadir4 = "./dataset/dataset04_Hippocampus"
    # datadir5 = "./dataset/dataset05_Prostate"
    datadir6 = "./dataset/dataset06_Lung"
    datadir7 = "./dataset/dataset07_Pancreas"
    datadir8 = "./dataset/dataset08_HepaticVessel"
    datadir9 = "./dataset/dataset09_Spleen"
    datadir10 = "./dataset/dataset10_Colon"
    datadir11 = "./dataset/dataset11_TCIAcovid19"
    datadir12 = "./dataset/dataset12_WORD"
    datadir13 = "./dataset/dataset13_AbdomenCT-1K"
    datadir14 = "./dataset/dataset_HNSCC"
    datadir15 = "./dataset/dataset_TCIAcolon"
    datadir16 = "./dataset/dataset_LIDC"

    datalist = []
    for json_path, base_dir in zip(
        [
            jsonlist0,
            jsonlist2,
            jsonlist3,
            jsonlist4,
            jsonlist6,
            jsonlist7,
            jsonlist8,
            jsonlist9,
            jsonlist10,
            jsonlist11,
            jsonlist12,
            jsonlist13,
            # jsonlist14,
            # jsonlist15,
            # jsonlist16,
        ],
        [
            datadir0,
            datadir2,
            datadir3,
            datadir4,
            datadir6,
            datadir7,
            datadir8,
            datadir9,
            datadir10,
            datadir11,
            datadir12,
            datadir13,
            # datadir14,
            # datadir15,
            # datadir16,
        ],
    ):
        datalist_i = load_decathlon_datalist(json_path, False, "training", base_dir=base_dir)
        datalist.extend([{"image": item["image"]} for item in datalist_i])

    print("Dataset all training: number of data: {}".format(len(datalist)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    if args.use_normal_dataset:
        train_ds = Dataset(data=datalist, transform=train_transforms)
    else:
        train_ds = dataset_in_memory.CachedDataset(
            data=datalist,
            transform=train_transforms,
            dataset_name="pretrain_train",
            hosts=[{"host": "localhost", "port": str(port)} for port in args.redis_ports],
            cluster_mode=True,
            capacity_per_node=200 * 1024 * 1024 * 1024,
            writer_buffer_size=0,  # Disable write buffer
        )

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    loader_class = DataLoader
    if not args.nouse_multi_epochs_loader:
        loader_class = timm.data.loader.MultiEpochsDataLoader
    train_loader = loader_class(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=list_data_collate,
    )

    return train_loader, None
