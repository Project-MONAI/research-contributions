import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from icon_registration import config


def get_dataset_mnist(split, number=5):
    ds = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./files/",
            transform=torchvision.transforms.ToTensor(),
            download=True,
            train=(split == "train"),
        ),
        batch_size=500,
    )
    images = []
    for _, batch in enumerate(ds):
        label = np.array(batch[1])
        batch_nines = label == number
        images.append(np.array(batch[0])[batch_nines])
    images = np.concatenate(images)

    ds = torch.utils.data.TensorDataset(torch.Tensor(images))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=128,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_1d(data_size=128, samples=6000, batch_size=128):
    x = np.arange(0, 1, 1 / data_size)
    x = np.reshape(x, (1, data_size))
    cx = np.random.random((samples, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1)) * 0.2 + 0.2

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2) - r))

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(circles, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_triangles(
    split=None, data_size=128, hollow=False, samples=6000, batch_size=128
):
    x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((samples, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((samples, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(images, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_retina(
    extra_deformation=False,
    downsample_factor=4,
    blur_sigma=None,
    warps_per_pair=20,
    fixed_vertical_offset=None,
    include_boundary=False,
):
    try:
        import elasticdeform
        import hub
    except:

        raise Exception(
            """the retina dataset requires the dependencies hub and elasticdeform.
            Try pip install hub elasticdeform"""
        )

    ds_name = f"retina{extra_deformation}{downsample_factor}{blur_sigma}{warps_per_pair}{fixed_vertical_offset}{include_boundary}.trch"

    import os

    if os.path.exists(ds_name):
        augmented_ds1_tensor, augmented_ds2_tensor = torch.load(ds_name)
    else:

        res = []
        for batch in hub.load("hub://activeloop/drive-train").pytorch(
            num_workers=0, batch_size=4, shuffle=False
        ):
            if include_boundary:
                res.append(batch["manual_masks/mask"] ^ batch["masks/mask"])
            else:
                res.append(batch["manual_masks/mask"])
        res = torch.cat(res)
        ds_tensor = res[:, None, :, :, 0] * -1.0 + (not include_boundary)

        if fixed_vertical_offset is not None:
            ds2_tensor = torch.cat(
                [torch.zeros(20, 1, fixed_vertical_offset, 565), ds_tensor], axis=2
            )
            ds1_tensor = torch.cat(
                [ds_tensor, torch.zeros(20, 1, fixed_vertical_offset, 565)], axis=2
            )
        else:
            ds2_tensor = ds_tensor
            ds1_tensor = ds_tensor

        warped_tensors = []
        print("warping images to generate dataset")
        for _ in tqdm.tqdm(range(warps_per_pair)):
            ds_2_list = []
            for el in ds2_tensor:
                case = el[0]
                # TODO implement random warping on gpu
                case_warped = np.array(case)
                if extra_deformation:
                    case_warped = elasticdeform.deform_random_grid(
                        case_warped, sigma=60, points=3
                    )
                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=25, points=3
                )

                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=12, points=6
                )
                ds_2_list.append(torch.tensor(case_warped)[None, None, :, :])
                ds_2_tensor = torch.cat(ds_2_list)
            warped_tensors.append(ds_2_tensor)

        augmented_ds2_tensor = torch.cat(warped_tensors)
        augmented_ds1_tensor = torch.cat([ds1_tensor for _ in range(warps_per_pair)])

        torch.save((augmented_ds1_tensor, augmented_ds2_tensor), ds_name)

    batch_size = 10
    import torchvision.transforms.functional as Fv

    if blur_sigma is None:
        ds1 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds1_tensor, downsample_factor)
        )
    else:
        ds1 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds1_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )
    d1 = torch.utils.data.DataLoader(
        ds1,
        batch_size=batch_size,
        shuffle=False,
    )
    if blur_sigma is None:
        ds2 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds2_tensor, downsample_factor)
        )
    else:
        ds2 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds2_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )

    d2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=batch_size,
        shuffle=False,
    )

    return d1, d2


def get_dataset_sunnyside(split, scale=1):
    import pickle

    with open("/playpen/tgreer/sunnyside.pickle", "rb") as f:
        array = pickle.load(f)
    if split == "train":
        array = array[1000:]
    elif split == "test":
        array = array[:1000]
    else:
        raise ArgumentError()

    array = array[:, :, :, 0]
    array = np.expand_dims(array, 1)
    array = array * scale
    array1 = array[::2]
    array2 = array[1::2]
    array12 = np.concatenate([array2, array1])
    array21 = np.concatenate([array1, array2])
    ds = torch.utils.data.TensorDataset(torch.Tensor(array21), torch.Tensor(array12))
    ds = torch.utils.data.DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
    )
    return ds


def get_cartilage_dataset():
    cartilage = torch.load("/playpen/tgreer/cartilage_uint8s.trch")
    return cartilage


def get_knees_dataset():
    brains = torch.load("/playpen/tgreer/kneestorch")
    #    with open("/playpen/tgreer/cartilage_eval_oriented", "rb") as f:
    #        cartilage = pickle.load(f)

    medbrains = []
    for b in brains:
        medbrains.append(F.avg_pool3d(b, 4))

    return brains, medbrains

def get_copdgene_dataset(data_folder, cache_folder="./data_cache", lung_only=True, downscale=2):
    '''
    This function load the preprocessed COPDGene train set.
    '''
    import os
    def process(iA, downscale, clamp=[-1000, 0], isSeg=False):
        iA = iA[None, None, :, :, :]
        #SI flip
        iA = torch.flip(iA, dims=(2,))
        if isSeg:
            iA = iA.float()
            iA = torch.nn.functional.max_pool3d(iA, downscale)
            iA[iA>0] = 1
        else:
            iA = torch.clip(iA, clamp[0], clamp[1]) + clamp[0]
            #TODO: For compatibility to the processed dataset(ranges between -1 to 0) used in paper, we subtract -1 here.
            # Should remove -1 later.
            iA = iA / torch.max(iA) - 1.
            iA = torch.nn.functional.avg_pool3d(iA, downscale)
        return iA

    cache_name = f"{cache_folder}/lungs_train_{downscale}xdown_scaled"
    if os.path.exists(cache_name):
        imgs = torch.load(cache_name, map_location='cpu')
        if lung_only:
            try:
                masks = torch.load(f"{cache_folder}/lungs_seg_train_{downscale}xdown_scaled", map_location='cpu')
            except FileNotFoundError:
                print("Segmentation data not found.")

    else:
        import itk
        import glob
        with open(f"{data_folder}/splits/train.txt") as f:
            pair_paths = f.readlines()
        imgs = []
        masks = []
        for name in tqdm.tqdm(list(iter(pair_paths))[:]):
            name = name[:-1] # remove newline

            image_insp = torch.tensor(np.asarray(itk.imread(glob.glob(f"{data_folder} /{name}/{name}_INSP_STD*_COPD_img.nii.gz")[0])))
            image_exp= torch.tensor(np.asarray(itk.imread(glob.glob(f"{data_folder} /{name}/{name}_EXP_STD*_COPD_img.nii.gz")[0])))
            imgs.append((process(image_insp), process(image_exp)))

            seg_insp = torch.tensor(np.asarray(itk.imread(glob.glob(f"{data_folder} /{name}/{name}_INSP_STD*_COPD_label.nii.gz")[0])))
            seg_exp= torch.tensor(np.asarray(itk.imread(glob.glob(f"{data_folder} /{name}/{name}_EXP_STD*_COPD_label.nii.gz")[0])))
            masks.append((process(seg_insp, True), process(seg_exp, True)))

        torch.save(imgs, f"{cache_folder}/lungs_train_{downscale}xdown_scaled")
        torch.save(masks, f"{cache_folder}/lungs_seg_train_{downscale}xdown_scaled")
    
    if lung_only:
        imgs = torch.cat([(torch.cat(d, 1)+1)*torch.cat(m, 1) for d,m in zip(imgs, masks)], dim=0)
    else:
        imgs = torch.cat([torch.cat(d, 1)+1 for d in imgs], dim=0)
    return torch.utils.data.TensorDataset(imgs)

def get_learn2reg_AbdomenCTCT_dataset(data_folder, cache_folder="./data_cache", clamp=[-1000,0], downscale=1):
    '''
    This function will return the training dataset of AbdomenCTCT registration task in learn2reg.
    '''

    # Check whether we have cached the dataset
    import os

    cache_name = f"{cache_folder}/learn2reg_abdomenctct_train_set_clamp{clamp}scale{downscale}"
    if os.path.exists(cache_name):
        imgs = torch.load(cache_name)
    else:
        import json
        import itk
        import glob
        with open(f"{data_folder}/AbdomenCTCT_dataset.json", 'r') as data_info:
            data_info = json.loads(data_info.read())
        train_cases = [c["image"].split('/')[-1].split('.')[0] for c in data_info["training"]]
        imgs = [np.asarray(itk.imread(glob.glob(data_folder + "/imagesTr/" + i + ".nii.gz")[0])) for i in train_cases]
        
        imgs = torch.Tensor(np.expand_dims(np.array(imgs), axis=1)).float()
        imgs = (torch.clamp(imgs, clamp[0], clamp[1]) - clamp[0])/(clamp[1] - clamp[0])

        # Cache the data
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        torch.save(imgs, cache_name)
    
    # Scale down the image
    if downscale > 1:
        imgs = F.avg_pool3d(imgs, downscale)
    return torch.utils.data.TensorDataset(imgs)

def get_learn2reg_lungCT_dataset(data_folder, cache_folder="./data_cache", lung_only=True, clamp=[-1000,0], downscale=1):
    '''
    This function will return the training dataset of LungCT registration task in learn2reg.
    '''
    import os

    cache_name = f"{cache_folder}/learn2reg_lung_train_set_lung_only" if lung_only else f"{cache_folder}/learn2reg_lung_train_set"
    cache_name += f"_clamp{clamp}scale{downscale}"
    if os.path.exists(cache_name):
        imgs = torch.load(cache_name)
    else:
        import json
        import itk
        import glob
        with open(f"{data_folder}/NLST_dataset.json", 'r') as data_info:
            data_info = json.loads(data_info.read())
        train_pairs = [[p['fixed'].split('/')[-1], p['moving'].split('/')[-1]] for p in data_info["training_paired_images"]]
        imgs = []
        for p in train_pairs:
            img = np.array([np.asarray(itk.imread(glob.glob(data_folder + "/imagesTr/" + i)[0])) for i in p])
            if lung_only:
                mask = np.array([np.asarray(itk.imread(glob.glob(data_folder + "/" + "/masksTr/" + i)[0])) for i in p])
                img = img * mask + clamp[0] * (1 - mask)
            imgs.append(img)
        
        imgs = torch.Tensor(np.array(imgs)).float()
        imgs = (torch.clamp(imgs, clamp[0], clamp[1]) - clamp[0])/(clamp[1] - clamp[0])

        # Cache the data
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        torch.save(imgs, cache_name)
    
    # Scale down the image
    if downscale > 1:
        imgs = F.avg_pool3d(imgs, downscale)
    return torch.utils.data.TensorDataset(imgs)


def make_batch(data, BATCH_SIZE, SCALE):
    image = torch.cat([random.choice(data) for _ in range(BATCH_SIZE)])
    image = image.reshape(BATCH_SIZE, 1, SCALE * 40, SCALE * 96, SCALE * 96)
    image = image.to(config.device)
    return image
