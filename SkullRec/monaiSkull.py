import argparse
import glob
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch

from monai.data import DataLoader, Dataset, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import Inferer, SimpleInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import AutoEncoder
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Resized,
    SaveImaged,
    ToDeviced,
)
from monai.utils import first, set_determinism

"""########## Dataset diretory
"""


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


train_images = sorted(glob.glob(os.path.join("./dataset/train/defective_skull/", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join("./dataset/train/complete_skull/", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-79], data_dicts[-79:]


test_images = sorted(glob.glob(os.path.join("./dataset/test/defects_cranial/", "*.nii.gz")))

test_data = [{"image": image} for image in test_images]


"""########## Transforms
"""


set_determinism(seed=0)


test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Resized(keys=["image"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys="image"),
    ]
)


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=["image", "label"]),
        # ToDeviced(keys=["image", "label"],device='cuda:0'),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=["image", "label"]),
    ]
)

test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Resized(keys=["image"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys="image"),
    ]
)

post_transforms = Compose(
    [
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        # Specify here the output directory. Default is './out_cranial_monai' in
        # the current directory
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir="./output_monai",
            output_postfix="completed",
            resample=False,
        ),
    ]
)


"""########## Load datasets and apply transforms
"""

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)


"""########## Network and training specifications
"""

device = torch.device("cuda:0")
model = AutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 64, 128, 128, 256),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=0,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")


max_epochs = 4
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.phase == "train":
        print("**********************start traininig*************************")

        for epoch in range(max_epochs):
            print(" -" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    inferer = SimpleInferer()
                    for val_data in val_loader:
                        val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))

                        val_outputs = inferer(val_inputs, model)

                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    elif args.phase == "test":
        print("**************generating predictions on the test set***************")
        weights_dir = "./pre_trained_weights/"
        model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
        model.eval()
        with torch.no_grad():
            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                inferer = SimpleInferer()
                test_data["pred"] = inferer(test_inputs, model)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                test_output = from_engine(["pred"])(test_data)
                print(test_output[0].detach().cpu().shape)
