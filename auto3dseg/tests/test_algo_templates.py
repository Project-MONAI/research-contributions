# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import sys
import unittest

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.apps.auto3dseg import AlgoEnsembleBestN, AlgoEnsembleBuilder, BundleGen, DataAnalyzer
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils.enums import AlgoKeys

sim_datalist = {
    "testing": [
        {"image": "val_image_001.nii.gz", "label": "val_label_001.nii.gz"},
        {"image": "val_image_002.nii.gz", "label": "val_label_002.nii.gz"},
    ],
    "training": [
        {"fold": 0, "image": "tr_image_001.nii.gz", "label": "tr_label_001.nii.gz"},
        {"fold": 0, "image": "tr_image_002.nii.gz", "label": "tr_label_002.nii.gz"},
        {"fold": 0, "image": "tr_image_003.nii.gz", "label": "tr_label_003.nii.gz"},
        {"fold": 0, "image": "tr_image_004.nii.gz", "label": "tr_label_004.nii.gz"},
        {"fold": 1, "image": "tr_image_005.nii.gz", "label": "tr_label_005.nii.gz"},
        {"fold": 1, "image": "tr_image_006.nii.gz", "label": "tr_label_006.nii.gz"},
        {"fold": 1, "image": "tr_image_007.nii.gz", "label": "tr_label_007.nii.gz"},
        {"fold": 1, "image": "tr_image_008.nii.gz", "label": "tr_label_008.nii.gz"},
        {"fold": 2, "image": "tr_image_009.nii.gz", "label": "tr_label_009.nii.gz"},
        {"fold": 2, "image": "tr_image_010.nii.gz", "label": "tr_label_010.nii.gz"},
        {"fold": 2, "image": "tr_image_011.nii.gz", "label": "tr_label_011.nii.gz"},
        {"fold": 2, "image": "tr_image_012.nii.gz", "label": "tr_label_012.nii.gz"},
    ],
}

algo_templates = os.path.join("auto3dseg", "algorithm_templates")

sys.path.insert(0, algo_templates)

num_images_per_batch = 2
num_epochs = 2
num_epochs_per_validation = 1
num_warmup_epochs = 1

train_params = {
    "num_epochs_per_validation": num_epochs_per_validation,
    "num_images_per_batch": num_images_per_batch,
    "num_epochs": num_epochs,
    "num_warmup_epochs": num_warmup_epochs,
    "use_pretrain": False,
    "pretrained_path": "",
}

pred_param = {"files_slices": slice(0, 1), "mode": "mean", "sigmoid": True}

SIM_TEST_CASES = [
    [{"sim_dim": (24, 24, 24), "modality": "MRI", "dints_search": False}],
    [{"sim_dim": (320, 320, 15), "modality": "MRI", "dints_search": False}],
    [{"sim_dim": (32, 32, 32), "modality": "CT", "dints_search": False}],
    [{"sim_dim": (32, 32, 32), "modality": "CT", "dints_search": True}],
]


def create_sim_data(dataroot, sim_datalist, sim_dim, **kwargs):
    """
    Create simulated data using create_test_image_3d.

    Args:
        dataroot: data directory path that hosts the "nii.gz" image files.
        sim_datalist: a list of data to create.
        sim_dim: the image sizes, e.g. a tuple of (64, 64, 64).
    """
    if not os.path.isdir(dataroot):
        os.makedirs(dataroot)

    # Generate a fake dataset
    for d in sim_datalist["testing"] + sim_datalist["training"]:
        im, seg = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
        nib_image = nib.Nifti1Image(im, affine=np.eye(4))
        image_fpath = os.path.join(dataroot, d["image"])
        nib.save(nib_image, image_fpath)

        if "label" in d:
            nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
            label_fpath = os.path.join(dataroot, d["label"])
            nib.save(nib_image, label_fpath)


def auto_run(work_dir, data_src_cfg, algos, search=False):
    """
    Similar to Auto3DSeg AutoRunner, auto_run function executes the data analyzer, bundle generation,
    and ensemble.

    Args:
        work_dir: working directory path.
        data_src_cfg: the input is a dictionary that includes dataroot, datalist and modality keys.
        algos: the algorithm templates (a dictionary of Algo classes).
        search: for dints only. Switch to include the search phase or not.

    Returns:
        A list of predictions made the ensemble inference.
    """

    data_src_cfg_file = os.path.join(work_dir, "input.yaml")
    ConfigParser.export_config_file(data_src_cfg, data_src_cfg_file, fmt="yaml")

    datastats_file = os.path.join(work_dir, "datastats.yaml")
    analyser = DataAnalyzer(data_src_cfg["datalist"], data_src_cfg["dataroot"], output_path=datastats_file)
    analyser.get_all_case_stats()

    bundle_generator = BundleGen(
        algo_path=work_dir,
        templates_path_or_url=algo_templates,
        algos=algos,
        data_stats_filename=datastats_file,
        data_src_cfg_name=data_src_cfg_file,
    )
    bundle_generator.generate(work_dir, num_fold=1)
    history = bundle_generator.get_history()

    for algo_dict in history:
        name = algo_dict[AlgoKeys.ID]
        algo = algo_dict[AlgoKeys.ALGO]
        if "dints" in name:
            algo.train(train_params=train_params, search=search)
        else:
            algo.train(train_params=train_params)

    builder = AlgoEnsembleBuilder(history, data_src_cfg_file)
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=len(history)))  # inference all models
    preds = builder.get_ensemble()(pred_param)
    return preds


class TestAlgoTemplates(unittest.TestCase):
    @parameterized.expand(SIM_TEST_CASES)
    def test_sim(self, input_params) -> None:
        work_dir = os.path.join("./tmp_sim_work_dir")
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)  # folders are created by failed tests
        os.makedirs(work_dir)

        dataroot_dir = os.path.join(work_dir, "sim_dataroot")
        datalist_file = os.path.join(work_dir, "sim_datalist.json")
        ConfigParser.export_config_file(sim_datalist, datalist_file)

        sim_dim = input_params["sim_dim"]
        create_sim_data(
            dataroot_dir, sim_datalist, sim_dim, rad_max=max(int(min(sim_dim) / 4), 1), rad_min=1, num_seg_classes=1
        )

        data_src_cfg = {"modality": input_params["modality"], "datalist": datalist_file, "dataroot": dataroot_dir}
        preds = auto_run(
            work_dir,
            data_src_cfg,
            ["dints", "segresnet", "segresnet2d", "swinunetr"],
            search=input_params["dints_search"],
        )
        self.assertTupleEqual(preds[0].shape, (2, sim_dim[0], sim_dim[1], sim_dim[2]))

        shutil.rmtree(work_dir)


if __name__ == "__main__":
    unittest.main()
