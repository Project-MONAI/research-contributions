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

import fire
import numpy as np
import os
import torch
import yaml

from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast


class DummyRunnerDiNTS(object):
    def __init__(self, output_path):
        config_file = []
        config_file.append(
            os.path.join(output_path, "configs", "hyper_parameters.yaml")
        )
        config_file.append(os.path.join(output_path, "configs", "network.yaml"))
        config_file.append(
            os.path.join(output_path, "configs", "transforms_train.yaml")
        )
        config_file.append(
            os.path.join(output_path, "configs", "transforms_validate.yaml")
        )

        parser = ConfigParser()
        parser.read_config(config_file)

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.input_channels = parser.get_parsed_content("training#input_channels")
        self.patch_size = parser.get_parsed_content("training#patch_size")
        self.patch_size_valid = parser.get_parsed_content("training#patch_size_valid")
        self.overlap_ratio = parser.get_parsed_content("training#overlap_ratio")

        output_classes = parser.get_parsed_content("training#output_classes")
        softmax = parser.get_parsed_content("training#softmax")
        self.label_channels = 1 if softmax else output_classes

        print("patch_size", self.patch_size)
        print("patch_size_valid", self.patch_size_valid)
        print("label_channels", self.label_channels)

        self.model = parser.get_parsed_content("training_network#network")
        self.model = self.model.to(self.device)

        self.loss_function = parser.get_parsed_content("training#loss")

        optimizer_part = parser.get_parsed_content("training#optimizer", instantiate=False)
        self.optimizer = optimizer_part.instantiate(params=self.model.parameters())

        train_transforms = parser.get_parsed_content("transforms_train")

        data_stats_file = os.path.join(output_path, "..", "datastats.yaml")
        with open(data_stats_file) as f_data_stat:
            data_stat = yaml.full_load(f_data_stat)

        pixdim = parser.get_parsed_content("transforms_train#transforms#3#pixdim")
        pixdim = [np.abs(pixdim[_i]) for _i in range(3)]

        self.max_shape = [-1, -1, -1]
        for _k in range(len(data_stat["stats_by_cases"])):
            image_shape = data_stat["stats_by_cases"][_k]["image_stats"]["shape"]
            image_shape = np.squeeze(image_shape)
            image_spacing = data_stat["stats_by_cases"][_k]["image_stats"]["spacing"]
            image_spacing = np.squeeze(image_spacing)
            image_spacing = [np.abs(image_spacing[_i]) for _i in range(3)]

            for _l in range(3):
                self.max_shape[_l] = max(
                    self.max_shape[_l],
                    int(np.ceil(float(image_shape[_l]) * image_spacing[_l] / pixdim[_l])),
                )
        print("max_shape", self.max_shape)

    def run(
        self,
        num_images_per_batch,
        num_sw_batch_size,
        validation_data_device,
    ):
        scaler = GradScaler()

        num_epochs = 2
        num_iterations = 6
        num_iterations_validation = 1
        num_patches_per_image = 1

        validation_data_device = validation_data_device.lower()
        if validation_data_device != "cpu" and validation_data_device != "gpu":
            raise ValueError("only cpu or gpu allowed for validation_data_device!")

        print("num_images_per_batch", num_images_per_batch)
        print("num_sw_batch_size", num_sw_batch_size)
        print("validation_data_device", validation_data_device)

        for _i in range(num_epochs):
            # training
            print("------  training  ------")

            self.model.train()

            for _j in range(num_iterations):
                print("iteration", _j + 1)

                inputs = torch.rand(
                    (
                        num_images_per_batch * num_patches_per_image,
                        self.input_channels,
                        self.patch_size[0],
                        self.patch_size[1],
                        self.patch_size[2],
                    )
                )
                labels = torch.rand(
                    (
                        num_images_per_batch * num_patches_per_image,
                        self.label_channels,
                        self.patch_size[0],
                        self.patch_size[1],
                        self.patch_size[2],
                    )
                )
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for param in self.model.parameters():
                    param.grad = None

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs.float(), labels)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                scaler.step(self.optimizer)
                scaler.update()

            # validation
            print("------  validation  ------")
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for _k in range(num_iterations_validation):
                    print("validation iteration", _k + 1)

                    val_images = torch.rand(
                        (1, self.input_channels, self.max_shape[0], self.max_shape[1], self.max_shape[2])
                    )

                    if validation_data_device == "gpu":
                        val_images = val_images.to(self.device)

                    with autocast():
                        val_outputs = sliding_window_inference(
                            val_images,
                            self.patch_size_valid,
                            num_sw_batch_size,
                            self.model,
                            mode="gaussian",
                            overlap=self.overlap_ratio,
                            sw_device=self.device,
                        )

            torch.cuda.empty_cache()

        return


if __name__ == '__main__':
    fire.Fire(DummyRunnerDiNTS)
