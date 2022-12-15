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
from monai.inferers import SlidingWindowInferer
from monai.losses import DeepSupervisionLoss
from torch.cuda.amp import GradScaler, autocast


class DummyRunnerSegResNet(object):
    def __init__(self, output_path, data_stats_file):
        config_file = []
        config_file.append(
            os.path.join(output_path, "configs", "hyper_parameters.yaml")
        )

        parser = ConfigParser()
        parser.read_config(config_file)

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.amp = parser.get_parsed_content("amp")
        self.input_channels = parser.get_parsed_content("input_channels")
        self.roi_size = parser.get_parsed_content("roi_size")
        self.overlap_ratio = 0.625
        self.num_sw_batch_size = 1

        output_classes = parser.get_parsed_content("output_classes")
        sigmoid = parser.get_parsed_content("sigmoid")
        self.label_channels = self.output_classes if sigmoid else 1

        print("roi_size", self.roi_size)
        print("label_channels", self.label_channels)

        self.model = parser.get_parsed_content("network")
        self.model = self.model.to(self.device)

        self.loss_function = parser.get_parsed_content("loss")
        self.loss_function = DeepSupervisionLoss(self.loss_function)

        optimizer_part = parser.get_parsed_content("optimizer", instantiate=False)
        self.optimizer = optimizer_part.instantiate(params=self.model.parameters())

        with open(data_stats_file) as f_data_stat:
            data_stat = yaml.full_load(f_data_stat)

        self.max_shape = [0, 0, 0]
        if parser.get_parsed_content("resample"):
            pixdim = parser.get_parsed_content("resample_resolution")

            for _k in range(len(data_stat["stats_by_cases"])):
                image_shape = data_stat["stats_by_cases"][_k]["image_stats"]["shape"]
                image_shape = np.squeeze(image_shape)
                image_spacing = data_stat["stats_by_cases"][_k]["image_stats"][
                    "spacing"
                ]
                image_spacing = np.squeeze(image_spacing)
                image_spacing = [np.abs(image_spacing[_i]) for _i in range(3)]

                new_shape = [
                    int(
                        np.ceil(float(image_shape[_l]) * image_spacing[_l] / pixdim[_l])
                    )
                    for _l in range(3)
                ]
                if np.prod(new_shape) > np.prod(self.max_shape):
                    self.max_shape = new_shape
        else:
            for _k in range(len(data_stat["stats_by_cases"])):
                image_shape = data_stat["stats_by_cases"][_k]["image_stats"]["shape"]
                image_shape = np.squeeze(image_shape)

                if np.prod(image_shape) > np.prod(self.max_shape):
                    self.max_shape = image_shape
        print("max_shape", self.max_shape)

        self.sliding_inferrer = SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=1,
            overlap=0.625,
            mode="gaussian",
            cache_roi_weight_map=True,
            progress=False,
            cpu_thresh=512 ** 3 // output_classes,
        )

    def run(
        self,
        num_images_per_batch,
    ):
        scaler = GradScaler()

        num_epochs = 2
        num_iterations = 6
        num_iterations_validation = 1

        print("num_images_per_batch", num_images_per_batch)

        for _i in range(num_epochs):
            # training
            print("------  training  ------")

            self.model.train()

            for _j in range(num_iterations):
                print("iteration", _j + 1)

                inputs = torch.rand(
                    (
                        num_images_per_batch,
                        self.input_channels,
                        self.roi_size[0],
                        self.roi_size[1],
                        self.roi_size[2],
                    )
                )
                labels = torch.rand(
                    (
                        num_images_per_batch,
                        self.label_channels,
                        self.roi_size[0],
                        self.roi_size[1],
                        self.roi_size[2],
                    )
                )
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for param in self.model.parameters():
                    param.grad = None

                with autocast(self.amp):
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()

            # validation
            print("------  validation  ------")
            self.model.eval()
            with torch.no_grad():
                for _k in range(num_iterations_validation):
                    print("validation iteration", _k + 1)

                    val_images = torch.rand(
                        (
                            1,
                            self.input_channels,
                            self.max_shape[0],
                            self.max_shape[1],
                            self.max_shape[2],
                        )
                    )
                    val_images = val_images.as_subclass(torch.Tensor).to(self.device)

                    with autocast(self.amp):
                        val_outputs = self.sliding_inferrer(
                            inputs=val_images, network=self.model
                        )

        return


if __name__ == "__main__":
    fire.Fire(DummyRunnerSegResNet)
