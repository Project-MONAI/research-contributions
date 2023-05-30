# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from skimage.transform import resize

from monai.metrics.utils import do_metric_reduction, ignore_background


def check_number(a):
    try:
        a = float(a)
        if np.abs(a) < np.finfo(np.float32).eps or int(a) / a == 1:
            # print("This is Integer")
            return int(a)
        else:
            # print("This is Float")
            return float(a)
    except ValueError:
        # print("This value is String")
        if a.lower() == "true":
            return True
        elif a.lower() == "false":
            return False
        elif a.lower() == "none":
            return None

        return str(a)


def check_list_tuple(a):
    if not isinstance(a, str):
        return a

    a = a.replace(" ", "")
    if a[0] == "(" and a[-1] == ")":
        part_split = a[1:-1].split(",")
        out = []
        for _s in range(len(part_split)):
            out.append(check_number(part_split[_s]))
        out = tuple(_i for _i in out)
        return out
    elif a[0] == "[" and a[-1] == "]":
        part_split = a[1:-1].split(",")
        out = []
        for _s in range(len(part_split)):
            out.append(check_number(part_split[_s]))
        return out

    return a


def parse_monai_specs(component_string):
    string_parts = component_string.split("|")
    component_name = string_parts[0]

    component_dict = {}
    for _k in range(1, len(string_parts)):
        part = string_parts[_k]
        part_split = part.split("~")
        _key = part_split[0]
        _val = part_split[1]

        _val_parts = _val.split(",")
        if len(_val_parts) == 1:
            component_dict[_key] = check_number(_val)
        else:
            component_dict[_key] = [check_number(_item) for _item in _val_parts]

    return component_name, component_dict


def keep_largest_cc(nda):
    labels = measure.label(nda > 0)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        largestCC = largestCC.astype(nda.dtype)
        return largestCC

    return nda


def resize_volume(nda, output_shape, order=1, preserve_range=True, anti_aliasing=False):
    return resize(nda, output_shape, order=order, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
