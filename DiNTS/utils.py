#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics.utils import do_metric_reduction, ignore_background
from skimage import measure
from skimage.transform import resize


def parse_network_specs(network_specs_str):

    network_specs = []
    network_specs_dict = {}

    num_layers = len(network_specs_str)
    for _i in range(num_layers):
        layer_specs_str = network_specs_str[_i]
        layer_specs_str_parts = layer_specs_str.split("_")

        id = layer_specs_str_parts[0].replace("id", "")
        level = int(layer_specs_str_parts[1].replace("lv", ""))
        prev_str_parts = layer_specs_str_parts[3].split(",")
        prev = []
        for _j in range(len(prev_str_parts)):
            if prev_str_parts[_j].lower() == "none":
                prev.append(None)
            else:
                prev.append(int(prev_str_parts[_j]))
        prev = tuple(prev)
        out = True if layer_specs_str_parts[4].lower() == "true" else False

        layer_specs = (level, layer_specs_str_parts[2], prev, out)
        network_specs_dict[id] = layer_specs

    for _i in range(num_layers):
        network_specs.append(network_specs_dict[str(_i)])

    return network_specs


def check_number(a):
    try:
        a = float(a)
        if np.abs(a) < np.finfo(np.float32).eps or int(a)/a == 1:
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


# def parse_monai_network_specs(network_string):
#     string_parts = network_string.split("|")
#     network_name = string_parts[0]

#     network_dict = {}        
#     for _k in range(1, len(string_parts)):
#         part = string_parts[_k]
#         part_split = part.split("~")
#         _key = part_split[0]
#         _val = part_split[1]

#         _val_parts = _val.split(",")
#         if len(_val_parts) == 1:
#             network_dict[_key] = check_number(_val)
#         else:
#             network_dict[_key] = [check_number(_item) for _item in _val_parts]

#     return network_name, network_dict


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


# def parse_monai_transform_specs(component_string):
#     string_parts = component_string.split("|")
#     component_name = string_parts[0]

#     component_dict = {}        
#     for _k in range(1, len(string_parts)):
#         part = string_parts[_k]
#         part_split = part.split("~")
#         _key = part_split[0]
#         _val = part_split[1]

#         _val1 = check_number(_val)
#         _val1 = check_list_tuple(_val1)
#         component_dict[_key] = _val1
#         # _val_parts = _val.split(",")
#         # if len(_val_parts) == 1:
#         #     _val1 = check_number(_val)
#         #     _val1 = check_list_tuple(_val1)
#         #     component_dict[_key] = _val1
#         # else:
#         #     component_dict[_key] = [check_number(_item) for _item in _val_parts]

#     return component_name, component_dict


def custom_compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """

    if not include_background:
        y_pred, y = ignore_background(
            y_pred=y_pred,
            y=y,
        )

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    f = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_o.device))
    return f  # returns array of Dice with shape: [batch, n_classes]


def keep_largest_cc(nda):

    labels = measure.label(nda>0)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        largestCC = largestCC.astype(nda.dtype)
        return largestCC

    return nda


def resize_volume(nda, output_shape, order=1, preserve_range=True, anti_aliasing=False):
    return resize(nda, output_shape, order=order, preserve_range=preserve_range, anti_aliasing=anti_aliasing)