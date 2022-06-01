# DiNTS

## Overview

Repository for Paper "DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation" (CVPR'21)

## Installation

The code was tested with Anaconda and Python 3.7. After installing the Anaconda environment:

Clone the repo:

Install dependencies:
For PyTorch dependency, see pytorch.org for more details.
Install custom dependencies, horovod and openmpi e.t.c:

Use docker files:
Create a Dockerfile with
```
FROM nvcr.io/nvidian/pytorch:20.03-py3
RUN HOROVOD_NCCL_LINK=SHARED HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir --upgrade horovod
RUN apt install graphviz
RUN pip install graphviz
RUN pip install torchviz
RUN pip install tensorboardx
RUN pip install monai
```

Then
```
docker build -f Dockerfile -t nvcr.io/nvidian/pytorch:20.03-py3-horovod .
```

Run into Docker:
```
sudo docker run -it --gpus all --pid=host --shm-size 8G -v /home/yufan/Projects/:/workspace/Projects/ nvcr.io/nvidian/pytorch:20.03-py3-horovod
```

## Commands

All the scripts are in `./code`, `main.sh` will call other
scripts to perform search (`search.sh`), retrain (`retrain.sh`), test on validation results (`test_val.sh`), inference on test data (`test.sh`) and fuse (`fuse.sh`) all 5 fold results. Need to change
variable values within submit_all.sh to run your specific experiments.
The general pipeline is:

search the model on each task independently and results will be saved in
```
/workspace/Projects/darts-oneshot/search-bilevel-ent-mem${MEM}-${TASK}
```
retrain the model given `${TASK}`, `${MODEL}`, `${FOLD}`, results will be saved in the save folder as the searched model folder
Record the best retrained model manually and change the numbers (BESTTask01 e.t.c) in `main.sh`
Test on validation results on test results
The test results will be saved in each retraining folder
Fuse the five fold results and the results will be saved in designated folder
(Needs to be combined with step 6) The fused results are upsampled results, needs to resample to the original nii data size. I downloaded the fused results to my local workstation and resampled it (using fuse.py). The path to the original msd dataset needs to be provided.

The above operations are performed by change the value of N in `main.sh` and run
bash `main.sh`
Also you need to change the values of TASKS, FOLDS e.t.c to run specific experiments

### Argument Explanation
The arguments can also be found in config.py. Here are some quick references:

- iters is the total searching or retraining iteration. warmup_iters in searching is the iterations where architecture is not updated (the lr_scheduler is working, the temperature is not annealed).
- EF is the temperature annealing scheme. 0 is no annealing, 2 is linear decay
- EP=1, EF=0, topology=1 means no temperature annealing, use topology loss and entropy loss (not degrading binarized model validation result).
- base_lr * gpu number will be the final network weights lr
- opset is the operation set. opset 0, and setting cellop=4 will be the same with C2FNAS. opset1 includes 5 operations, 3D conv, 3 psudo 3d. Details can be found in operation.py
- path_iters is the number of iteration to print out architecture informations. Also in sampling based searching strategy, this is the training iteration for each sampled sub-model
- path_num is the number of pathes per block during sampling based training
- use_max will use one-shot supernet training, not sampling based method (should always add this one if training the whole supernet)
- there are several other arguments like finetune, retrain (deprecated), ef_end. those are related to binarize the model in searching stage and then finetune it. We are not using it since we double the channel number during retraining.

## Utilities

To further fuse the output segmetnation masks, user needs to create a `.json` file describing shape of each image for resizing/re-sampling. An example is shown in `./code/msd_size.json`.
```
{
    "Task07_Pancreas": {
        "pancreas_044.nii.gz": [
            512,
            512,
            87
        ],
        "pancreas_144.nii.gz": [
            512,
            512,
            48
        ],
        "pancreas_039.nii.gz": [
            512,
            512,
            87
        ]
    }
}
```

## Bibtex
```
@inproceedings{he2021dints,
  title={DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation},
  author={He, Yufan and Yang, Dong and Roth, Holger and Zhao, Can and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5841--5850},
  year={2021}
}
@inproceedings{yu2020c2fnas,
  title={C2fnas: Coarse-to-fine neural architecture search for 3d medical image segmentation},
  author={Yu, Qihang and Yang, Dong and Roth, Holger and Bai, Yutong and Zhang, Yixiao and Yuille, Alan L and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4126--4135},
  year={2020}
}
```
