# Installation

We provide installation instructions here.

## Setup

### Using Docker

The simplest way to use SwinMM is to use our docker image [`swinmm`](https://drive.google.com/file/d/1EGSoqN-HphyMV_gKUq-g7_BSwTTg35oA/view?usp=sharing), which has contained all the needed dependencies. Download the `swinmm.tar` into the `SwinMM` directory and try the following scripts:

```bash
cd SwinMM
docker import - swinmm < swinmm.tar
docker run --runtime=nvidia --gpus=all -m="800g" --shm-size="32g" -itd -v ./:/volume swinmm /bin/bash
docker exec -it swinmm /bin/bash
conda activate SwinMM
```

To use docker, make sure you have installed `docker` and `nvidia-docker`.

### Manual

For fast dataset loading, we required the users to install the Redis database, for example, on Ubuntu: `sudo apt-get install redis`

We also recommend the users install the PyTorch-based version from the official website.

Two packages are recommended to install manually according to their complicated dependencies: [bagua==0.9.2](https://github.com/BaguaSys/bagua), [monai==0.9.0](https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies)

The others can be installed through `pip install -r requirements.txt`

## Datasets

Our pre-training dataset includes 5833 volumes from 8 public datasets:

- [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- [MSD](http://medicaldecathlon.com/)
- [TCIACovid19](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19/)
- [WORD](https://github.com/HiLab-git/WORD)
- [TCIA-Colon](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY/)
- [LiDC](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI/)
- [HNSCC](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)

We choose two popular datasets to test the downstream segmentation performance:

- [WORD](https://github.com/HiLab-git/WORD) (The Whole abdominal Organ Dataset)
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/#challenge/584e75606a3c77492fe91bba) (Automated Cardiac Diagnosis Challenge)

The json files can be downloaded from [pretrain_jsons](https://drive.google.com/file/d/1gJThxBvnJnc2_N1nFX7xywjFWFw7DSEY/view?usp=sharing) and [word_jsons](https://drive.google.com/file/d/1Td4T_k2QlEcTETz9TERGsVdOyebD5ULv/view?usp=sharing);

The dataset is organized as below:

```text
SwinMM
├── WORD
│   └── dataset
│       └── dataset12_WORD
│           ├── imagesTr
│           ├── imagesTs
│           ├── imagesVal
│           ├── labelsTr
│           ├── labelsTs
│           ├── labelsVal
│           └── dataset12_WORD.json
└── Pretrain
    ├── dataset
    │   ├── dataset00_BTCV
    │   ├── dataset02_Heart
    │   ├── dataset03_Liver
    │   ├── dataset04_Hippocampus
    │   ├── dataset06_Lung
    │   ├── dataset07_Pancreas
    │   ├── dataset08_HepaticVessel
    │   ├── dataset09_Spleen
    │   ├── dataset10_Colon
    │   ├── dataset11_TCIAcovid19
    │   ├── dataset12_WORD
    │   ├── dataset13_AbdomenCT-1K
    │   ├── dataset_HNSCC
    │   ├── dataset_TCIAcolon
    │   └── dataset_LIDC
    └── jsons
        ├── dataset00_BTCV.json
        ├── dataset01_BrainTumour.json
        ├── dataset02_Heart.json
        ├── dataset03_Liver.json
        ├── dataset04_Hippocampus.json
        ├── dataset05_Prostate.json
        ├── dataset06_Lung.json
        ├── dataset07_Pancreas.json
        ├── dataset08_HepaticVessel.json
        ├── dataset09_Spleen.json
        ├── dataset10_Colon.json
        ├── dataset11_TCIAcovid19.json
        ├── dataset12_WORD.json
        ├── dataset13_AbdomenCT-1K.json
        ├── dataset_HNSCC.json
        ├── dataset_TCIAcolon.json
        └── dataset_LIDC.json

```
