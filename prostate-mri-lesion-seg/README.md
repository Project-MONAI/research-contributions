# Prostate MRI Lesion Segmentation - MONAI Deploy MAP

<p float="left">
  <img src="imgs/organ_seg.png" width="200" />
  <img src="imgs/lesion_prob.png" width="200" />
  <img src="imgs/lesion_mask.png" width="200" />
</p>

This workflow takes T2, ADC, and HighB MRI series as input and produces several NIfTI files as output. These outputs contain organ and lesion segmentations and lesion probability maps.

## Software and Setup

In order to run this workflow and build a MAP you will need to [install MONAI Deploy App SDK](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/installing_app_sdk.html). The workflow is currently verified against version 0.5.1. On most systems this can be easily done with the command `pip install monai-deploy-app-sdk==0.5.1`.

It is also recommended to have an NVIDIA GPU with at least 12 GB of memory available.

## Using this Repository

The easiest way to get started with this workflow is to run on a test image taken from the [ProstateX](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656) dataset. The `tutorial.ipynb` file walks through this process using ProstateX-0004 which can be [downloaded separately](https://drive.google.com/drive/folders/17oOgs1jIgJUxtkBRLATUdNrLSNQCLpCn?usp=drive_link) and placed in a `test-data/` directory.

After validating on test data, you can test this image on your own study or dataset. One of the main considerations when adapting to a new dataset will be the making sure the [DICOM Series Selector Operator](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/modules/_autosummary/monai.deploy.operators.DICOMSeriesSelectorOperator.html#monai.deploy.operators.DICOMSeriesSelectorOperator) is configured to properly differentiate between the different naming schemes and properties of the new dataset. 

If all three (T2, ADC, HighB) series are not detected properly in the study, the pipeline will not complete. If any of these modalities are incorrectly routed, the pipeline results will not be accurate. The workflow currently saves intermediate copies (in NIfTI) of these series in the output folder so it is possible to verify they were picked up (and preprocessed) correctly.

The current set of rules in `app.py` filter based on SeriesDescription, ImageType, etc., and work with ProstateX. Please refer to MONAI documentation for guidance on modifying these rules for custom filtering.

There are also scripts in the `scripts/` directory that test the workflow locally (i.e., not containerized) and test the workflow build with a MAP. These should help test incremental changes to the code or series selection rules.

## Models

The models needed to build and execute the pipeline (1 organ segmentation model, 5 lesion segmentation models) are hosted separately on [Google Drive here](https://drive.google.com/drive/folders/1wO4h5AON0MA3dxwnzl9cJlxjPxsfXcCF?usp=sharing).

Download these models and put them inside a folder named `prostate_mri_lesion_seg_app/models` alongside the rest of the application code. Pipeline creation and execution will not complete if the model file path is changed or renamed.

## License

This work was developed by NVIDIA and the NIH National Cancer Institute (NCI). Please refer to the LICENSE for terms of use.