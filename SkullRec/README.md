## Convolutional Neural Networks for Automatic Craniofacial Reconstruction

<img src="https://github.com/Jianningli/research-contributions/blob/master/SkullRec/figs/dataset.png" alt="dataset" width="600"/>


### Prepare the dataset

	
* Download the MUG500+ Dataset: [[Data Repo](https://figshare.com/articles/dataset/MUG500_Repository/9616319)], [[Descriptor](https://www.sciencedirect.com/science/article/pii/S2352340921008003)]


* Unzip and Extract the .NRRDs into one folder

``` Python
from pathlib import Path
import shutil
pathlist = Path('./9616319').glob('**/*.nrrd')
for path in pathlist:
     path_in_str = str(path)
     shutil.copyfile(path_in_str, './complete_nrrds/'+path_in_str[-10:-5]+'.nrrd')
     print(path_in_str)
     
```

* Denoise and Crop the Skulls

The denoising codes come from [this repository](https://github.com/Jianningli/autoimplant/blob/master/src/pre_post_processing.py). <br>
The axial dimension of all the skull images are cropped to 256. <br>
If the axial dimension is smaller than 256, zero padding can be used.


* Create Facial and Cranial Defects on the Skulls 
``` Python
facialDefects.py  #create defects around the face 
cranialDefects.py  #create defects around the cranium
```
* Convert NRRDs to Nifty (for MONAI Dataset loader)

``` Python
#codes attributes to the stack overflow anser: 
#https://stackoverflow.com/questions/47761353/nrrd-to-nifti-file-conversion
import vtk

def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info

def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


baseDir = './complete_nrrds/'
files = glob(baseDir+'/*.nrrd')
print(files)
for file in files:
  m, info = readnrrd(file)
  fname=baseDir+'nifty/'+file[-10:-5]+ '.nii.gz'
  writenifti(m,fname,info)
```

 * Split Training and Test Set
 Except the 21 corrupted files in MUG500+ dataset

* Alternatively, you can directly download the training-ready datasets from [here]()



### Train a CNN using MONAI for Skull Reconstruction

The MONAI codes are adapted from the [MONAI 3D Spleen segmentation example](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)

* Training Your MONAI Model 
 
```Python
python monaiSkull.py --train # Training
python monaiSkull.py --test # test, generate predictions (complete skulls) for test data

```

* Alternatively, you can try out the pre-trained model 
1. Clone this repository
2. Download the pre-processed dataset
3. Move the dataset folders into the current directory of the repository
4. Evaluate on the test set (or your own skull data pre-processed the same way as the dataset)


An Tensorflow equivalent implementation of an auto-encoder network for skull shape completion can be found at: [Github Repo](https://github.com/Jianningli/autoimplant), [Paper](https://link.springer.com/content/pdf/10.1007/978-3-030-60946-7.pdf#page=86)

### Reference
If you use the dataset and/or the pre-trained model in your research, please consider citing the following:



```
@article{li2021mug500+,
  title={MUG500+: Database of 500 high-resolution healthy human skulls and 29 craniotomy skulls and implants},
  author={Li, Jianning and Krall, Marcell and Trummer, Florian and Memon, Afaque Rafique and Pepe, Antonio and Gsaxner, Christina and Jin, Yuan and Chen, Xiaojun and Deutschmann, Hannes and Zefferer, Ulrike and others},
  journal={Data in Brief},
  volume={39},
  pages={107524},
  year={2021},
  publisher={Elsevier}
}

```
and,

```
@incollection{li2020baseline,
  title={A baseline approach for AutoImplant: the MICCAI 2020 cranial implant design challenge},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Campe, Gord von and Egger, Jan},
  booktitle={Multimodal Learning for Clinical Decision Support and Clinical Image-Based Procedures},
  pages={75--84},
  year={2020},
  publisher={Springer}
}
```


