## Convolutional Neural Networks for Automatic Craniofacial Reconstruction

<img src="https://github.com/Jianningli/research-contributions/blob/master/SkullRec/figs/dataset.png" alt="dataset" width="600"/>

The MONAI codes are adapted from the [MONAI 3D Spleen segmentation example](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)


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

The denoising codes come from [this repository](https://github.com/Jianningli/autoimplant/blob/master/src/pre_post_processing.py). 
The axial dimension of all the skull images are cropped to 256. If the axial dimension is smaller than 256, zero padding can be used.


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



### Train a CNN using MONAI

* Training Your MONAI Model 
 
```Python
python monaiSkull.py --train # Training
python monaiSkull.py --test # test, generate predictions (complete skulls) for test data

```



#### alternatively, you can try out the pre-trained model


An Tensorflow equivalent implementation of an auto-encoder network for skull shape completion can be found at: [Github Repo](https://github.com/Jianningli/autoimplant), [Paper](https://link.springer.com/content/pdf/10.1007/978-3-030-60946-7.pdf#page=86)



