## Convolutional Neural Networks for Automatic Craniofacial Reconstruction

<img src="https://github.com/Jianningli/research-contributions/blob/master/SkullRec/figs/dataset.png" alt="dataset" width="600"/>

The MONAI codes are adapted from the [MONAI 3D Spleen segmentation example](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)


### Prepare the dataset

	
* Download the MUG500+ Dataset: [[Data Repo](https://figshare.com/articles/dataset/MUG500_Repository/9616319)], [[Descriptor](https://www.sciencedirect.com/science/article/pii/S2352340921008003)]


* Unzip and Extract the .NRRD files into one folder

``` Python
from pathlib import Path
import shutil
pathlist = Path('./9616319/0_labelsTr').glob('**/*.nrrd')
for path in pathlist:
     # because path is object not string
     path_in_str = str(path)
     shutil.copyfile(path_in_str, './complete_nrrds/'+path_in_str[-10:-5]+'.nrrd')
     print(path_in_str)
```

### Train a CNN using MONAI



#### alternatively, you can try out the pre-trained model


An Tensorflow equivalent implementation of an auto-encoder network for skull shape completion can be found at: [Github Repo](https://github.com/Jianningli/autoimplant), [Paper](https://link.springer.com/content/pdf/10.1007/978-3-030-60946-7.pdf#page=86)


```

```
