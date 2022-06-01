1. Download the pretrained weights on the SkullFix dataset [here](https://files.icg.tugraz.at/f/d6b9f18c422948a8b0f1/?dl=1). <br>
Download the Facial Defects SkullFix dataset [here](https://files.icg.tugraz.at/f/5b7f31c4465b437e996d/?dl=1)

2. The following python snippet aligns the reconstruction results with the input using a similarity transformation

```Python
import ants
import os
from glob import glob


baseDir1 = './input_defective_skull/'
files1 = glob(baseDir1+'/*.nii.gz')


baseDir2 = './reconstruction_results/'
files2 = glob(baseDir2+'/*.nii.gz')


for i in range(len(files1)):
    fixed = ants.image_read(files1[i])
    moving = ants.image_read(files2[i])
    outs = ants.registration(fixed, moving, type_of_transforme = 'Similarity')
    warped_img = outs['warpedmovout']
    warped_to_moving = outs['invtransforms']
    NamePrefix = str(i).zfill(3)
    ants.image_write(warped_img,  './registered_reconstruction_results/'+ NamePrefix +'.nii.gz')


```


3. Facial reconstruction results on the SkullFix dataset. The first to the last column shows the axial view of the reconstruction (shown in brown) and input (shown in red) before and after alignment, the reconstructed face and input in 3D, respectively. The recovered facial area can be obtained via subtraction between the input and the aligned reconstruction results:
<img src="https://github.com/Jianningli/research-contributions/blob/master/SkullRec/figs/monai_results.png" alt="dataset" width="600"/>
