## Examplary Skull Reconstruction Results Using the Pre-trained Model

Download the pre-trained weights [here]( https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/skull_rec_best_metric_model.pth). <br>
The (auto-encoder) network was trained on 429-79=350 (79 images for validation) skull images from the [pre-processed dataset](https://files.icg.tugraz.at/f/9642058af1744b4b961b/?dl=1).

### cranial reconstruction (input-pred-gt)
Cranial reconstruction is relatively easy, since the cranium contains no subtle and complex structures.

<img src="https://github.com/Project-MONAI/research-contributions/blob/main/SkullRec/figs/cranial_rec.png" alt="dataset" width="400"/>

### facial reconstruction (input-pred-gt)

It is obvious from the results that the subtle facial structures are difficult to be reconstructed and recovered. <br>
The pre-trained model serve only as a baseline. <br>
 <br>
<img src="https://github.com/Project-MONAI/research-contributions/blob/main/SkullRec/figs/facial_rec.png" alt="dataset" width="400"/>
