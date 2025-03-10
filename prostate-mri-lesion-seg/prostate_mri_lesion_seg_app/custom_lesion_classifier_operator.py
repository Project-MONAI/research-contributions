'''
Prostate-MRI_Lesion_Detection, v3.0 (Release date: September 17, 2024)
DEFINITIONS: AUTHOR(S) NVIDIA Corp. and National Cancer Institute, NIH

PROVIDER: the National Cancer Institute (NCI), a participating institute of the
National Institutes of Health (NIH), and an agency of the United States Government.

SOFTWARE: the machine readable, binary, object code form,
and the related documentation for the modules of the Prostate-MRI_Lesion_Detection, v2.0
software package, which is a collection of operators which accept (T2, ADC, and High
b-value DICOM images) and produce prostate organ and lesion segmentation files

RECIPIENT: the party that downloads the software.

By downloading or otherwise receiving the SOFTWARE, RECIPIENT may
use and/or redistribute the SOFTWARE, with or without modification,
subject to RECIPIENT’s agreement to the following terms:

1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS
OF HUMAN SUBJECTS.  RECIPIENT is responsible for
compliance with all laws and regulations applicable to the use
of the SOFTWARE.

2. THE SOFTWARE is distributed for NON-COMMERCIAL RESEARCH PURPOSES ONLY. RECIPIENT is
responsible for appropriate-use compliance.

3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and
the name of the author of the SOFTWARE in all written publications
containing any data or information regarding or resulting from use
of the SOFTWARE.

4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

5.	RECIPIENT agrees not to use any trademarks, service marks, trade names,
logos or product names of NVIDIA, NCI or NIH to endorse or promote products derived
from the SOFTWARE without specific, prior and written permission.

6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its
own copyright statement to its modifications or derivative works of the SOFTWARE
and may provide additional or different license terms and conditions in its
sublicenses of modifications or derivative works of the SOFTWARE provided that
RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies
with the conditions stated in this Agreement. Whenever Recipient distributes or
redistributes the SOFTWARE, a copy of this Agreement must be included with
each copy of the SOFTWARE.'''

import logging
from pathlib import Path
import yaml
import copy

# MONAI Deploy App SDK imports
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, InputContext, Operator, OutputContext
from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader

# MONAI imports
from monai.data import MetaTensor
from monai.transforms import (
    ResampleToMatch,
    Spacing
)

# AI/CV imports
import numpy as np
from skimage.measure import label, regionprops
import nibabel as nib
import torch

# Local imports
from resnet import ResNet, BasicBlock
from common import standard_normalization_multi_channel
from common import crop_pos_classification_multi_channel_3d

###############################################################################
class ProstateLesionClassifierOperator(Operator):
    """Performs Prostate Lesion segmentation with a 3D image converted from a mp-DICOM MRI series."""

    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        model_path: Path,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        **kwargs,
    ):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

        self.model_path = model_path
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.app_context = app_context
        self.input_name_image_t2 = "image_t2"
        self.input_name_image_adc = "image_adc"
        self.input_name_image_highb = "image_highb"
        self.input_name_image_organ_seg = "image_organ_seg"
        self.input_name_image_lesion_seg = "image_lesion_seg"
        self.output_name_saved_images_folder = ""

        # The base class has an attribute called fragment to hold the reference to the fragment object
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image_t2)
        spec.input(self.input_name_image_adc)
        spec.input(self.input_name_image_highb)
        spec.input(self.input_name_image_organ_seg)
        spec.input(self.input_name_image_lesion_seg)
        spec.output(self.output_name_saved_images_folder).condition(
            ConditionType.NONE
        )  # Output not requiring a receiver

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image_t2 = op_input.receive(self.input_name_image_t2)
        if not input_image_t2:
            raise ValueError("Input image (T2) is not found.")
        input_image_adc = op_input.receive(self.input_name_image_adc)
        if not input_image_adc:
            raise ValueError("Input image (ADC) is not found.")
        input_image_highb = op_input.receive(self.input_name_image_highb)
        if not input_image_highb:
            raise ValueError("Input image (High b-value) is not found.")
        image_organ_seg = op_input.receive(self.input_name_image_organ_seg)
        if not image_organ_seg:
            raise ValueError("Input image (Organ segmentation) is not found.")
        image_lesion_seg = op_input.receive(self.input_name_image_lesion_seg)
        if not image_lesion_seg:
            raise ValueError("Input image (Lesion segmentation) is not found.")

        print("\nBeginning lesion classification...")

        # Instantiate network and send to GPU
        net = ResNet(block=BasicBlock,
                            layers=[1,1,1,1],
                            block_inplanes=[32,64,128,256],
                            n_classes=4)
        if torch.cuda.is_available():
            net = net.to("cuda")
        net.eval()

        # Load model weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(self.model_path / "classifier/model_best.pth.tar", map_location=device, weights_only=False)
        net.load_state_dict(checkpoint["state_dict"])

        # Load images and preprocess
        data = self.preprocess(input_image_t2, input_image_adc, input_image_highb, image_organ_seg, image_lesion_seg)
        if torch.cuda.is_available():
            inputs = data["image"].to("cuda")
        else:
            inputs = data["image"]
        inputs_shape = ( inputs.size()[-3], inputs.size()[-2], inputs.size()[-1])
        print("Inputs shape: ", inputs_shape)

        # Create affine transformation
        affine = data["affine"]
        affine = affine.detach().numpy()
        affine = np.squeeze(affine)
        codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(np.linalg.inv(affine)))

        # Read shape
        nda_shape = data["nda_shape"]
        nda_shape = nda_shape.detach().numpy()
        nda_shape = np.squeeze(nda_shape)

        # Prepare cropped patches
        nda_resize_crops, _ = crop_pos_classification_multi_channel_3d(data["image"], data["nda_pred_resize"], crop_size=[64, 64, 64])
        print("nda_resize_crops shape: ", nda_resize_crops.shape)
        nda_resize_crops = np.expand_dims(nda_resize_crops, axis=0)
        if torch.cuda.is_available():
            nda_resize_crops = torch.tensor(nda_resize_crops).to("cuda")
        else:
            nda_resize_crops = torch.tensor(nda_resize_crops)

        # Run inference
        inputs = nda_resize_crops
        if len(list(inputs.size())) == 6:
            inputs = torch.squeeze(inputs, dim=0)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy().squeeze()

         # Write prostate organ information
        nda_wp = data["pred_wp"]
        nda_wp = nda_wp.squeeze()
        nda_regions = label(nda_wp)
        regions = np.unique(nda_regions)
        nda_region = (nda_regions==regions[0]).astype(np.uint8)
        props = regionprops(nda_region)

        info = {}
        info_file = []
        info["Organ"] = "Prostate"
        info["Major_Axis_Length"] = props[0].major_axis_length * 0.5
        info["Volume"] = float(np.sum(nda_wp.astype(np.uint16))) * 0.5 * 0.5 * 0.5
        print(info)
        info_file.append(info)

        # Create lesion output scores
        nda_regions = label(data["nda_pred_resize"])
        regions = np.unique(nda_regions)
        for _i in range(len(regions)):
            if regions[_i] == 0:
                continue
            nda_region = (nda_regions==regions[_i]).astype(np.uint8)

            predicted_value = int(predicted) if predicted.ndim == 0 else int(predicted[_i-1])
            print("predicted_value:", predicted_value)

            props = regionprops(nda_region)
            if predicted_value == 2 and props[0].major_axis_length > 40:
                predicted_value = 3

            info = {}
            info["Lesion_ID"] = _i
            info["Major_Axis_Length"] = props[0].major_axis_length * 0.5
            info["Volume"] = float(np.sum(nda_region.astype(np.uint16))) * 0.5 * 0.5 * 0.5
            info["PI_RADS"] = predicted_value + 2
            print(info)
            info_file.append(info)

        with open(self.output_folder / "lesions.txt" , "w") as out_file:
            _ = yaml.dump(info_file, stream=out_file)

        # Now emit data to the output ports of this operator
        op_output.emit(self.output_folder, self.output_name_saved_images_folder)

    def preprocess(self, image_t2, image_adc, image_highb, image_organ_seg, image_lesion_seg):
        """Composes transforms for preprocessing input before predicting on a model."""

        affine_orig, nda = [], []

        # Load images and create Metatensors
        print("Loading images...")
        t2, t2_metadata = InMemImageReader(image_t2).get_data(image_t2)
        adc, adc_metadata = InMemImageReader(image_adc).get_data(image_adc)
        highb, highb_metadata = InMemImageReader(image_highb).get_data(image_highb)
        organ, organ_metadata = InMemImageReader(image_organ_seg).get_data(image_organ_seg)
        lesion, lesion_metadata = InMemImageReader(image_lesion_seg).get_data(image_lesion_seg)
        t2_metatensor = MetaTensor(t2[None], meta=t2_metadata)
        adc_metatensor = MetaTensor(adc[None], meta=adc_metadata)
        highb_metatensor = MetaTensor(highb[None], meta=highb_metadata)
        organ_metatensor = MetaTensor(organ[None], meta=organ_metadata)
        lesion_metatensor = MetaTensor(lesion[None], meta=lesion_metadata)
        affine_orig = torch.tensor(t2_metadata["nifti_affine_transform"])

        # Resample images to match T2
        print("Resampling ADC/HIGHB to match T2...")
        adc_metatensor = ResampleToMatch()(adc_metatensor, t2_metatensor)  # NOTE: metadata may be incorrect
        highb_metatensor = ResampleToMatch()(highb_metatensor, t2_metatensor)  # NOTE: metadata may be incorrect
        nda_shape = torch.tensor([t2_metatensor.array.shape[1], t2_metatensor.array.shape[2], t2_metatensor.array.shape[3]])

        # Resample to isotropic spacing
        print("Resampling all channels to (0.5, 0.5, 0.5)...")
        t2_resampled = Spacing(pixdim=(0.5, 0.5, 0.5), mode=("bilinear"))(t2_metatensor)
        adc_resampled = Spacing(pixdim=(0.5, 0.5, 0.5), mode=("bilinear"))(adc_metatensor)
        highb_resampled = Spacing(pixdim=(0.5, 0.5, 0.5), mode=("bilinear"))(highb_metatensor)
        organ_resampled = Spacing(pixdim=(0.5, 0.5, 0.5), mode=("nearest"))(organ_metatensor)
        lesion_resampled = Spacing(pixdim=(0.5, 0.5, 0.5), mode=("nearest"))(lesion_metatensor)
        nda_resize_shape = torch.tensor([t2_resampled.array.shape[1], t2_resampled.array.shape[2], t2_resampled.array.shape[3]])
        lesion_resampled = np.squeeze(lesion_resampled, axis=0)

        # Combine volumes into 4D tensor
        combined_volume = np.concatenate([t2_resampled, adc_resampled, highb_resampled, organ_resampled], axis=0)
        combined_volume = MetaTensor(combined_volume, meta=t2_metadata)
        print(f"Combined volume shape: {combined_volume.shape}")

        # Normalize
        nda_resize = standard_normalization_multi_channel(combined_volume[0:3])

        # Create rest of the sample dictionary
        nda_wp = organ
        sample = {
            "affine": affine_orig,
            "image": nda_resize,
            "nda_pred_resize": lesion_resampled,
            "nda_shape": nda_shape,
            "nda_resize_shape": nda_resize_shape,
            "pred_wp": nda_wp,
        }

        return sample
