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

import os
import copy
import logging
import numpy as np

import logging
from pathlib import Path

# MONAI Deploy App SDK imports
from monai.deploy.core import ExecutionContext, Image, InputContext, Operator, OutputContext
from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec

# MONAI imports
from monai.data import MetaTensor
from monai.transforms import SaveImage

# AI/CV imports
import SimpleITK as sitk
from skimage.transform import resize
import nibabel as nib
import torch
from torch.utils.data import Dataset

# Local imports
from rrunet3D import RRUNet3D
from common import standard_normalization_multi_channel

def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

class SegmentationDataset(Dataset):
    def __init__(self, output_path, data_purpose):
        self.data_purpose = data_purpose
        self.output_path = output_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Preprocesses input data for model prediction"""
        # Load T2 as reference image
        t2_path = f"{self.output_path}/t2/t2.nii.gz"
        t2_sitk = sitk.ReadImage(t2_path)
        t2_nib = nib.load(t2_path)
        affine_orig = t2_nib.affine
        spacing_orig = t2_nib.header.get_zooms()

        # Initialize data arrays
        nda = []

        # Load T2 data
        t2_canonical = nib.as_closest_canonical(t2_nib)
        nda.append(t2_canonical.get_fdata())

        # Process ADC and HighB images
        for modality in ['adc', 'highb']:
            img_path = f"{self.output_path}/{modality}/{modality}.nii.gz"
            img_sitk = sitk.ReadImage(img_path)

            # Resample to match T2 dimensions
            img_resampled = sitk.Resample(
                img_sitk, t2_sitk.GetSize(),
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                t2_sitk.GetOrigin(),
                t2_sitk.GetSpacing(),
                t2_sitk.GetDirection(),
                0,
                t2_sitk.GetPixelID()
            )
            sitk.WriteImage(img_resampled, img_path)

            # Load resampled data
            img_nib = nib.as_closest_canonical(nib.load(img_path))
            nda.append(img_nib.get_fdata())

        # Stack input modalities
        nda = np.stack(nda, axis=0).astype(np.float32)
        nda_shape = nda.shape[1:]

        # Load prostate segmentation
        wp_path = f"{self.output_path}/organ/organ.nii.gz"
        wp_nib = nib.as_closest_canonical(nib.load(wp_path))
        nda_wp = (wp_nib.get_fdata() > 0.0).astype(np.float32)

        if nda_wp.shape != tuple(nda_shape):
            print("[error] nda_wp.shape != tuple(nda_shape)")
            input()

        # Calculate target shape for resampling
        spacing_target = (0.5, 0.5, 0.5)
        shape_target = [int(round(nda_shape[i] * spacing_orig[i] / spacing_target[i])) for i in range(3)]

        # Resample input volumes and segmentation
        nda_resize = np.zeros(shape=[3] + shape_target, dtype=np.float32)
        for s in range(3):
            nda_resize[s] = resize(nda[s], output_shape=shape_target, order=1)

        nda_wp_resize = (resize(nda_wp, output_shape=shape_target, order=0) > 0.0).astype(np.uint8)

        # Calculate ROI with margin
        margin = 32
        bbox = bbox2_3D(nda_wp_resize)
        bbox_new = np.array(bbox)
        for i in range(3):
            bbox_new[2*i] = max(0, bbox[2*i] - margin)
            bbox_new[2*i + 1] = min(shape_target[i] - 1, bbox[2*i + 1] + margin)

        # Crop ROI and normalize
        nda_resize_roi = nda_resize[:, bbox_new[0]:bbox_new[1], bbox_new[2]:bbox_new[3], bbox_new[4]:bbox_new[5]]
        nda_resize_roi = standard_normalization_multi_channel(nda_resize_roi)
        print("nda_resize_roi shape:", nda_resize_roi.shape)

        return {
            "affine": affine_orig,
            "bbox_new": bbox_new,
            "image": nda_resize_roi,
            "image_filename": "/input/t2.nii.gz",
            "nda_shape": np.array(nda_shape),
            "nda_resize_shape": np.array([nda_resize.shape[1], nda_resize.shape[2], nda_resize.shape[3]]),
            "pred_wp": nda_wp,
        }


class ProstateLesionSegOperator(Operator):
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
        self.output_name_seg = "seg_image"
        self.output_name_saved_images_folder = ""

        # The base class has an attribute called fragment to hold the reference to the fragment object
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image_t2)
        spec.input(self.input_name_image_adc)
        spec.input(self.input_name_image_highb)
        spec.input(self.input_name_image_organ_seg)
        spec.output(self.output_name_seg)
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

        # Set relevant metadata and save to disk as nii
        input_image_t2._metadata["affine"] = input_image_t2._metadata["nifti_affine_transform"]
        input_image_adc._metadata["affine"] = input_image_adc._metadata["nifti_affine_transform"]
        input_image_highb._metadata["affine"] = input_image_highb._metadata["nifti_affine_transform"]
        image_organ_seg._metadata["affine"] = image_organ_seg._metadata["nifti_affine_transform"]
        self.convert_and_save(input_image_t2, input_image_adc, input_image_highb, image_organ_seg, self.output_folder)

        print("\nBeginning lesion segmentation...")

        # Instantiate network and send to GPU
        nets = [
            RRUNet3D(
            in_channels=3,
            out_channels=2,
            blocks_down="1,2,3,4",
            blocks_up="3,2,1",
            num_init_kernels=32,
            recurrent=False,
            residual=True,
            attention=False,
            debug=False,
            )
            for _ in range(5)
        ]
        if torch.cuda.is_available():
            nets = [net.to("cuda") for net in nets]
        for net in nets:
            net.eval()

        # Set model weights to models in container
        tags = ["fold0", "fold1", "fold2", "fold3", "fold4"]
        weight_files = [
            self.model_path / tags[0] / "model_best_fold0.pth.tar",
            self.model_path / tags[1] / "model_best_fold1.pth.tar",
            self.model_path / tags[2] / "model_best_fold2.pth.tar",
            self.model_path / tags[3] / "model_best_fold3.pth.tar",
            self.model_path / tags[4] / "model_best_fold4.pth.tar",
        ]

        # Create DataLoader and preprocess image
        print("Loading input...")
        validation_dataset = SegmentationDataset(output_path=self.output_folder, data_purpose="testing")
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)
        data = next(iter(validation_loader))
        if torch.cuda.is_available():
            inputs = data["image"].to("cuda")
        else:
            inputs = data["image"]
        inputs_shape = ( inputs.size()[-3], inputs.size()[-2], inputs.size()[-1])

        def run_inference(tag, model_name, net):
            self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=self.output_folder,
            model_name=model_name,
            tag=tag,
            )

        import concurrent.futures

        # Perform inference in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_inference, tags[i], weight_files[i], nets[i]) for i in range(len(tags))]
            concurrent.futures.wait(futures)

        # Convert to Image and transpose back to DHW
        lesion_mask = self.merge_volumes(output_path=self.output_folder, data=data, tags=tags)
        lesion_mask = Image(
            data=lesion_mask.T, metadata=input_image_t2.metadata()
        )

        # Now emit data to the output ports of this operator
        op_output.emit(lesion_mask, self.output_name_seg)
        op_output.emit(self.output_folder, self.output_name_saved_images_folder)

    def convert_and_save(self, image1, image2, image3, organ_mask, output_path):
        """Converts and saves the input Images on disk in nii.gz format."""

        # Create a single SaveImage operator
        save_op = SaveImage(output_dir=output_path, output_postfix="", output_dtype=np.float32, resample=False)

        # Define the images and their output names
        images = [
            (image1, "t2"),
            (image2, "adc"),
            (image3, "highb"),
            (organ_mask, "organ")
        ]

        # Process each image
        for img, name in images:
            meta_tensor = MetaTensor(np.expand_dims(img.asnumpy().T, axis=0), meta=img.metadata())
            meta_tensor.meta["filename_or_obj"] = name
            save_op(meta_tensor)

    def custom_inference(self, data, inputs, inputs_shape, net, output_path, tag, model_name: str = "") -> np.ndarray:
        """Performs inference on the input image."""

        # Load model weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_name, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint["state_dict"])

        # Initialize variables
        output_classes = 2
        np_output_prob = np.zeros(shape=(output_classes,) + inputs_shape, dtype=np.float32)
        np_count = np.zeros(shape=(output_classes,) + inputs_shape, dtype=np.float32)

        # Create input ranges that are multiples of 32
        print("Inputs shape: ", inputs.size())
        multiple = 32
        output_len_x, output_len_y, output_len_z = inputs_shape[0], inputs_shape[1], inputs_shape[2]
        ranges_x = [(0, output_len_x // multiple * multiple)]
        ranges_y = [(0, output_len_y // multiple * multiple)]
        ranges_z = [(0, output_len_z // multiple * multiple)]
        if output_len_x // multiple * multiple < output_len_x:
            ranges_x += [(output_len_x - output_len_x // multiple * multiple, output_len_x)]
        if output_len_y // multiple * multiple < output_len_y:
            ranges_y += [(output_len_y - output_len_y // multiple * multiple, output_len_y)]
        if output_len_z // multiple * multiple < output_len_z:
            ranges_z += [(output_len_z - output_len_z // multiple * multiple, output_len_z)]

        # Inference
        with torch.set_grad_enabled(False):
            for rx in ranges_x:
                for ry in ranges_y:
                    for rz in ranges_z:

                        output_patch = net(inputs[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]])
                        output_patch = output_patch.cpu().detach().numpy()
                        output_patch = np.squeeze(output_patch)
                        np_output_prob[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]] += output_patch
                        np_count[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]] += 1.0

        # Convert output back to numpy 3D array
        outputs = np_output_prob / np_count
        outputs_prob = copy.deepcopy(outputs)
        outputs = np.argmax(outputs, axis=0)
        outputs = np.squeeze(outputs).astype(np.uint8)

        # Initialize output placeholder
        nda_resize_shape = data["nda_resize_shape"]
        nda_resize_shape = nda_resize_shape.detach().numpy()
        nda_resize_shape = np.squeeze(nda_resize_shape)
        outputs_resize = np.zeros(shape=(nda_resize_shape[0], nda_resize_shape[1], nda_resize_shape[2]), dtype=np.uint8)

        outputs_prob_resize = np.zeros(
            shape=(output_classes, nda_resize_shape[0], nda_resize_shape[1], nda_resize_shape[2]), dtype=np.float32
        )
        bbox_new = data["bbox_new"]
        # bbox_new = bbox_new.detach().numpy()
        bbox_new = np.squeeze(bbox_new)
        outputs_resize[bbox_new[0] : bbox_new[1], bbox_new[2] : bbox_new[3], bbox_new[4] : bbox_new[5]] = outputs
        outputs_prob_resize[
            :, bbox_new[0] : bbox_new[1], bbox_new[2] : bbox_new[3], bbox_new[4] : bbox_new[5]
        ] = outputs_prob

        # Resample to original dimensions
        nda_shape = data["nda_shape"]
        nda_shape = nda_shape.detach().numpy()
        nda_shape = np.squeeze(nda_shape)
        outputs_orig = resize(outputs_resize, output_shape=nda_shape, order=0)
        outputs_orig = (outputs_orig > 0.0).astype(np.uint8)
        outputs_prob_orig = np.zeros(shape=(3, nda_shape[0], nda_shape[1], nda_shape[2]), dtype=np.float32)
        for _s in range(output_classes):
            outputs_prob_orig[_s, ...] = resize(outputs_prob_resize[_s, ...], output_shape=nda_shape, order=1)
        outputs_prob_orig = outputs_prob_orig.astype(np.float32)

        # Make lesion directory if it doesn't exist
        if not os.path.exists(str(output_path) + "/lesion"):
            os.makedirs(str(output_path) + "/lesion")

        # Write image to disk
        output_filename = (str(output_path) + "/lesion/" + tag + "_lesion_prob.nii.gz")
        print("Created file:", output_filename)

        # Create affine transformation matrix
        affine = data["affine"]
        affine = affine.detach().numpy()
        affine = np.squeeze(affine)
        codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(np.linalg.inv(affine)))

        for _j in range(1, output_classes):
            reverted_nda_prob = nib.orientations.apply_orientation(outputs_prob_orig[_j, ...], codes)
            nib.save(nib.Nifti1Image(reverted_nda_prob, affine), os.path.join(output_path, output_filename))

    def merge_volumes(self, data, tags, output_path):
        """Merges the probability maps and creates a lesion mask."""

        # Merge probability maps
        affine = []
        for i in range(len(tags)):
            img_prob = nib.load(str(output_path) + "/lesion/" + str(tags[i]) + "_lesion_prob.nii.gz")
            if i == 0:
                nda_prob = img_prob.get_fdata()
                affine = img_prob.affine
            else:
                nda_prob += img_prob.get_fdata()
        nda_prob = nda_prob / (len(tags))
        nib.save(nib.Nifti1Image(nda_prob, affine), str(output_path) + "/lesion/" + "merged_lesion_prob.nii.gz")

        # Outlier rejection based on original prostate segmentation
        nda_wp = nib.load(str(output_path) + "/organ/organ.nii.gz").get_fdata()
        nda_wp = np.squeeze(nda_wp)
        nda_prob = np.multiply(nda_prob, nda_wp.astype(np.float32))

        # Print statistics of the probability map to two decimal places
        print("nda_prob min:", np.min(nda_prob))
        print("nda_prob max:", np.max(nda_prob))
        print("nda_prob mean:", np.mean(nda_prob))
        print("nda_prob std:", np.std(nda_prob))

        # Create lesion mask
        threshold = 0.6344772701607316
        nda_prob = (nda_prob >= threshold).astype(np.uint8)
        nib.save(nib.Nifti1Image(nda_prob, affine), str(output_path) + "/lesion/" + "lesion_mask.nii.gz")

        # Check if lesion_mask is all 0's
        if np.sum(nda_prob) == 0:
            print("**No lesions detected**")

        return nda_prob
