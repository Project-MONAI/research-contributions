"""
Prostate-MRI_Lesion_Detection, v2.0 (Release date: August 2, 2023)
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
each copy of the SOFTWARE."""

import copy
import logging
import os
from typing import Optional

import nibabel as nib
import numpy as np

# AI/CV imports
import SimpleITK as sitk
import torch

# Local imports
from network import RRUNet3D
from skimage.transform import resize
from torch.utils.data import Dataset

# MONAI Deploy App SDK imports
import monai.deploy.core as md

# MONAI imports
from monai.data import MetaTensor
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.transforms import SaveImage


def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]


def standard_normalization_multi_channel(nda):
    for _i in range(nda.shape[0]):
        if np.amax(np.abs(nda[_i, ...])) < 1e-7:
            continue
        nda[_i, ...] = (nda[_i, ...] - np.mean(nda[_i, ...])) / np.std(nda[_i, ...])

    return nda


class SegmentationDataset(Dataset):
    def __init__(self, output_path, data_purpose):
        self.data_purpose = data_purpose
        self.output_path = output_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Composes transforms for preprocessing input before predicting on a model."""
        print("Pre-processing input image...")

        affine_orig, nda = [], []

        # Load T2 in ITK format

        t2_name = str(self.output_path) + "/t2/t2.nii.gz"
        t2 = sitk.ReadImage(t2_name)

        # Load T2 in Nibabel format
        img = nib.as_closest_canonical(nib.load(t2_name))
        affine_orig = img.affine
        spacing_orig = img.header.get_zooms()
        nda.append(img.get_fdata())

        # Resample ADC

        adc_name = str(self.output_path) + "/adc/adc.nii.gz"
        adc = sitk.ReadImage(adc_name)
        adc = sitk.Resample(
            adc,
            t2.GetSize(),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            t2.GetOrigin(),
            t2.GetSpacing(),
            t2.GetDirection(),
            0,
            t2.GetPixelID(),
        )
        sitk.WriteImage(adc, adc_name)

        # Load ADC
        img = nib.as_closest_canonical(nib.load(adc_name))
        nda.append(img.get_fdata())

        # Resample HighB

        highb_name = str(self.output_path) + "/highb/highb.nii.gz"
        highb = sitk.ReadImage(highb_name)
        highb = sitk.Resample(
            highb,
            t2.GetSize(),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            t2.GetOrigin(),
            t2.GetSpacing(),
            t2.GetDirection(),
            0,
            t2.GetPixelID(),
        )
        sitk.WriteImage(highb, highb_name)

        # Load HighB
        img = nib.as_closest_canonical(nib.load(highb_name))
        nda.append(img.get_fdata())

        # Stack input modalities
        nda = np.stack(nda, axis=0)
        nda = nda.astype(np.float32)
        nda_shape = [nda.shape[1], nda.shape[2], nda.shape[3]]

        # Read in whole prostate segmentation

        img_wp_filename = str(self.output_path) + "/organ/organ.nii.gz"
        img_wp = nib.as_closest_canonical(nib.load(img_wp_filename))
        nda_wp = img_wp.get_fdata()
        nda_wp = (nda_wp > 0.0).astype(np.float32)
        if nda_wp.shape != tuple(nda_shape):
            print("[error] nda_wp.shape != tuple(nda_shape)")
            input()

        # Resample to isotropic spacing
        spacing_target = (0.5, 0.5, 0.5)
        shape_target = []
        for _s in range(3):
            shape_target_s = float(nda_shape[_s]) * spacing_orig[_s] / spacing_target[_s]
            shape_target_s = np.round(shape_target_s).astype(np.int16)
            shape_target.append(shape_target_s)

        nda_resize = np.zeros(shape=[3] + shape_target, dtype=np.float32)
        for _s in range(3):
            nda_resize[_s, ...] = resize(nda[_s, ...], output_shape=shape_target, order=1)
        nda_resize_shape = [nda_resize.shape[1], nda_resize.shape[2], nda_resize.shape[3]]
        nda_wp_resize = resize(nda_wp, output_shape=shape_target, order=0)
        nda_wp_resize = (nda_wp_resize > 0.0).astype(np.uint8)

        # Calculate ROI for whole prostate
        margin = 32
        bbox = bbox2_3D(nda_wp_resize)
        bbox_new = np.array(bbox)
        for _s in range(3):
            bbox_new[2 * _s] = max(0, bbox[2 * _s] - margin)
            bbox_new[2 * _s + 1] = min(shape_target[_s] - 1, bbox[2 * _s + 1] + margin)

        # Crop ROI and preprocess (normalize: 0-mean, 1-stddev)

        nda_resize_roi = nda_resize[:, bbox_new[0] : bbox_new[1], bbox_new[2] : bbox_new[3], bbox_new[4] : bbox_new[5]]
        nda_resize_roi = standard_normalization_multi_channel(nda_resize_roi)

        sample = {
            "affine": affine_orig,
            "bbox_new": bbox_new,
            "image": nda_resize_roi,
            "image_filename": "/input/t2.nii.gz",
            "nda_shape": np.array(nda_shape),
            "nda_resize_shape": np.array(nda_resize_shape),
            "pred_wp": nda_wp,
        }

        return sample


@md.input("image1", Image, IOType.IN_MEMORY)
@md.input("image2", Image, IOType.IN_MEMORY)
@md.input("image3", Image, IOType.IN_MEMORY)
@md.input("organ_mask", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(
    pip_packages=["monai>=1.0.1", "torch>=1.12.1", "numpy>=1.21", "nibabel", "SimpleITK", "scikit-image", "highdicom"]
)
class CustomProstateLesionSegOperator(Operator):
    """Performs Prostate Lesion segmentation with a 3D image converted from a mp-DICOM MRI series."""

    def __init__(self, model_name: Optional[str] = ""):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()

        self._model_name = model_name.strip() if isinstance(model_name, str) else ""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        output_path = context.output.get().path

        # Load inputs
        image1 = op_input.get("image1")
        if not image1:
            raise ValueError("Input image1 is not found.")
        image2 = op_input.get("image2")
        if not image2:
            raise ValueError("Input image2 is not found.")
        image3 = op_input.get("image3")
        if not image3:
            raise ValueError("Input image3 is not found.")
        organ_mask = op_input.get("organ_mask")
        if not organ_mask:
            raise ValueError("Input organ_mask is not found.")

        # Set relevant metadata and save to disk as nii
        image1._metadata["affine"] = image1._metadata["nifti_affine_transform"]
        image2._metadata["affine"] = image2._metadata["nifti_affine_transform"]
        image3._metadata["affine"] = image3._metadata["nifti_affine_transform"]
        organ_mask._metadata["affine"] = organ_mask._metadata["nifti_affine_transform"]
        self.convert_and_save(image1, image2, image3, organ_mask, output_path)

        # Create model and move to GPU
        input_channels = 3
        output_classes = 2
        enc = "1,2,3,4"
        dec = "3,2,1"
        recurrent = False
        residual = True
        attention = False

        net = RRUNet3D(
            in_channels=input_channels,
            out_channels=output_classes,
            blocks_down=enc,
            blocks_up=dec,
            num_init_kernels=32,
            recurrent=recurrent,
            residual=residual,
            attention=attention,
            debug=False,
        )
        net = net.to("cuda")
        net.eval()

        # Set model weights to models in container
        tags = ["fold0", "fold1", "fold2", "fold3", "fold4"]

        weight_files = [
            "/opt/monai/app/models/" + tags[0] + "/model_best_fold0.pth.tar",
            "/opt/monai/app/models/" + tags[1] + "/model_best_fold1.pth.tar",
            "/opt/monai/app/models/" + tags[2] + "/model_best_fold2.pth.tar",
            "/opt/monai/app/models/" + tags[3] + "/model_best_fold3.pth.tar",
            "/opt/monai/app/models/" + tags[4] + "/model_best_fold4.pth.tar",
        ]

        # Create DataLoader and preprocess image
        print("Loading input...")
        validation_dataset = SegmentationDataset(output_path=output_path, data_purpose="testing")
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)
        data = next(iter(validation_loader))
        inputs = data["image"].to("cuda")
        inputs_shape = (inputs.size()[-3], inputs.size()[-2], inputs.size()[-1])
        print("inputs_shape:", inputs_shape)

        self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=output_path,
            model_name=weight_files[0],
            tag=tags[0],
        )
        self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=output_path,
            model_name=weight_files[1],
            tag=tags[1],
        )
        self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=output_path,
            model_name=weight_files[2],
            tag=tags[2],
        )
        self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=output_path,
            model_name=weight_files[3],
            tag=tags[3],
        )
        self.custom_inference(
            data=data,
            inputs=inputs,
            inputs_shape=inputs_shape,
            net=net,
            output_path=output_path,
            model_name=weight_files[4],
            tag=tags[4],
        )

        lesion_mask = self.merge_volumes(output_path=output_path, tags=tags)
        lesion_mask = Image(
            data=lesion_mask.T, metadata=image1.metadata()
        )  # Convert to Image and transpose back to DHW

        op_output.set(lesion_mask, "seg_image")

    def convert_and_save(self, image1, image2, image3, organ_mask, output_path):
        """Converts and saves the input Images on disk in nii.gz format."""

        save_op_t2 = SaveImage(output_dir=output_path, output_postfix="", output_dtype=np.float32, resample=False)
        t2_image = MetaTensor(np.expand_dims(image1.asnumpy().T, axis=0), meta=image1.metadata())
        t2_image.meta["filename_or_obj"] = "t2"
        save_op_t2(t2_image)
        save_op_adc = SaveImage(output_dir=output_path, output_postfix="", output_dtype=np.float32, resample=False)
        adc_image = MetaTensor(np.expand_dims(image2.asnumpy().T, axis=0), meta=image2.metadata())
        adc_image.meta["filename_or_obj"] = "adc"
        save_op_adc(adc_image)
        save_op_highb = SaveImage(output_dir=output_path, output_postfix="", output_dtype=np.float32, resample=False)
        highb_image = MetaTensor(np.expand_dims(image3.asnumpy().T, axis=0), meta=image3.metadata())
        highb_image.meta["filename_or_obj"] = "highb"
        save_op_highb(highb_image)
        save_op_organ_mask = SaveImage(
            output_dir=output_path, output_postfix="", output_dtype=np.float32, resample=False
        )
        organ_mask_image = MetaTensor(np.expand_dims(organ_mask.asnumpy().T, axis=0), meta=organ_mask.metadata())
        organ_mask_image.meta["filename_or_obj"] = "organ"
        save_op_organ_mask(organ_mask_image)

    def custom_inference(self, data, inputs, inputs_shape, net, output_path, tag, model_name: str = "") -> np.ndarray:
        """Performs inference on the input image."""

        output_classes = 2
        current_model_path = model_name
        current_model = torch.load(current_model_path)
        net.load_state_dict(current_model["state_dict"])

        # Initialize variables
        np_output_prob = np.zeros(shape=(output_classes,) + inputs_shape, dtype=np.float32)
        np_count = np.zeros(shape=(output_classes,) + inputs_shape, dtype=np.float32)

        # Create input ranges that are multiple of 32
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

        # Run inference
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
        bbox_new = bbox_new.detach().numpy()
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

        # Outlier rejection based on original prostate segmentation
        nda_wp = data["pred_wp"].cpu().detach().numpy()
        nda_wp = np.squeeze(nda_wp)
        for _j in range(output_classes):
            outputs_prob_orig[_j, ...] = np.multiply(outputs_prob_orig[_j, ...], nda_wp.astype(np.float32))

        # Create affine transformation matrix
        affine = data["affine"]
        affine = affine.detach().numpy()
        affine = np.squeeze(affine)
        codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(np.linalg.inv(affine)))

        # Make lesion directory if it doesn't exist
        if not os.path.exists(str(output_path) + "/lesion"):
            os.makedirs(str(output_path) + "/lesion")

        # Write image to disk
        output_filename = data["image_filename"]

        output_filename = (
            str(output_path)
            + "/lesion/"
            + tag
            + output_filename[0].replace(os.sep, "_").replace("t2.", "prob.").replace("input", "lesion")
        )
        print("output filename:", output_filename)
        for _j in range(1, output_classes):
            reverted_nda_prob = nib.orientations.apply_orientation(outputs_prob_orig[_j, ...], codes)
            nib.save(nib.Nifti1Image(reverted_nda_prob, affine), os.path.join(output_path, output_filename))

    def merge_volumes(self, tags, output_path):
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

        # Create lesion mask
        threshold = 0.6344772701607316
        nda_prob = (nda_prob >= threshold).astype(np.uint8)
        nib.save(nib.Nifti1Image(nda_prob, affine), str(output_path) + "/lesion/" + "lesion_mask.nii.gz")

        return nda_prob
