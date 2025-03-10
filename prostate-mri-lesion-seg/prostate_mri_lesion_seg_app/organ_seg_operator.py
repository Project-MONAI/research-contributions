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

# MONAI Deploy App SDK imports
from monai.deploy.core import AppContext, ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.operators.monai_seg_inference_operator import InfererType, InMemImageReader, MonaiSegInferenceOperator

# MONAI imports
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    DataStatsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

class ProstateSegOperator(Operator):
    """Performs Prostate segmentation with a 3D image converted from a DICOM MRI (T2) series."""

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
        self.input_name_image = "image"
        self.output_name_seg = "seg_image"
        self.output_name_saved_images_folder = ""

        # Call the base class __init__() last.
        # Also, the base class has an attribute called fragment for storing the fragment object
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)
        spec.output(self.output_name_seg)
        spec.output(self.output_name_saved_images_folder).condition(
            ConditionType.NONE
        )  # Output not requiring a receiver

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image (T2) is not found.")

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)
        pre_transforms = self.pre_process(_reader, str(self.output_folder))
        post_transforms = self.post_process(pre_transforms, str(self.output_folder))

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            self.fragment,
            roi_size=(128, 128, 16),
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            overlap=0.5,
            app_context=self.app_context,
            model_name="",
            inferer=InfererType.SLIDING_WINDOW,
            sw_batch_size=4,
            model_path=self.model_path,
            name="monai_seg_inference_op",
        )

        # Setting the keys used in the dictionary based transforms may change.
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now emit data to the output ports of this operator
        op_output.emit(infer_operator.compute_impl(input_image, context), self.output_name_seg)
        op_output.emit(self.output_folder, self.output_name_saved_images_folder)

    def pre_process(self, img_reader, out_dir: str = "./input_images") -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        my_key = self._input_dataset_key
        print("\nBeginning organ segmentation...")

        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                DataStatsd(keys=my_key, name="Loaded image"),
                EnsureChannelFirstd(keys=my_key),
                DataStatsd(keys=my_key, name="Channel-first image"),
                Orientationd(keys=my_key, axcodes="RAS"),
                Spacingd(keys=my_key, pixdim=[1.0, 1.0, 1.0], mode=["bilinear"]),
                NormalizeIntensityd(keys=my_key, nonzero=True, channel_wise=True),
                DataStatsd(keys=my_key, name="Resampled and normalized image"),
                EnsureTyped(keys=my_key),
            ]
        )

    def post_process(self, pre_transforms: Compose, out_dir: str = "./prediction_output") -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pred_key = self._pred_dataset_key

        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                DataStatsd(keys=pred_key, name="Model output"),
                Invertd(
                    keys=pred_key,
                    transform=pre_transforms,
                    orig_keys=self._input_dataset_key,
                    nearest_interp=False,
                    to_tensor=True,
                ),
                DataStatsd(keys=pred_key, name="Inverted output"),
                AsDiscreted(keys=pred_key, argmax=True, threshold=0.5),
                DataStatsd(keys=pred_key, name="AsDiscrete output")
            ]
        )
