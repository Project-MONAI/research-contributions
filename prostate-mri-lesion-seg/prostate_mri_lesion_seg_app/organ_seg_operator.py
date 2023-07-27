import logging
from os import path
from typing import Optional
from numpy import uint8

# MONAI Deploy App SDK imports
import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, MonaiSegInferenceOperator

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

@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai>=1.0.1", "torch>=1.12.1", "numpy>=1.21", "nibabel"])
class ProstateSegOperator(Operator):
    """Performs Prostate segmentation with a 3D image converted from a DICOM MRI (T2) series."""

    def __init__(self, model_name: Optional[str] = ""):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "label"
        self._model_name = model_name.strip() if isinstance(model_name, str) else ""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image = op_input.get("image")
        if not input_image:
            raise ValueError("Input image is not found.")

        output_path = context.output.get().path

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process(pre_transforms, output_path)

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            (
                128,
                128,
                16,
            ),
            pre_transforms,
            post_transforms,
            model_name=self._model_name,
        )

        # Setting the keys used in the dictionary based transforms may change.
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now let the built-in operator handles the work with the I/O spec and execution context.
        infer_operator.compute(op_input, op_output, context)

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        my_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                DataStatsd(keys=my_key, name='Loaded image'),
                
                EnsureChannelFirstd(keys=my_key),
                DataStatsd(keys=my_key, name='Channel-first image'),
                
                Orientationd(keys=my_key, axcodes="RAS"),
                Spacingd(keys=my_key, pixdim=[1.0, 1.0, 1.0], mode=["bilinear"]),
                NormalizeIntensityd(keys=my_key, nonzero=True, channel_wise=True),
                DataStatsd(keys=my_key, name='Resampled and normalized image'),

                EnsureTyped(keys=my_key),
            ]
        )

    def post_process(self, pre_transforms: Compose, out_dir: str = "./") -> Compose:        
        """Composes transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key
        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                DataStatsd(keys=pred_key, name='Model output'),

                Invertd(keys=pred_key, transform=pre_transforms, orig_keys=self._input_dataset_key, nearest_interp=False, to_tensor=True),
                DataStatsd(keys=pred_key, name='Inverted output'),

                AsDiscreted(keys=pred_key, argmax=True, threshold=0.5),
                DataStatsd(keys=pred_key, name='AsDiscrete output'),
            ]
        )