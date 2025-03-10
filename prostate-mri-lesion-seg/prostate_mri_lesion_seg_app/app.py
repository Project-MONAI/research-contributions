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

# MONAI Deploy SDK imports
from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

# Local imports
from organ_seg_operator import ProstateSegOperator
from custom_lesion_seg_operator import ProstateLesionSegOperator
from custom_lesion_classifier_operator import ProstateLesionClassifierOperator


# Custom rules for T2, ADC, and HIGHB series selection in ProstateX
Rules_T2 = """
{
    "selections": [
        {
            "name": "t2",
            "conditions" :
            {
                "Modality": "MR",
                "ImageType": [
                            "ORIGINAL",
                            "PRIMARY"
                        ],
                "SeriesDescription": "((?=AX T2)|(?=AX T2  6 NSA)|(?=AX T2  FAST)|(?=AX T2 . voxel 7x.8)|(?=AX T2 B1 Default)|(?=AX T2 CS)|(?=AX T2 cs 3.0)|(?=AX T2 FAST)|(?=AX T2 FRFSE)|(?=AX T2 N/S)|(?=AX T2 No sense)|(?=AX T2 NS)|(?=AX T2 NSA 3)|(?=AX T2 NSA 4)|(?=AX T2 NSA 5)|(?=AX T2 PROP)|(?=Ax T2 PROSTATE)|(?=AX T2 SMALL FOV)|(?=Ax T2 thin FRFSE)|(?=sT2 TSE ax no post)|(?=T2 AX)|(?=T2 AX SMALL FOV[*]? IF MOTION REPEAT[*]?)|(?=T2 AXIAL 3MM)|(?=T2 TRA 3mm)|(?=T2 TSE Ax)|(?=T2 TSE ax cs)|(?=T2 TSE ax hi)|(?=T2 TSE ax hi sense)|(?=T2 TSE ax no sense)|(?=T2 TSE ax NS)|(?=T2 TSE ax NSA 3)|(?=t2_tse_tra)|(?=t2_tse_tra_320_p2)|(?=t2_tse_tra_3mm _SFOV_TE 92)|(?=t2_tse_tra_Grappa3)|(?=T2W_TSE)|(?=T2W_TSE_ax)|(?=T2W_TSE_ax PSS Refoc 52)|(?=T2W_TSE_ax zoom PSS Refoc))"
            }
        }
    ]
}
"""

Rules_ADC = """
{
    "selections": [
        {
            "name": "adc",
            "conditions":
            {
                "Modality": "MR",
                "ImageType": [
                            "DIFFUSION",
                            "ADC"
                        ],
                "SeriesDescription": "((?=ADC (10^-6 mm²/s))|(?=Apparent Diffusion Coefficient (mm2/s))|(?=AX DIFFUSION_ADC_DFC_MIX)|(?=.*AX DWI (50,1500)_ADC.*)|(?=AX DWI_ADC_DFC_MIX)|(?=b_1500 prostate_ADC)|(?=b_2000 prostate_ADC)|(?=d3B ADC 3B 750  ERC SSh_DWI FAST SENSE)|(?=dADC)|(?=dADC 0_1500)|(?=dADC 100 400 600)|(?=dADC 2)|(?=dADC 3)|(?=dADC ALL)|(?=dADC b 0 1000 2000)|(?=dADC from 0_1500)|(?=dADC from b0_600)|(?=dADC from B0-1500)|(?=dADC Map)|(?=dADC map 1)|(?=dADC MAP 2)|(?=dADC_1 axial)|(?=dADC_b375_750_1150)|(?=ddADC MAP)|(?=DIFF bv1400_ADC)|(?=diff tra b 50 500 800 WIP511b alle spoelen_ADC)|(?=diffusie-3Scan-4bval_fs_ADC)|(?=dReg - WIP SSh_DWI FAST SENSE)|(?=dSSh_DWI SENSE)|(?=DWI PROSTATE_ADC)|(?=dWIP 3B 600 w ERC SSh_DWI S2Ovs2)|(?=dWIP 3B ADC 3B 600 w/o ERC SSh_DWI FAST SENSE)|(?=dWIP SSh_DWI FAST SENSE)|(?=ep2d_diff_new 16 measipat_ADC)|(?=ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC)|(?=ep2d_diff_tra_DYNDIST_ADC)|(?=ep2d_diff_tra_DYNDIST_MIX_ADC)|(?=ep2d_diff_tra2x2_Noise0_FS_DYNDIST_ADC)|(?=ep2d-advdiff-3Scan-4bval_spair_511b_ADC))"
            }
        }
    ]
}
"""

Rules_HIGHB = """
{
    "selections": [
        {
            "name": "highb",
            "conditions" :
            {
                "Modality": "MR",
                "ImageType": [
                            "DIFFUSION",
                            "TRACEW"
                        ],
                "SeriesDescription": "((?=3B 2000 w ERC SSh_DWI)|(?=AX DIFFUSION_CALC_BVAL_DFC_MIX)|(?=AX DWI)|(?=.*AX DWI (50,1500).*)|(?=Ax DWI BH)|(?=AX DWI_TRACEW_DFC_MIX)|(?=Axial FOCUS DWI 1400)|(?=b_1500 prostate)|(?=b_2000 prostate)|(?=DIFF bv1400)|(?=diff tra b 50 500 800 WIP511b alle spoelenCALC_BVAL)|(?=diffusie-3Scan-4bval_fsCALC_BVAL)|(?=DW_Synthetic: Ax DWI All B-50-800 Synthetic B-1400)|(?=.*DW_Synthetic: Ax Focus 50,500,800,1400,2000.*)|(?=DWI PROSTATE)|(?=DWI_5b_0_1500)|(?=DWI_b2000)|(?=DWI_b2000_new)|(?=DWI_b2000_new SENSE)|(?=DWI_b2000_NSA6 SENSE)|(?=ep2d_diff_b1400_new 32 measipat)|(?=ep2d_diff_tra_DYNDIST_MIXCALC_BVAL$)|(?=ep2d_diff_tra_DYNDISTCALC_BVAL)|(?=ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL)|(?=ep2d-advdiff-3Scan-high bvalue 1400)|(?=sb_1500)|(?=sb_2000)|(?=sB1400)|(?=sb1500)|(?=sb-1500)|(?=sb1500 r5 only)|(?=sb-2000)|(?=sDWI_b_2000)|(?=sDWI_b2000))"
            }
        }
    ]
}
"""

class AIProstateLesionSegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.debug(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.debug(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.debug(f"Begin {self.compose.__name__}")

        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        self._logger.info(f"App input and output path: {app_input_path}, {app_output_path}")

        # Data pipeline
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="dcm_loader_op"
        )
        series_selector_T2_op = DICOMSeriesSelectorOperator(self, rules=Rules_T2, name="series_selector_T2")
        series_to_vol_T2_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_T2")
        series_selector_ADC_op = DICOMSeriesSelectorOperator(self, rules=Rules_ADC, name="series_selector_ADC")
        series_to_vol_ADC_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_ADC")
        series_selector_HIGHB_op = DICOMSeriesSelectorOperator(self, rules=Rules_HIGHB, name="series_selector_HIGHB")
        series_to_vol_HIGHB_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_HIGHB")

        # AI operators
        organ_seg_op = ProstateSegOperator(self, app_context=app_context, model_path=model_path/"organ", name="organ_seg_op")
        lesion_seg_op = ProstateLesionSegOperator(self, app_context=app_context, model_path=model_path, name="lesion_seg_op")
        lesion_classifier_op = ProstateLesionClassifierOperator(self, app_context=app_context, model_path=model_path, name="lesion_classifier_op")

        #################### Pipeline DAG ####################
        # Data ingestion
        self.add_flow(study_loader_op, series_selector_T2_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(study_loader_op, series_selector_ADC_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(study_loader_op, series_selector_HIGHB_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(series_selector_T2_op, series_to_vol_T2_op, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(series_selector_ADC_op, series_to_vol_ADC_op, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(series_selector_HIGHB_op, series_to_vol_HIGHB_op, {("study_selected_series_list", "study_selected_series_list")})

        # Organ inference
        self.add_flow(series_to_vol_T2_op, organ_seg_op, {("image", "image")})

        # Lesion inference
        self.add_flow(series_to_vol_T2_op, lesion_seg_op, {("image", "image_t2")})
        self.add_flow(series_to_vol_ADC_op, lesion_seg_op, {("image", "image_adc")})
        self.add_flow(series_to_vol_HIGHB_op, lesion_seg_op, {("image", "image_highb")})
        self.add_flow(organ_seg_op, lesion_seg_op, {("seg_image", "image_organ_seg")})

        # Lesion classification
        self.add_flow(series_to_vol_T2_op, lesion_classifier_op, {("image", "image_t2")})
        self.add_flow(series_to_vol_ADC_op, lesion_classifier_op, {("image", "image_adc")})
        self.add_flow(series_to_vol_HIGHB_op, lesion_classifier_op, {("image", "image_highb")})
        self.add_flow(organ_seg_op, lesion_classifier_op, {("seg_image", "image_organ_seg")})
        self.add_flow(lesion_seg_op, lesion_classifier_op, {("seg_image", "image_lesion_seg")})

        self._logger.debug(f"End {self.compose.__name__}")
        #################### Pipeline DAG ####################

if __name__ == "__main__":
    # Creates the app and test it standalone.
    logging.info(f"Begin {__name__}")
    AIProstateLesionSegApp().run()
    logging.info(f"End {__name__}")
