import logging

from custom_lesion_seg_operator import CustomProstateLesionSegOperator

# Local imports
from organ_seg_operator import ProstateSegOperator

# MONAI Deploy SDK imports
from monai.deploy.core import Application, resource
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

# This is a sample series selection rule in JSON, simply selecting MRI series.
# Please see more details in DICOMSeriesSelectorOperator.
# Sample_Rules_Text = """
# {
#     "selections": [
#         {
#             "name": "MRI Series",
#             "conditions": {
#                 "StudyDescription": "(.*?)",
#                 "Modality": "MR",
#                 "SeriesDescription": "(.*?)",
#                 "ImageType": ["PRIMARY", "ORIGINAL"],
#             }
#         }
#     ]
# }
# """

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


@resource(cpu=1, gpu=1, memory="7Gi")
class AIProstateLesionSegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.

        self._logger.debug(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.debug(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.debug(f"Begin {self.compose.__name__}")

        # Data pipeline
        study_loader_op = DICOMDataLoaderOperator()
        series_selector_T2_op = DICOMSeriesSelectorOperator(rules=Rules_T2)
        series_to_vol_T2_op = DICOMSeriesToVolumeOperator()
        series_selector_ADC_op = DICOMSeriesSelectorOperator(rules=Rules_ADC)
        series_to_vol_ADC_op = DICOMSeriesToVolumeOperator()
        series_selector_HIGHB_op = DICOMSeriesSelectorOperator(rules=Rules_HIGHB)
        series_to_vol_HIGHB_op = DICOMSeriesToVolumeOperator()

        # Organ and lesion operators
        organ_seg_op = ProstateSegOperator(model_name="organ")
        lesion_seg_op = CustomProstateLesionSegOperator()

        #################### Pipeline DAG ####################
        # Data ingestion
        self.add_flow(study_loader_op, series_selector_T2_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(study_loader_op, series_selector_ADC_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(study_loader_op, series_selector_HIGHB_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            series_selector_T2_op, series_to_vol_T2_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(
            series_selector_ADC_op, series_to_vol_ADC_op, {"study_selected_series_list": "study_selected_series_list"}
        )
        self.add_flow(
            series_selector_HIGHB_op,
            series_to_vol_HIGHB_op,
            {"study_selected_series_list": "study_selected_series_list"},
        )

        # Organ inference
        self.add_flow(series_to_vol_T2_op, organ_seg_op, {"image": "image"})

        # Lesion Inference
        self.add_flow(series_to_vol_T2_op, lesion_seg_op, {"image": "image1"})
        self.add_flow(series_to_vol_ADC_op, lesion_seg_op, {"image": "image2"})
        self.add_flow(series_to_vol_HIGHB_op, lesion_seg_op, {"image": "image3"})
        self.add_flow(organ_seg_op, lesion_seg_op, {"seg_image": "organ_mask"})
        #################### Pipeline DAG ####################

        self._logger.debug(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    #     -m <model file>, for model file path
    # e.g.
    #     python3 app.py -i input -m model.ts

    AIProstateLesionSegApp(do_run=True)
