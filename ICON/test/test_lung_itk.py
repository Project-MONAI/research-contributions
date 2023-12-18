import itk
import numpy as np
import unittest
import matplotlib.pyplot as plt
import numpy as np

import icon_registration.test_utils
import icon_registration.pretrained_models
import icon_registration.itk_wrapper


class TestItkRegistration(unittest.TestCase):
    def test_itk_registration(self):

        model = icon_registration.pretrained_models.LungCT_registration_model(
            pretrained=True
        )
        
        icon_registration.test_utils.download_test_data()

        image_exp = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz"
            )
        )
        image_insp = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_INSP_STD_COPD_img.nii.gz"
            )
        )
        image_exp_seg = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_EXP_STD_COPD_label.nii.gz"
            )
        )
        image_insp_seg = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_INSP_STD_COPD_label.nii.gz"
            )
        )

        image_insp_preprocessed = (
            icon_registration.pretrained_models.lung_network_preprocess(
                image_insp, image_insp_seg
            )
        )
        image_exp_preprocessed = (
            icon_registration.pretrained_models.lung_network_preprocess(
                image_exp, image_exp_seg
            )
        )

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            model, image_insp_preprocessed, image_exp_preprocessed, finetune_steps=None
        )

        assert isinstance(phi_AB, itk.CompositeTransform)
        interpolator = itk.LinearInterpolateImageFunction.New(image_insp_preprocessed)

        warped_image_insp_preprocessed = itk.resample_image_filter(
            image_insp_preprocessed,
            transform=phi_AB,
            interpolator=interpolator,
            size=itk.size(image_exp_preprocessed),
            output_spacing=itk.spacing(image_exp_preprocessed),
            output_direction=image_exp_preprocessed.GetDirection(),
            output_origin=image_exp_preprocessed.GetOrigin(),
        )

        # log some images to show the registration
        import os
        os.environ["FOOTSTEPS_NAME"] = "test"
        import footsteps

        plt.imshow(
            np.array(
                itk.checker_board_image_filter(
                    warped_image_insp_preprocessed, image_exp_preprocessed
                )
            )[140]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "grid_lung.png")
        plt.clf()
        plt.imshow(np.array(warped_image_insp_preprocessed)[140])
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "warped_lung.png")
        plt.clf()
        plt.imshow(
            np.array(warped_image_insp_preprocessed)[140]
            - np.array(image_exp_preprocessed)[140]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "difference_lung.png")
        plt.clf()

        insp_points = icon_registration.test_utils.read_copd_pointset(
            "test_files/lung_test_data/copd1_300_iBH_xyz_r1.txt"
        )
        exp_points = icon_registration.test_utils.read_copd_pointset(
            "test_files/lung_test_data/copd1_300_eBH_xyz_r1.txt"
        )
        dists = []
        for i in range(len(insp_points)):
            px, py = (
                exp_points[i],
                np.array(phi_BA.TransformPoint(tuple(insp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))

        self.assertLess(np.mean(dists), 1.5)
        dists = []
        for i in range(len(insp_points)):
            px, py = (
                insp_points[i],
                np.array(phi_AB.TransformPoint(tuple(exp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))
        self.assertLess(np.mean(dists), 2.3)

