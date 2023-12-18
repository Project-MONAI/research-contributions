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
        print("brain GradICON")
        import os

        os.environ["FOOTSTEPS_NAME"] = "test"
        import footsteps

        icon_registration.test_utils.download_test_data()

        model = icon_registration.pretrained_models.brain_registration_model(
            pretrained=True
        )

        image_A = itk.imread(
            f"{icon_registration.test_utils.TEST_DATA_DIR}/brain_test_data/2_T1w_acpc_dc_restore_brain.nii.gz"
        )

        image_B = itk.imread(
            f"{icon_registration.test_utils.TEST_DATA_DIR}/brain_test_data/8_T1w_acpc_dc_restore_brain.nii.gz"
        )

        image_A_processed = icon_registration.pretrained_models.brain_network_preprocess(
            image_A
        )

        image_B_processed = icon_registration.pretrained_models.brain_network_preprocess(
            image_B
        )

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            model, image_A_processed, image_B_processed
        )

        assert isinstance(phi_AB, itk.CompositeTransform)
        interpolator = itk.LinearInterpolateImageFunction.New(image_A)

        warped_image_A = itk.resample_image_filter(
            image_A_processed,
            transform=phi_AB,
            interpolator=interpolator,
            size=itk.size(image_B),
            output_spacing=itk.spacing(image_B),
            output_direction=image_B.GetDirection(),
            output_origin=image_B.GetOrigin(),
        )

        plt.imshow(
            np.array(itk.checker_board_image_filter(warped_image_A, image_B_processed))[40]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "grid.png")
        plt.clf()
        plt.imshow(np.array(warped_image_A)[40])
        plt.savefig(footsteps.output_dir + "warped.png")
        plt.clf()


        reference = np.load(icon_registration.test_utils.TEST_DATA_DIR / "brain_test_data/2_and_8_warped_itkfix.npy")       

        np.save(
            footsteps.output_dir + "warped_brain.npy",
            itk.array_from_image(warped_image_A)[40],
        )

        self.assertLess(
            np.mean(np.abs(reference - itk.array_from_image(warped_image_A)[40])), 1e-5
        )
