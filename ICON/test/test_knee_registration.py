import unittest


class TestKneeRegistration(unittest.TestCase):
    def test_knee_registration(self):
        print("OAI ICON")

        import icon_registration.pretrained_models
        from icon_registration.mermaidlite import compute_warped_image_multiNC
        from icon_registration.losses import flips

        import torch
        import numpy as np

        import subprocess

        print("Downloading test data)")
        import icon_registration.test_utils

        icon_registration.test_utils.download_test_data()
        t_ds = torch.load(
            icon_registration.test_utils.TEST_DATA_DIR / "icon_example_data"
        )
        batched_ds = list(zip(*[t_ds[i::2] for i in range(2)]))
        net = icon_registration.pretrained_models.OAI_knees_registration_model(
            pretrained=True
        )
        # Run on the four downloaded image pairs

        with torch.no_grad():
            dices = []
            folds_list = []

            for x in batched_ds[:]:
                # Seperate the image data used for registration from the segmentation used for evaluation,
                # and shape it for passing to the network
                x = list(zip(*x))
                x = [torch.cat(r, 0).cuda().float() for r in x]
                fixed_image, fixed_cartilage = x[0], x[2]
                moving_image, moving_cartilage = x[1], x[3]

                # Run the registration.
                # Our network expects batches of two pairs,
                # moving_image.size = torch.Size([2, 1, 80, 192, 192])
                # fixed_image.size = torch.Size([2, 1, 80, 192, 192])
                # intensity normalized to have min 0 and max 1.

                net(moving_image, fixed_image)

                # Once registration is run, net.phi_AB and net.phi_BA are functions that map
                # tensors of coordinates from image B to A and A to B respectively.

                # Evaluate the registration
                # First, evaluate phi_AB on a tensor of coordinates to get an explicit map.
                phi_AB_vectorfield = net.phi_AB(net.identity_map)
                fat_phi = torch.nn.Upsample(
                    size=moving_cartilage.size()[2:],
                    mode="trilinear",
                    align_corners=False,
                )(phi_AB_vectorfield[:, :3])
                sz = np.array(fat_phi.size())
                spacing = 1.0 / (sz[2::] - 1)

                # Warp the cartilage of one image to match the other using the explicit map.
                warped_moving_cartilage = compute_warped_image_multiNC(
                    moving_cartilage.float(), fat_phi, spacing, 1
                )

                # Binarize the segmentations
                wmb = warped_moving_cartilage > 0.5
                fb = fixed_cartilage > 0.5

                # Compute the dice metric
                intersection = wmb * fb
                dice = (
                    2
                    * torch.sum(intersection, [1, 2, 3, 4]).float()
                    / (torch.sum(wmb, [1, 2, 3, 4]) + torch.sum(fb, [1, 2, 3, 4]))
                )
                print("Batch DICE:", dice)
                dices.append(dice)

                # Compute the folds metric
                f = [flips(phi[None]).item() for phi in phi_AB_vectorfield]
                print("Batch folds per image:", f)
                folds_list.append(f)

            mean_dice = torch.mean(torch.cat(dices).cpu())
            print("Mean DICE SCORE:", mean_dice)
            self.assertTrue(mean_dice.item() > 0.68)
            mean_folds = np.mean(folds_list)
            print("Mean folds per image:", mean_folds)
            self.assertTrue(mean_folds < 300)

    def test_knee_registration_gradICON(self):
        print("OAI gradICON")

        import icon_registration.pretrained_models
        from icon_registration.mermaidlite import compute_warped_image_multiNC
        from icon_registration.losses import flips

        import torch
        import numpy as np

        import subprocess

        print("Downloading test data)")
        import icon_registration.test_utils

        icon_registration.test_utils.download_test_data()
        t_ds = torch.load(
            icon_registration.test_utils.TEST_DATA_DIR / "icon_example_data"
        )
        batched_ds = list(zip(*[t_ds[i::2] for i in range(2)]))
        net = icon_registration.pretrained_models.OAI_knees_gradICON_model(
            pretrained=True
        )
        # Run on the four downloaded image pairs

        with torch.no_grad():
            dices = []
            folds_list = []

            for x in batched_ds[:]:
                # Seperate the image data used for registration from the segmentation used for evaluation,
                # and shape it for passing to the network
                x = list(zip(*x))
                x = [torch.cat(r, 0).cuda().float() for r in x]
                fixed_image, fixed_cartilage = x[0], x[2]
                moving_image, moving_cartilage = x[1], x[3]

                # Run the registration.
                # Our network expects batches of two pairs,
                # moving_image.size = torch.Size([2, 1, 80, 192, 192])
                # fixed_image.size = torch.Size([2, 1, 80, 192, 192])
                # intensity normalized to have min 0 and max 1.

                net(moving_image, fixed_image)

                # Once registration is run, net.phi_AB and net.phi_BA are functions that map
                # tensors of coordinates from image B to A and A to B respectively.

                # Evaluate the registration
                # First, evaluate phi_AB on a tensor of coordinates to get an explicit map.
                phi_AB_vectorfield = net.phi_AB(net.identity_map)
                fat_phi = torch.nn.Upsample(
                    size=moving_cartilage.size()[2:],
                    mode="trilinear",
                    align_corners=False,
                )(phi_AB_vectorfield[:, :3])
                sz = np.array(fat_phi.size())
                spacing = 1.0 / (sz[2::] - 1)

                # Warp the cartilage of one image to match the other using the explicit map.
                warped_moving_cartilage = compute_warped_image_multiNC(
                    moving_cartilage.float(), fat_phi, spacing, 1
                )

                # Binarize the segmentations
                wmb = warped_moving_cartilage > 0.5
                fb = fixed_cartilage > 0.5

                # Compute the dice metric
                intersection = wmb * fb
                dice = (
                    2
                    * torch.sum(intersection, [1, 2, 3, 4]).float()
                    / (torch.sum(wmb, [1, 2, 3, 4]) + torch.sum(fb, [1, 2, 3, 4]))
                )
                print("Batch DICE:", dice)
                dices.append(dice)

                # Compute the folds metric
                f = [flips(phi[None]).item() for phi in phi_AB_vectorfield]
                print("Batch folds per image:", f)
                folds_list.append(f)

            mean_dice = torch.mean(torch.cat(dices).cpu())
            print("Mean DICE SCORE:", mean_dice)
            self.assertTrue(mean_dice.item() > 0.68)
            mean_folds = np.mean(folds_list)
            print("Mean folds per image:", mean_folds)
            self.assertTrue(mean_folds < 300)
