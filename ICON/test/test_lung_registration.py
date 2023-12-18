import os
import unittest

import itk
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import icon_registration
import icon_registration.pretrained_models

COPD_spacing = {
    "copd1": [0.625, 0.625, 2.5],
    "copd2": [0.645, 0.645, 2.5],
    "copd3": [0.652, 0.652, 2.5],
    "copd4": [0.590, 0.590, 2.5],
    "copd5": [0.647, 0.647, 2.5],
    "copd6": [0.633, 0.633, 2.5],
    "copd7": [0.625, 0.625, 2.5],
    "copd8": [0.586, 0.586, 2.5],
    "copd9": [0.664, 0.664, 2.5],
    "copd10": [0.742, 0.742, 2.5],
}


def readPoint(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """
    with open(f_path) as fp:
        content = fp.read().split("\n")

        # Read number of points from second
        count = len(content) - 1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float64)

        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split("\t")
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])

        return points


def calc_warped_points(source_list_t, phi_t, dim, spacing, phi_spacing):
    """
    :param source_list_t: source image.
    :param phi_t: the inversed displacement. Domain in source coordinate.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :return: a N*3 tensor containg warped positions in the physical coordinate.
    """
    warped_list_t = F.grid_sample(phi_t, source_list_t, align_corners=True)

    warped_list_t = torch.flip(warped_list_t.permute(0, 2, 3, 4, 1), [4])[0, 0, 0]
    warped_list_t = torch.mul(
        torch.mul(warped_list_t, torch.from_numpy(dim - 1.0)),
        torch.from_numpy(phi_spacing),
    )

    return warped_list_t


def eval_with_data(source_list, target_list, phi, dim, spacing, origin, phi_spacing):
    """
    :param source_list: a numpy list of markers' position in source image.
    :param target_list: a numpy list of markers' position in target image.
    :param phi: displacement map in numpy format.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :param return: res, [dist_x, dist_y, dist_z] res is the distance between
    the warped points and target points in MM. [dist_x, dist_y, dist_z] are
    distances in MM along x,y,z axis perspectively.
    """
    origin_list = np.repeat(
        [
            origin,
        ],
        target_list.shape[0],
        axis=0,
    )

    # Translate landmark from landmark coord to phi coordinate
    target_list_t = (
        torch.from_numpy((target_list - 1.0) * spacing) - origin_list * phi_spacing
    )
    source_list_t = (
        torch.from_numpy((source_list - 1.0) * spacing) - origin_list * phi_spacing
    )

    # Pay attention to the definition of align_corners in grid_sampling.
    # Translate landmarks to voxel index in image space [-1, 1]
    source_list_norm = source_list_t / phi_spacing / (dim - 1.0) * 2.0 - 1.0
    source_list_norm = source_list_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    phi_t = torch.from_numpy(phi).double()

    warped_list_t = calc_warped_points(
        source_list_norm, phi_t, dim, spacing, phi_spacing
    )

    pdist = torch.nn.PairwiseDistance(p=2)
    dist = pdist(target_list_t, warped_list_t)
    idx = torch.argsort(dist).numpy()
    dist_x = torch.mean(torch.abs(target_list_t[:, 0] - warped_list_t[:, 0])).item()
    dist_y = torch.mean(torch.abs(target_list_t[:, 1] - warped_list_t[:, 1])).item()
    dist_z = torch.mean(torch.abs(target_list_t[:, 2] - warped_list_t[:, 2])).item()
    res = torch.mean(dist).item()

    return res, [dist_x, dist_y, dist_z]


def compute_dice(x, y):
    eps = 1e-11
    y_loc = set(np.where(y.flatten() == 1)[0])
    x_loc = set(np.where(x.flatten() == 1)[0])
    # iou
    intersection = set.intersection(x_loc, y_loc)
    # recall
    len_intersection = len(intersection)
    tp = float(len_intersection)
    fn = float(len(y_loc) - len_intersection)
    fp = float(len(x_loc) - len_intersection)

    if len(y_loc) != 0 or len(x_loc) != 0:
        return 2 * tp / (2 * tp + fn + fp + eps)

    return 0.0


def eval_copd_highres(seg_folder, case_id, phi, phi_inv, phi_spacing, origin, dim):
    """
    :param dataset_path: the path to the dataset folder. The folder structure assumption:
    dataset_path/landmarks stores landmarks files; dataset_path/segments stores segmentation maps

    :param case_id: a list of case id
    :param phi: phi defined in expiration image domain. Numpy array. Bx3xDxWxH. Orientation: SI, AP, RL orientation
    :param phi_inv: phi_inv defined in inspiration image domain. Numpy array. Bx3xDxWxH
    :param phi_spacing: physical spacing of the domain of phi. Numpy array. Bx3xDxWxH
    """

    def _eval(phi, case_id, phi_spacing, origin, dim, inv=False):
        result = {}
        if inv:
            source_file = os.path.join(seg_folder, f"{case_id}_300_iBH_xyz_r1.txt")
            target_file = os.path.join(seg_folder, f"{case_id}_300_eBH_xyz_r1.txt")
        else:
            source_file = os.path.join(seg_folder, f"{case_id}_300_eBH_xyz_r1.txt")
            target_file = os.path.join(seg_folder, f"{case_id}_300_iBH_xyz_r1.txt")

        spacing = COPD_spacing[case_id]

        source_list = readPoint(source_file)
        target_list = readPoint(target_file)

        mTRE, mTRE_seperate = eval_with_data(
            source_list, target_list, phi, dim, spacing, origin, phi_spacing
        )

        result["mTRE"] = mTRE
        result["mTRE_X"] = mTRE_seperate[0]
        result["mTRE_Y"] = mTRE_seperate[1]
        result["mTRE_Z"] = mTRE_seperate[2]

        return result

    results = {}
    for i, case in enumerate(case_id):
        results[case] = _eval(
            phi[i : i + 1], case, phi_spacing[i], origin[i], dim[i], inv=False
        )
        results[f"{case}_inv"] = _eval(
            phi_inv[i : i + 1], case, phi_spacing[i], origin[i], dim[i], inv=True
        )
    return results


class TestLungRegistration(unittest.TestCase):
    def test_lung_registration(self):
        print("lung gradICON")

        net = icon_registration.pretrained_models.LungCT_registration_model(
            pretrained=True
        )
        icon_registration.test_utils.download_test_data()

        root = str(icon_registration.test_utils.TEST_DATA_DIR / "lung_test_data")
        cases = [f"copd{i}_highres" for i in range(1, 2)]
        hu_clip_range = [-1000, 0]

        # Load data
        def process(iA, isSeg=False):
            iA = iA[None, None, :, :, :]
            if isSeg:
                iA = iA.float()
                iA = torch.nn.functional.max_pool3d(iA, 2)
                iA[iA > 0] = 1
            else:
                iA = torch.clip(iA, hu_clip_range[0], hu_clip_range[1])
                iA = iA / 1000

                iA = torch.nn.functional.avg_pool3d(iA, 2)
            return iA

        def read_itk(path):
            img = itk.imread(path)
            return torch.tensor(np.asarray(img)), np.flipud(list(img.GetOrigin()))

        dirlab = []
        dirlab_seg = []
        dirlab_origin = []
        for name in tqdm.tqdm(list(iter(cases))[:]):
            image_insp, _ = read_itk(f"{root}/{name}_INSP_STD_COPD_img.nii.gz")
            image_exp, _ = read_itk(f"{root}/{name}_EXP_STD_COPD_img.nii.gz")
            seg_insp, _ = read_itk(f"{root}/{name}_INSP_STD_COPD_label.nii.gz")
            seg_exp, origin = read_itk(f"{root}/{name}_EXP_STD_COPD_label.nii.gz")

            # dirlab.append((process(image_insp), process(image_exp)))
            dirlab.append(
                (
                    ((process(image_insp) + 1) * process(seg_insp, True)),
                    ((process(image_exp) + 1) * process(seg_exp, True)),
                )
            )
            dirlab_seg.append((process(seg_insp, True), process(seg_exp, True)))
            dirlab_origin.append(origin)

        def make_batch(dataset, dataset_seg):
            image_A = torch.cat([p[0] for p in dataset]).cuda()
            image_B = torch.cat([p[1] for p in dataset]).cuda()

            image_A_seg = torch.cat([p[0] for p in dataset_seg]).cuda()
            image_B_seg = torch.cat([p[1] for p in dataset_seg]).cuda()
            return image_A, image_B, image_A_seg, image_B_seg

        net.cuda()

        train_A, train_B, train_A_seg, train_B_seg = make_batch(dirlab, dirlab_seg)

        phis = []
        phis_inv = []
        warped_A = []
        for i in range(train_A.shape[0]):
            with torch.no_grad():
                print(net(train_A[i : i + 1], train_B[i : i + 1]))

            phis.append((net.phi_AB_vectorfield.detach() * 2.0 - 1.0))
            phis_inv.append((net.phi_BA_vectorfield.detach() * 2.0 - 1.0))
            warped_A.append(net.warped_image_A[:, 0:1].detach())

        warped_A = torch.cat(warped_A)

        phis_np = (torch.cat(phis).cpu().numpy() + 1.0) / 2.0
        phis_inv_np = (torch.cat(phis_inv).cpu().numpy() + 1.0) / 2.0
        res = eval_copd_highres(
            root,
            [c[:-8] for c in cases],
            phis_np,
            phis_inv_np,
            np.repeat(np.array([[1.0, 1.0, 1.0]]), len(cases), axis=0),
            np.array([np.flipud(i) for i in dirlab_origin]),
            np.repeat(np.array([[350, 350, 350]]), len(cases), axis=0),
        )

        results = []
        results_inv = []
        for k, v in res.items():
            result = []
            for m, n in v.items():
                result.append(n)
            if "_inv" in k:
                results_inv.append(result)
            else:
                results.append(result)
        results = np.array(results)

        results_str = f"mTRE: {results[:,0].mean()}, mTRE_X: {results[:,1].mean()}, mTRE_Y: {results[:,2].mean()}, mTRE_Z: {results[:,3].mean()}"
        print(results_str)
        results_inv = np.array(results_inv)
        results_inv_str = f"mTRE: {results_inv[:,0].mean()}, mTRE_X: {results_inv[:,1].mean()}, mTRE_Y: {results_inv[:,2].mean()}, mTRE_Z: {results_inv[:,3].mean()}"
        print(results_inv_str)

        # Compute Dice

        warped_train_A_seg = F.grid_sample(
            train_A_seg.float(),
            torch.cat(phis).flip([1]).permute([0, 2, 3, 4, 1]),
            padding_mode="zeros",
            mode="nearest",
            align_corners=True,
        )
        dices = []
        for i in range(warped_train_A_seg.shape[0]):
            dices.append(
                compute_dice(
                    train_B_seg[i].cpu().numpy(),
                    warped_train_A_seg[i].detach().cpu().numpy(),
                )
            )
        print(np.array(dices).mean())
