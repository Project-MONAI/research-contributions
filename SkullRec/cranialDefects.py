import random
from glob import glob

import nrrd
import numpy as np
from scipy.ndimage import zoom


def generate_hole_implants(data, cube_dim):
    x_ = data.shape[0]
    y_ = data.shape[1]
    z_ = data.shape[2]
    full_masking = np.ones(shape=(x_, y_, z_))
    x = random.randint(int(cube_dim / 2), x_ - int(cube_dim / 2))
    y = random.randint(int(cube_dim / 2), y_ - int(cube_dim / 2))
    z = int(z_ * (3 / 4))
    cube_masking = np.zeros(shape=(cube_dim, cube_dim, z_ - z))
    print(cube_masking.shape)
    full_masking[
        x - int(cube_dim / 2) : x + int(cube_dim / 2), y - int(cube_dim / 2) : y + int(cube_dim / 2), z:z_
    ] = cube_masking
    return full_masking


def generate_cude(size):
    for i in range(len(pair_list)):
        print("generating data:", pair_list[i])
        temp, header = nrrd.read(pair_list[i])

        full_masking = generate_hole_implants(temp, size)

        c_masking_1 = full_masking == 1
        c_masking_1 = c_masking_1 + 1 - 1

        defected_image = c_masking_1 * temp

        c_masking = full_masking == 0
        c_masking = c_masking + 1 - 1
        f1 = defected_dir + pair_list[i][-10:-5] + ".nrrd"
        f2 = implant_dir + pair_list[i][-10:-5] + ".nrrd"
        nrrd.write(f1, defected_image, header)


if __name__ == "__main__":
    pair_list = glob("{}/*.nrrd".format("./complete_skull/"))

    defected_dir = "./defects_cranial/"
