import random
from glob import glob

import cc3d
import nrrd
import numpy as np


def bbox_cal(data):
    a = np.round(data)
    x0 = np.sum(a, axis=2)
    xx = np.sum(x0, axis=1)
    yy = np.sum(x0, axis=0)
    resx = next(x for x, val in enumerate(list(xx)) if val > 0)

    resxx = next(x for x, val in enumerate(list(xx)[::-1]) if val > 0)

    resy = next(x for x, val in enumerate(list(yy)) if val > 0)

    resyy = next(x for x, val in enumerate(list(yy)[::-1]) if val > 0)
    z0 = np.sum(a, axis=1)
    zz = np.sum(z0, axis=0)
    resz = next(x for x, val in enumerate(list(zz)) if val > 0)

    reszz = next(x for x, val in enumerate(list(zz)[::-1]) if val > 0)

    return resx, resxx, resy, resyy, resz, reszz


data_list = glob("{}/*.nrrd".format("./complete_skull/"))
defected_dir = "./defects_facial/"


for i in range(len(data_list)):
    a, h = nrrd.read(data_list[i])

    x_s, y_s, z_s = a.shape
    resx, resxx, resy, resyy, resz, reszz = bbox_cal(a)
    x_extend = random.randint(resx, resx + (x_s - resxx - resx))
    y_extend = random.randint(resy, resy + (y_s - resyy - resy))
    z_extend = random.randint(resz, resz + (z_s - reszz - resz))
    a[resx : resx + x_extend, resy : resy + 98, resz : z_s - reszz] = 0
    fname1 = defected_dir + data_list[i][-10:-5] + ".nrrd"
    nrrd.write(fname1, a, h)


"""

from pathlib import Path
import shutil
pathlist = Path('./9616319/0_labelsTr').glob('**/*.nrrd')
for path in pathlist:
     # because path is object not string
     path_in_str = str(path)
     shutil.copyfile(path_in_str, './complete_nrrds/'+path_in_str[-10:-5]+'.nrrd')
     print(path_in_str)
"""
