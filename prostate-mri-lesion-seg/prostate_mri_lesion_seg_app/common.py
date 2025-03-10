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

import numpy as np
import copy
from skimage.measure import label

###############################################################################
def bounding_box_3d(nda):

    r = np.any(nda, axis=(1, 2))
    c = np.any(nda, axis=(0, 2))
    z = np.any(nda, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # values are the indices with non-zero entries
    bbox = np.zeros(shape=(6,), dtype=np.int16)
    bbox[0] = rmin
    bbox[1] = rmax
    bbox[2] = cmin
    bbox[3] = cmax
    bbox[4] = zmin
    bbox[5] = zmax

    return bbox

def crop_pos_classification_multi_channel_3d(nda, nda_gt, crop_size):
    if nda_gt.ndim == 4:
        nda_ref = np.amax(nda_gt, axis=0)
    elif nda_gt.ndim == 3:
        nda_ref = copy.deepcopy(nda_gt)

    if np.amax(nda_gt) < 1:
        return nda, []

    label_image = label(nda_gt)
    num_crop = len(np.unique(label_image)) - 1

    if num_crop == 1:
        bbox = bounding_box_3d(label_image == 1)

        indices = np.zeros(shape=(3,), dtype=np.int16)
        for j in range(3):
            indices[j] = 0.5 * (bbox[2 * j] + bbox[2 * j + 1])
            indices[j] = indices[j] - int(float(crop_size[j]) / 2.0)
            indices[j] = np.maximum(indices[j], 0)
            indices[j] = np.minimum(indices[j], nda_ref.shape[j] - crop_size[j])

        nda = nda[
            ...,
            indices[0]:indices[0] + crop_size[0],
            indices[1]:indices[1] + crop_size[1],
            indices[2]:indices[2] + crop_size[2]
            ]

        gt = np.unique(nda_gt[label_image==1])[0]
    elif num_crop > 1:
        images = []
        labels = []
        for k in range(num_crop):
            bbox = bounding_box_3d(label_image == k + 1)

            indices = np.zeros(shape=(3,), dtype=np.int16)
            for j in range(3):
                indices[j] = 0.5 * (bbox[2 * j] + bbox[2 * j + 1])
                indices[j] = indices[j] - int(float(crop_size[j]) / 2.0)
                indices[j] = np.maximum(indices[j], 0)
                indices[j] = np.minimum(indices[j], nda_ref.shape[j] - crop_size[j])

            nda_crop = nda[
                ...,
                indices[0]:indices[0] + crop_size[0],
                indices[1]:indices[1] + crop_size[1],
                indices[2]:indices[2] + crop_size[2]
                ]

            gt = np.unique(nda_gt[label_image == k + 1])[0]

            images.append(nda_crop)
            labels.append(gt)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        return images, labels

    return nda, gt

def standard_normalization_multi_channel(nda):
    for _i in range(nda.shape[0]):
        if np.amax(np.abs(nda[_i, ...])) < 1e-7:
            continue
        nda[_i, ...] = (nda[_i, ...] - np.mean(nda[_i, ...])) / np.std(nda[_i, ...])

    return nda
