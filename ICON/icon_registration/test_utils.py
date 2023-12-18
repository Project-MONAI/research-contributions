import pathlib
import subprocess
import sys
import numpy as np

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "test_files"


def download_test_data():
    subprocess.run(
        [
            "girder-client",
            "--api-url",
            "https://data.kitware.com/api/v1",
            "localsync",
            "61d3a99d4acac99f429277d7",
            str(TEST_DATA_DIR),
        ],
        #stdout=sys.stdout,
    )


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


def read_copd_pointset(f_path):
    """Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.

    :param f_path: the path to the file containing
        the position of points from copdgene dataset.
    :return: numpy array of points in physical coordinates
    """
    spacing = COPD_spacing[f_path.split("/")[-1].split("_")[0]]
    spacing = np.expand_dims(spacing, 0)
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

        # The copd gene points are in index space instead of physical space.
        # Move them to physical space.
        return (points - 1) * spacing
