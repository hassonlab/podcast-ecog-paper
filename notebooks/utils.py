import numpy as np

from mne.io import read_raw_fif
from mne_bids import BIDSPath
from nilearn import datasets, image

import pandas as pd
from sklearn.utils import Bunch
from scipy.spatial import KDTree
from nilearn.datasets._utils import (
    get_dataset_descr,
    get_dataset_dir,
    fetch_single_file,
)
import shutil
import os


def get_coord_atlas_labels(
    coords: np.array, atlas_map: str, atlas_labels: list[str]
) -> list[str]:
    atlas_map = image.load_img(atlas_map)
    atlas_image = atlas_map.get_fdata().astype(int)

    # find non-zero labels
    image_label_coords = np.nonzero(atlas_image)

    # transform label indices to MNI space
    atlas_coords = np.vstack(
        image.coord_transform(*image_label_coords, atlas_map.affine)
    ).T

    # find nearest neighbor
    # dists = cdist(
    #     coords.astype(np.float32), atlas_coords.astype(np.float32), metric="euclidean"
    # )
    # nearest_neighbor = dists.argmin(-1)
    tree = KDTree(atlas_coords)
    dists, nearest_neighbor = tree.query(coords, k=1)

    # look up neighbor index in map
    x = image_label_coords[0][nearest_neighbor]
    y = image_label_coords[1][nearest_neighbor]
    z = image_label_coords[2][nearest_neighbor]

    # convert map index to label
    elec_label_ids = atlas_image[x, y, z]
    elec_labels = [atlas_labels[i] for i in elec_label_ids]

    return elec_labels


def fetch_atlas_glasser_2016(
    data_dir=None,
    url=None,
    resume=True,
    verbose=1,
):
    # https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911?file=5534027
    # https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911?file=5594360
    atlas_file_number = "5594360"
    labels_file_number = "5534027"
    url = "https://ndownloader.figshare.com/files/{}"

    dataset_name = "glasser_2016"
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    files = [
        ("glasser2016_rois.txt", url.format(labels_file_number), {}),
        ("glasser2016_map.nii.gz", url.format(atlas_file_number), {}),
    ]

    files_ = []
    for filename, file_url, file_opts in files:
        move_fn = os.path.join(data_dir, filename)
        if not os.path.isfile(move_fn):
            data_fn = fetch_single_file(
                file_url, data_dir, resume=resume, verbose=verbose, **file_opts
            )
            shutil.move(data_fn, move_fn)
        data_fn = move_fn
        files_.append(data_fn)

    # this doesn't allow us to move files easily.
    # files_ = fetch_files(data_dir, files, resume=resume, verbose=verbose)

    fdescr = get_dataset_descr(dataset_name)
    labels = pd.read_csv(files_[0], header=None, sep=" ", names=["label"], index_col=0)
    params = dict(maps=files_[1], labels=labels, description=fdescr)

    return Bunch(**params)


def get_sub_coords(scale: int = 1000):
    edf_path = BIDSPath(
        root="../../monkey/derivatives/ecogprep",
        datatype="ieeg",
        description="highgamma",
        extension=".fif",
    )

    sub_coords = []
    for raw_fif in edf_path.match():
        raw = read_raw_fif(raw_fif, verbose=False)
        ch2loc = {ch["ch_name"]: ch["loc"][:3] for ch in raw.info["chs"]}
        coords = np.vstack([ch2loc[ch] for ch in raw.info["ch_names"]]) * scale

        sub_coords.append(coords)

    coords = np.vstack(sub_coords)
    return coords


def generate_null_dist(n_perms: int = 10000, dim_size: int = 2384, seed: int = None):
    rng = np.random.default_rng(seed=seed)
    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        a = rng.normal(size=dim_size)
        b = rng.normal(size=dim_size)
        null_dist[i] = np.corrcoef(a, b)[0, 1]
    return null_dist


def calculate_pvalues(
    observed: np.ndarray,
    null_distribution: np.ndarray,
    alternative: str = "two-sided",
    adjustment: int = 1,
) -> np.ndarray:
    """Calculate p-value
    See https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_resampling.py#L1133-L1602
    """
    n_resamples = len(null_distribution)

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=1) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=1) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less, "greater": greater, "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return pvalues
