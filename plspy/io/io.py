import os
from typing import List, Optional, Tuple, Union

import nibabel
import numpy as np

from .. import exceptions


def open_images_in_dir(
    dir_path: str,
) -> Tuple[List[nibabel.nifti1.Nifti1Image], List[str]]:
    """
    Open all images in a directory and return them as a list.

    Filenames are sorted alphanumerically and Nifti images are loaded
    in using same ordering.

    Parameters
    ----------
    dir_path : str
        Full path to directory containing neuro image files.

    Returns
    -------
    images : List[nibabel.nifti1.Nifti1Image]
        List of images loaded in as Nifti image files.
    filenames : List[str]
        List of file paths of respective loaded images.
    """
    # TODO : check if fpath exists; throw error if not

    # grab all files in directory and omit .hdr files
    # sort alphanumerically
    filenames = sorted(
        [
            f.name
            for f in os.scandir(dir_path)
            if f.is_file() and not f.name.endswith(".hdr")
        ]
    )

    # use nibabel.load to load in each file using list comprehension
    images = [nibabel.load(f"{dir_path}/{fl}") for fl in filenames]

    return images, filenames


def open_single_image_in_dir(fpath: str) -> nibabel.nifti1.Nifti1Image:
    """
    Open single image in a directory and return it.

    Essentially a wrapper for nibabel.load()

    Parameters
    ----------
    fpath : str
        Full path to image file.

    Returns
    -------
    image : nibabel.nifti1.Nifti1Image
        image loaded in as Nifti image files.

    """
    # TODO : check if fpath exists; throw error if not

    # grab sinle file

    image = nibabel.load(f"{fpath}")

    return image


def open_images_from_paths_list(
    fpaths: List[str],
) -> List[nibabel.nifti1.Nifti1Image]:
    """
    Take list of file paths and open each image.

    Returns list of opened images. Sort is unchanged from input.

    Parameters
    ----------
    fpaths : List[str]
        List of full paths to invidual image files.

    Returns
    -------
    images : List[nibabel.nifti1.Nifti1Image]
        list of images loaded in as Nifti image files.

    """
    images = [open_single_image_in_dir(pth) for pth in fpaths]
    return images


def concat_images(
    *args: List[nibabel.nifti1.Nifti1Image], **kwargs: str
) -> nibabel.nifti1.Nifti1Image:
    """
    Use nibabel.concat_images to concatenate separate images.

    For reference, see the nibabel documentation on
    `nibabel.concat_images https://nipy.org/nibabel/reference/nibabel.funcs.html#concat-images`_

        Parameters
        ----------
        *args : tuple
        Any argument passed into nibabel.concat_images. For our purposes it's usually
        one arg of type List[nibabel.nifti1.Nifti1Image]
        **kwargs : dict, optional
        Any keyword arg passed into nibabel.concat_images.

        Returns
        -------
    nibabel.nifti1.Nifti1Image
        Concatenated image composed of original input images.
    """
    return nibabel.concat_images(*args, **kwargs)


def read_dir_to_one_image(
    fpath: str, *args: List[nibabel.nifti1.Nifti1Image], **kwargs: str
) -> nibabel.nifti1.Nifti1Image:
    """
    Load all image files from a directory and concatenate them into
    a single image.

    Use nibabel.concat_images to concatenate separate images.

    For reference, see the nibabel documentation on
    `nibabel.concat_images https://nipy.org/nibabel/reference/nibabel.funcs.html#concat-images`_

        Parameters
        ----------
    fpath : str
        Full path to image file.
        *args : tuple
        Any argument passed into nibabel.concat_images. For our purposes it's usually
        one arg of type List[nibabel.nifti1.Nifti1Image]
        **kwargs : dict, optional
        Any keyword arg passed into nibabel.concat_images.

        Returns
        -------
    nibabel.nifti1.Nifti1Image
        Concatenated image composed of original input images.
    """

    images = open_images_in_dir(fpath)

    single_image = concat_images(images, *args, **kwargs)

    return single_image


def open_multiple_imgs_from_dirs(
    dir_list: List[str], *args: List[nibabel.nifti1.Nifti1Image], **kwargs: str
) -> List[nibabel.nifti1.Nifti1Image]:
    """
    Open multiple directories full of images and return a list of concatenated images.
    Must all be valid filepaths.

    Use nibabel.concat_images to concatenate separate images.

    For reference, see the nibabel documentation on
    `nibabel.concat_images https://nipy.org/nibabel/reference/nibabel.funcs.html#concat-images`_

        Parameters
        ----------
    dir_list : List[str]
        List of full paths to directories of image files.
        *args : tuple
        Any argument passed into nibabel.concat_images. For our purposes it's usually
        one arg of type List[nibabel.nifti1.Nifti1Image]
        **kwargs : dict, optional
        Any keyword arg passed into nibabel.concat_images.

        Returns
        -------
    nibabel.nifti1.Nifti1Image
        Concatenated image composed of original input images.

        Parameters
        ----------
        *args : tuple
        Any argument passed into nibabel.concat_images. For our purposes it's usually
        one arg of type List[nibabel.nifti1.Nifti1Image]
        **kwargs : dict, optional
        Any keyword arg passed into nibabel.concat_images.

        Returns
        -------
    nibabel.nifti1.Nifti1Image
        Concatenated image composed of original input images.
    """

    img_list = []

    for i in range(len(dir_list)):
        img_list.append(read_dir_to_one_image(dir_list[i]))

    return img_list


def extract_single_matrix(img: nibabel.nifti1.Nifti1Image) -> np.ndarray:
    """
    Grab NumPy array from nibabel object.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        image from which to extract NumPy array.

    Returns
    -------
    mat : np.ndarray
        NumPy array extracted from image.

    """

    mat = img.dataobj

    # remove extra single dimension at end if present
    if mat.shape[-1] == 1:
        print(f"Shape before : {mat.shape}")
        img.dataobj = mat.reshape(mat.shape[:-1])
        print(f"Shape after : {mat.shape}")

    return mat


def extract_matrices_from_image_list(
    img_list: List[nibabel.nifti1.Nifti1Image],
) -> List[np.ndarray]:
    """
    Run extract_single_matrix on each item in passed in list and
    return a list of matrices.

    Parameters
    ----------
    img_list : List[nibabel.nifti1.Nifti1Image
        list of nibabel images.

    Returns
    -------
    mats : List[np.ndarray]
        List of extracted NumPy arrays.

    """

    mats = []

    for img in img_list:
        # extract the matrix
        mat = extract_single_matrix(img)
        # squeeze matrix (remove axes with length 1) and append
        mats.append(np.squeeze(mat))

    return mats


def realign_axes_time_first(matrix: np.ndarray) -> np.ndarray:
    """
    Realign matrix axes so that time is the first axis and the rest follow normally

    Parameters
    ----------
    matrix : np.ndarray
        NumPy array to be reshaped.

    Returns
    -------
    matrix_reshaped : np.ndarray
        Reshaped NumPy array with time as first axis.

    """
    # matrix = matrix.transpose(
    #         matrix.shape[3], matrix.shape[:3]
    # )
    matrix_reshaped = np.transpose(matrix, (3, 0, 1, 2))
    return matrix_reshaped


def extract_matrices_image_list_realign(
    img_list: List[nibabel.nifti1.Nifti1Image],
) -> Tuple[List[nibabel.nifti1.Nifti1Image], Tuple[int]]:
    """
    Extract matrices from a list of nibabel image objects and then
    realign the axes so that time is the first axis.

    Parameters
    ----------
    img_list  : List[nibabel.nifti1.Nifti1Image]
        list of nibabel image objects loaded in from

    Returns
    -------
    mats : List[nibabel.nifti1.Nifti1Image]
        list of NumPy arrays containing time-aligned extracted matrices for
        each subject.
    shape : Tuple[int]
        tuple containing shape of single subject's shape, of form
        (time-point, x, y, z).
    """

    mats = extract_matrices_from_image_list(img_list)

    for i in range(len(mats)):
        mats[i] = realign_axes_time_first(mats[i])

    return (mats, mats[0].shape)


def create_binary_mask_from_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Create mask from list of matrices.

    Mask will only include values where, for each index, a value does not
    equal 0 for any subject in the list.

    TODO: finish implementing

    Parameters
    ----------
    matrices : List[np.ndarray]
        matrices to use to create binary mask.

    Returns
    -------
    mask : np.ndarray
        NumPy array mask containing True/False values corresponding to
        each value in the brain space.

    """

    mats = np.array(matrices)

    # reshape to concat all images along time axis
    mats_concat = mats.reshape((-1,) + mats.shape[2:])

    # get the mask
    # mask = np.ma.masked_where(mats_concat, )

    # create mask where values are False if all values throughout all
    # subjects' time series for a particular index are 0
    mask = np.logical_and.reduce(mats_concat, where=(mats_concat != 0), axis=0)

    return mask


def create_threshold_mask_from_matrices(
    matrices: List[np.ndarray], threshold: float = 0.15
) -> np.ndarray:
    """
    Creates mask where only values above the threshold value are included.
    Threshold is user-specifiable but defaults to 0.15.

    Parameters
    ----------
    matrices : List[np.ndarray]
        Matrices to use to create threshold mask.
    threshold : float
        Floating-point value used as cutoff for threshold mask.

    Returns
    -------
    mask : np.ndarray
        Mask filled with True/False values depending on whether or not
        the mean of the values at that index are greater than the
        threshold value.


    """

    if threshold < 0 or threshold > 1:
        raise exceptions.OutOfRangeError(
            f"threshold must be greater than 0 or less than 1. Value passed in : {threshold}"
        )

    mats = np.array(matrices)
    # individual means across all time points for each subject
    mats_time_mean = np.mean(mats, axis=1)
    # global means for all subjects
    mean_all = np.mean(mats_time_mean, axis=0)

    # returns true/false matrix where values are True if they're above the threshold
    # formula : threshold * (max - min) + min
    cond = mean_all > (
        threshold * (np.max(mean_all) - np.min(mean_all)) + np.min(mean_all)
    )

    # get the mask
    mask = np.ma.masked_where(cond, mean_all)

    # return mask.mask
    return mask.mask


def apply_mask_matrices_deprecated(matrices, mask, fill_value):
    """
    This will be removed in future versions of plspy, do not use.

    Parameters
    ----------

    Returns
    -------


    """

    masked = []

    for i in range(len(matrices)):
        # broadcast mask to time axis
        mask_all = np.broadcast_to(mask.mask, matrices[i].shape)
        masked_single = np.ma.MaskedArray(
            data=matrices[i], mask=mask_all, fill_value=fill_value
        )
        masked.append(masked_single.data)

    return masked


def apply_mask_matrices(
    matrices: List[np.ndarray], mask: np.ndarray
) -> List[np.ndarray]:
    """
    Apply a mask to a list of matrices.

    This implementation excludes values that do not meet the mask's
    threshold/binary constraints. It flattens and returns the masked arrays
    as individual arrays in a list.

    Parameters
    ----------
    matrices : List[np.ndarray]
        List of NumPy matrices to mask.
    mask : np.ndarray
        List of True/False values to use for masking matrices.

    Returns
    -------
    masked : List[np.ndarray]
        list of masked NumPy arrays.

    """

    # return np.array([m[mask] for m in matrices])

    masked = []

    for i in range(len(matrices)):
        # broadcast mask to time axis
        mask_all = np.broadcast_to(mask, matrices[i].shape)
        masked.append(matrices[i][mask_all])

    return masked


def create_and_apply_mask_list(
    matrices: List[np.ndarray],
    mask_type: str = "threshold",
    threshold: float = 0.15,
) -> np.ndarray:
    """Creates a mask and applies it to a list of matrices.

    Takes the mask to use, a threshold value if using thresholding, and
    returns masked and stacked neural matrices.


    Parameters
    ----------
    matrices : List[np.ndarray]
        Matrices to use to create threshold mask.
    threshold : float
        Floating-point value used as cutoff for threshold mask.

    Returns
    -------
    masked_array : np.ndarray
        Return flattened, masked matrices in np.ndarra.

    """

    if mask_type == "threshold":
        mask = create_threshold_mask_from_matrices(
            matrices, threshold=threshold
        )
    elif mask_type == "binary":
        pass
    else:
        raise exceptions.NotImplementedError(
            f"Mask type {mask_type} is not implemented."
        )

    return np.array(apply_mask_matrices(matrices, mask))


def open_onsets_txt(filepath: str, tr: float) -> List[np.ndarray]:
    """Open onsets file for each subject from directory path.

    Onset .txts must be in single columns, one for each condition.

    Parameters
    ----------
    filepath  : str
        file path to directory containing onset files
    tr : float
        TR value used to convert from time to slice index

    Returns
    -------
    onsets_inds : List[np.ndarray]
        List of NumPy arrays containing onset arrays
    """

    # grab file paths from directory `filepath`
    files = sorted(
        [
            f.path
            for f in os.scandir(filepath)
            if f.is_file() and f.name.endswith(".txt")
        ]
    )

    # load each onset file into index in list
    onsets = [np.loadtxt(f, dtype=float) for f in files]

    # divide onset by TR, cast to float, and transpose for each subject
    onsets_inds = [np.rint(onset / tr).astype(int).T for onset in onsets]

    return onsets_inds


def extract_onset_slices_single_subject(
    matrix: np.ndarray,
    onsets: np.ndarray,
    onset_length: int,
    tr: float,
    return_indiv: bool = True,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Extract the onsets from a subject's matrix and reorder them according to
    condition.

    Can return slices in concatenated (False) or list (True) form depending
    on value of `return_indiv`.


    Parameters
    ----------
    matrix : np.ndarray
        Matrix to slice using onset info.
    onsets : List[np.ndarray]
        List of condition-based onset times.
    onset_lenth: int
        Duration of time to select after onset.
    tr : float
        Repition time value.
    return_indiv: bool
        True returns a list of invidual slices, False returns a
        concatenated np.ndarray.

    Returns
    -------
    onset_slices_conditions : Union[np.ndarray, List[np.ndarray]]
        Either list of Numpy arrays or NumPy array depending on value
        of return_indiv (concatenated or list form).

    """

    # convert onset duration to number of volumes to select
    num_vols = int(np.rint(onset_length * tr))

    # generate indices by creating ranges from onset time to onset_length
    indices = np.array(
        [
            np.array(
                [
                    np.arange(onsets[i, j], onsets[i, j] + num_vols)
                    for j in range(onsets[i].shape[0])
                ]
            )
            for i in range(onsets.shape[0])
        ]
    )

    # apply indices ranges to matrix to get conditions in list form
    onsets_slices_conditions = [
        matrix[indices[i]].reshape(
            -1, matrix.shape[-3], matrix.shape[-2], matrix.shape[-1]
        )
        for i in range(len(indices))
    ]
    # return concatenated or list form depending on `retur_indiv`
    if not return_indiv:
        return np.array(onsets_slices_conditions)
    else:
        return onsets_slices_conditions


def extract_onset_slices_list(
    matrices: List[np.ndarray],
    onsets: List[np.ndarray],
    onset_length: int,
    tr: float,
    use_one: bool = False,
) -> List[np.ndarray]:
    """
    Extract the onsets from a list of subject matrices and reorder them according to
    condition.


    Parameters
    ----------
    matrix : np.ndarray
        Matrix to slice using onset info.
    onsets : List[np.ndarray]
        List of condition-based onset times.
    onset_lenth: int
        Length of onset slice from start of onset.
    tr : float
        Repition time value.
    use_one : bool
        If True, only use first set of onset values. TODO: allow user to
        specify which index.

    Returns
    -------
    condition_lists : List[np.ndarray]
        List of Numpy arrays where each sliced subject is an index.

    """
    condition_lists = []

    # use first onsets as reference
    if use_one:
        onset = onsets[0]
    for i in range(len(matrices)):
        if not use_one:
            onset = onsets[i]
        # use single extraction function and append to list
        condition_lists.append(
            extract_onset_slices_single_subject(
                matrices[i], onset, onset_length, tr, return_indiv=True
            )
        )
    return condition_lists


def concat_assemble_group(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Take list of matrices with condition sets and turn into single-group
    matrix.

    Parameters
    ----------
    matrices : List[np.ndarray]
        List of NumPy matrices to concatenate into single-group array

    Returns
    -------
    matrices_concat : np.ndarray
        Concatenated, single-group NumPy matrix

    """
    group_list = []

    # concat condition-wise
    for j in range(len(matrices[0])):
        for i in range(len(matrices)):
            group_list.append(matrices[i][j])

    return np.array(group_list)


def concat_flatten_all_groups(groups_list: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate all groups and return as single matrix.

    Parameters
    ----------
    groups_list : List[np.ndarray]
        List of numpy arrays where each index is its own group of subjects.

    Returns
    -------
    full_flat : np.ndarray
        Flattened array of all subjects across all groups/conditions.
        Ready for input to PLS.

    """
    full_unflat = np.concatenate(groups_list, axis=0)
    full_flat = full_unflat.reshape(full_unflat.shape[0], -1)
    return full_flat


def remap_vectorized_subject_to_4d(
    vector: np.ndarray, mask: np.ndarray, original_shape: Tuple[int]
) -> np.ndarray:
    """Remaps indices in the vector to its original, unmasked 4D space.

    Takes masked, vectorized subject and uses the global mask and original
    shape to map the vector indices to its original 4D space. The previously
    masked values are initialized to zeros.

    Parameters
    ----------
    vector : np.ndarray
        1D array/vector containing the masked and vectorized indices for
        a single subject
    mask : np.ndarray
        a 3D global mask with the same dimensions as a single time slice.
        Contains the True/False values for whether not a value should have
        been masked from the original 4D subject.
    original_shape : Tuple[int]
        a list/tuple containing the 4 dimensions from the original subject,
        time first.

    Returns
    -------
    reconstituted : np.ndarray
        4D array with the vector values placed in the original indices from
        the unmasked 4D array.
    """

    # create list of tuples where each tuple corresponds to t,x,y,z coord
    # in original 4D dimension
    # mask == True means that index was added into the vector;
    # this is recovering that information
    vector_indices = np.where(mask == True)

    reconstructed = np.zeros(original_shape)

    # reshape into time slices
    vector_time_sliced = vector.reshape(original_shape[0], -1)

    # iterate over time slices
    for i in range(reconstructed.shape[0]):
        # iterate over sets of indices using 0th list as reference length
        for j in range(len(vector_indices[0])):
            # map index in vector to 4D space
            reconstructed[
                i,
                vector_indices[0][j],
                vector_indices[1][j],
                vector_indices[2][j],
            ] = vector_time_sliced[i][j]

    return reconstructed
