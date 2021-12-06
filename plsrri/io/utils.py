import os
import nibabel as nib
import numpy as np

from .. import exceptions


def open_images_in_dir(fpath):
    """
    Open all images in a directory and return them as a list.
    
    TODO: add more documentation as this evolves
    """
    # TODO: check if fpath exists; throw error if not

    # grab all files in directory and omit headers
    files = [
        f.name for f in os.scandir(fpath) if f.is_file() and not f.name.endswith(".hdr")
    ]

    images = [nib.load(f"{fpath}/{fl}") for fl in files]

    return images


def open_single_image_in_dir(fpath):
    """
    Open single image in a directory and return it.
    
    TODO: add more documentation as this evolves
    """
    # TODO: check if fpath exists; throw error if not

    # grab sinle file

    image = nib.load(f"{fpath}")

    return image


def concat_images(*args, **kwargs):
    """
    Use nibabel.concat_images to concatenate separate images
    """
    return nib.concat_images(*args, **kwargs)


def read_dir_to_one_image(fpath, *args, **kwargs):
    """
    Load all image files from a directory and concatenate them int
    a single image
    """

    images = open_images_in_dir(fpath)

    single_image = concat_images(images, *args, **kwargs)

    return single_image


def open_multiple_imgs_from_dirs(dir_list, *args, **kwargs):
    """
    Open multiple directories full of images and return a list of concatenated images.
    Must all be valid filepaths.
    """

    img_list = []

    for i in range(len(dir_list)):
        img_list.append(read_dir_to_one_image(dir_list[i]))

    return img_list


def extract_single_matrix(img):
    """
    Grab NumPy array from nibabel object
    """

    mat = img.dataobj

    if mat.shape[-1] == 1:
        print(f"Shape before: {mat.shape}")
        images[i].dataobj = mat.reshape(mat.shape[:-1])
        print(f"Shape after: {mat.shape}")

    return mat


def extract_matrices_from_image_list(img_list):
    """
    Run extract_single_matrix on each item in passed in list and
    return a list of matrices.
    """

    mats = []

    for img in img_list:
        # extract the matrix
        mat = extract_single_matrix(img)
        # squeeze matrix (remove axes with length 1) and append
        mats.append(np.squeeze(mat))

    return mats


def realign_axes_time_first(matrix):
    """
    Realign matrix axes so that time is the first axis and the rest follow normally
    """
    # matrix = matrix.transpose(
    #         matrix.shape[3], matrix.shape[:3]
    # )
    matrix = np.transpose(matrix, (3, 0, 1, 2))
    return matrix


def extract_matrices_image_list_realign(img_list):
    """
    Extract matrices from a list of nibabel image objects and then
    realign the axes so that time is the first axis.

    Parameters
    ----------
    img_list : list
        list of nibabel image objects loaded in from 

    Returns
    -------
    mats : list
        list of NumPy arrays containing time-aligned extracted matrices for each subject
    shape : tuple
        tuple containing shape of single subject's shape, of form 
        (time-point, x, y, z)
    """

    mats = extract_matrices_from_image_list(img_list)

    for i in range(len(mats)):
        mats[i] = realign_axes_time_first(mats[i])

    return (mats, mats[0].shape)


def create_binary_mask_from_matrices(matrices):
    """

    """

    mats = np.array(matrices)

    # reshape to concat all images along time axis
    mats_concat = mats.reshape((-1,) + mats.shape[2:])

    # get the mask
    mask = np.ma.masked_where(cond, mean_all)

    return mask.mask


def create_threshold_mask_from_matrices(matrices, threshold=0.15):
    """

    """

    if threshold < 0 or threshold > 1:
        raise exceptions.OutOfRangeError(
            f"threshold must be greater than 0 or less than 1. Value passed in: {threshold}"
        )

    mats = np.array(matrices)
    # individual means across all time points for each subject
    mats_time_mean = np.mean(mats, axis=1)
    # global means for all subjects
    mean_all = np.mean(mats_time_mean, axis=0)

    # returns true/false matrix where values are True if they're above the threshold
    # formula: threshold * (max - min) + min
    cond = mean_all > (
        threshold * (np.max(mean_all) - np.min(mean_all)) + np.min(mean_all)
    )

    # get the mask
    mask = np.ma.masked_where(cond, mean_all)

    return mask.mask


def apply_mask_to_matrix(matrix, mask):
    """

    """
    return matrix[mask]


def apply_mask_matrices(matrices, mask):
    """

    """

    # return np.array([m[mask] for m in matrices])

    masked = []

    for i in range(len(matrices)):
        # broadcast mask to time axis
        mask_all = np.broadcast_to(mask, matrices[i].shape)
        masked.append(matrices[i][mask_all])

    return masked


def create_and_apply_mask_list(matrices, mask_type="threshold", threshold=0.15):
    """

    """

    if mask_type == "threshold":
        mask = create_threshold_mask_from_matrices(matrices, threshold=threshold)
    elif mask_type == "binary":
        pass
    else:
        raise exceptions.NotImplementedError(
            f"Mask type {mask_type} is not implemented."
        )

    return np.array(apply_mask_matrices(matrices, mask))


def open_onsets_txt(filepath, onset_length):
    """

    """

    onsets = []

    f = open(filepath, "r")
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        # complicated one-liner that strips newlines, splits values on
        # spaces, and casts list elements to integers
        onsets.append(list(map(int, lines[i].strip("\n").split(" "))))

    return onsets


def extract_onset_slices_single_subject(
    matrix, onsets, onset_length, return_indiv=False
):
    """
    Extract the onsets from a subject's matrix and reorder them according to
    condition.

    """

    onset_slices_conditions = []

    for i in range(len(onsets)):
        # absurd one-liner i cooked up that grabs onset slices from a single
        # subject for each condition and flattens it into one NumPy array
        onset_slices_conditions.append(
            np.array([matrix[m : m + onset_length] for m in onsets[i]]).reshape(
                -1, *matrix.shape[1:]
            )
        )

    # return concatenated or list form depending on `retur_indiv`
    if not return_indiv:
        onsets_slices_concat = np.array(onsets_slices_conditions, axis=0)
        return onsets_slices_concat
    else:
        return onset_slices_conditions


def extract_onset_slices_single_subject_no_concat(
    matrix, onsets, onset_length, return_flat=True
):
    """
    """
    onset_slices_conditions = []

    for i in range(len(onsets)):
        # absurd one-liner i cooked up that grabs onset slices from a single
        # subject for each condition and takes the mean
        onset_slices_conditions.append(
            np.mean(np.array([matrix[m : m + onset_length] for m in onsets[i]]), axis=0)
        )

    onset_slices_concat = np.array(onsets_slices_conditions, axis=0)

    # return flattened (one row for one subject) by default
    if return_flat:
        onset_slices_concat = onset_slices_concat.reshape(-1, *matrix.shape[1:])
    else:
        return onset_slices_concat


# def extract_onsets_all_subjects(
