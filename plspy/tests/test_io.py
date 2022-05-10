import numpy as np
import pytest

import plspy

rand_s = np.random.RandomState(950613)


def test_remap_vectorized_subject_to_4d():
    # create 5 mock subjects to use to create the mask
    # we'll just be testing on the first subject
    mock_subjects = [rand_s.rand(20, 10, 10, 10) for i in range(5)]

    # create mask from 5 subjects
    mask = plspy.io.create_threshold_mask_from_matrices(mock_subjects, threshold=0.15)

    # generate masked and vectorized subjects
    mock_masked = plspy.io.apply_mask_matrices(mock_subjects, mask)

    # recover masked, vectorized subject into 4d space
    recovered = plspy.io.remap_vectorized_subject_to_4d(
        mock_masked[0], mask, mock_subjects[0].shape
    )

    # check every index in original and recovered
    for i in range(mock_subjects[0].shape[0]):
        for j in range(mock_subjects[0].shape[1]):
            for k in range(mock_subjects[0].shape[2]):
                for l in range(mock_subjects[0].shape[3]):
                    # check equality if the values were masked initially
                    if mask[j, k, l]:
                        assert np.allclose(
                            mock_subjects[0][i, j, k, l], recovered[i, j, k, l]
                        )
                    # otherwise, make sure all other values are 0
                    else:
                        assert recovered[i, j, k, l] == 0
