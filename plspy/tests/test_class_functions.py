# TODO: build out unit tests and end-to-end tests

import numpy as np
import plspy
import pytest

rand_s = np.random.RandomState(950613)


# basic placeholder test for now
def test_mean_centre():
    X = rand_s.rand(50, 100)
    cond_order = np.array([[5, 5]] * 5)
    res = plspy.class_functions._mean_centre(
        X, cond_order, mctype=0, return_means=False
    )
    assert res.shape == (10, 100)
    res = plspy.class_functions._mean_centre(
        X, cond_order, mctype=1, return_means=False
    )
    assert res.shape == (10, 100)
    res = plspy.class_functions._mean_centre(
        X, cond_order, mctype=2, return_means=False
    )
    assert res.shape == (50, 100)
