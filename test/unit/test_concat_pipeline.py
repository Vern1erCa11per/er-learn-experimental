from unittest import TestCase

import numpy as np
from sklearn.preprocessing import StandardScaler

from feature.pipelines import ConcatPipeline


class TestConcatPipeline(TestCase):
    def test_fit(self):
        X = [[np.array([2, 1, 1]), np.array([2, 1, 2])],
             [np.array([0, 1, 3]), np.array([-2, 1, 4]), np.array([-2, 1, 5])]
             ]
        steps = [("step_1", StandardScaler())]
        sut = ConcatPipeline(steps)

        expected = [np.array([[1.11803399, 0., -1.41421356],
                              [1.11803399, 0., -0.70710678]]),
                    np.array([[0., 0., 0.],
                              [-1.11803399, 0., 0.70710678],
                              [-1.11803399, 0., 1.41421356]])]

        actual = sut.fit_transform(X)
        sut.fit(X)
        self.assertEqual(len(actual), len(expected))

        for ndarray_1, ndarray_2 in zip(actual, expected):
            np.testing.assert_array_almost_equal(ndarray_1, ndarray_2)