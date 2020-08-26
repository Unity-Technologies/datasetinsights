from unittest.mock import MagicMock, Mock, patch, call

import numpy as np

from datasetinsights.stats.visualization.bbox2d_plot import (
    function_that_does_something
)


# Exercise 6
def test_function_that_does_something():
    # arrange
    image = np.zeros((100, 200, 3))
    left, top, right, bottom = 0, 0, 1, 1
    # act
    function_that_does_something(image, left, top, right, bottom, label="car")
    # assert
