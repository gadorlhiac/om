# This file is part of OM.
#
# OM is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# OM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with OM.
# If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2020 SLAC National Accelerator Laboratory
#
# Based on OnDA - Copyright 2014-2019 Deutsches Elektronen-Synchrotron DESY,
# a research centre of the Helmholtz Association.
"""
Algorithms for the processing of x-ray emission spectroscopy data.

This module contains algorithms that carry out x-ray emission spectroscopy-related data
processing (spectrum generation, etc.).
"""

from typing import Dict

import numpy  # type: ignore
from scipy import ndimage  # type: ignore


class XESAnalysis:
    """
    See documentation of the '__init__' function.
    """

    def __init__(
        self,
        rotation: float,
        min_row: int,
        max_row: int,
    ) -> None:
        """
        XES algorithm for calculating spectra from 2D camera.

        This algorithm extracts spectrum information from a 2d camera image. The image
        is rotated until the spectrum information is aligned to the vertical axis. The
        image area containing the spectrum information is then integrated.

        Arguments:

            rotation (int): The rotation in degrees that should be applied to align the
                linear signal on 2D camera with vertical axis.

            min_row (int): The minimum row index defining the region of integration for
                the spectrum after the signal has been rotated.

            max_row (int): The maximim row index defining the region of integration for
                the spectrum after the signal has been rotated.
        """
        self._rotation: float = rotation
        self._min_row: int = min_row
        self._max_row: int = max_row

    def generate_spectrum(self, data: numpy.ndarray) -> Dict[str, numpy.ndarray]:
        """
        Calculates spectrum information from camera image data.

        This function extracts spectrum information from a 2d image retrieved from a
        camera.

        Arguments:

            data (numpy.ndarray): The camera image data from which the spectrum will be
                generated.

        Returns:

            A dictionary with information about the XES spectrum extracted from the
            camera image data. The dictionary has the following keys:

            - A key named "spectrum" whose value is an array of intensities and
              position can be related to spectral energy.

            - A key named "spectrum_smooth" whose value is an array of intensities and
              position can be related to spectral energy.

            TODO: Are these 1d arrays?
        """

        imr: numpy.ndarray = ndimage.rotate(data, self._rotation, order=0)
        spectrum: numpy.ndarray = numpy.mean(
            imr[:, self._min_row : self._max_row], axis=1
        )
        spectrum_smoothed: numpy.ndarray = ndimage.filters.gaussian_filter1d(
            spectrum, 2
        )

        return {
            "spectrum": spectrum,
            "spectrum_smoothed": spectrum_smoothed,
        }


def _running_mean(x: numpy.ndarray, n: int) -> numpy.ndarray:
    # TODO: Document this function.
    return ndimage.filters.uniform_filter1d(x, n, mode="constant")[: -(n - 1)]
