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
Retrieval of EPIX 2M detector data from psana.

This module contains functions that retrieve EPIX 10K 2M detector data from the psana
software framework (used at the LCLS facility).
"""
from typing import Any, Dict

import numpy  # type: ignore

from om.utils import exceptions


def epixka2m_detector_data(event: Dict[str, Any]) -> numpy.ndarray:
    """
    Retrieves a Epix10KA 2M detector data frame from psana.

    This function retrieves a single EPIX10KA 2M detector frame from psana. It returns
    the frame as a 2D array storing pixel data.

    Arguments:

        event: A dictionary storing the event data.

    Returns:

        One frame of detector data.
    """
    epixka2m_psana: numpy.ndarray = event["additional_info"][
        "psana_detector_interface"
    ]["detector_data"].calib(event["data"])
    if epixka2m_psana is None:
        raise exceptions.OmDataExtractionError(
            "Could not retrieve detector data from psana."
        )

    # Rearranges the data into 'slab' format.
    epixka2m_reshaped: numpy.ndarray = epixka2m_psana.reshape(16 * 352, 384)

    return epixka2m_reshaped


def epixka_detector_data(event: Dict[str, Any]) -> numpy.ndarray:
    """
    Retrieves a Epix10KA detector data frame from psana.

    This function retrieves a single EPIX10KA detector frame from psana. It returns the
    frame as a 2D array storing pixel data.

    Arguments:

        event: A dictionary storing the event data.

    Returns:

        One frame of detector data.
    """
    epixka_psana: numpy.ndarray = event["additional_info"]["psana_detector_interface"][
        "detector_data"
    ].calib(event["data"])
    if epixka_psana is None:
        raise exceptions.OmDataExtractionError(
            "Could not retrieve detector data from psana."
        )

    # Rearranges the data into 'slab' format.
    epixka_reshaped: numpy.ndarray = epixka_psana.reshape(2 * 352, 2 * 384)

    return epixka_reshaped
