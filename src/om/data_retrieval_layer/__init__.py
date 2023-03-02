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
# Copyright 2020 -2021 SLAC National Accelerator Laboratory
#
# Based on OnDA - Copyright 2014-2019 Deutsches Elektronen-Synchrotron DESY,
# a research centre of the Helmholtz Association.
"""
OM's Data Retrieval Layer.

This package contains OM's Data Retrieval Layer (which manages the retrieval of data
and data events from various sources). Functions and classes for different detectors,
facilities and software frameworks are implemented in separate modules in the package.
Other modules contain utilities functions and classes.
"""

from .data_retrieval_zmq import Jungfrau1MZmqDataRetrieval  # noqa: F401
from .data_retrieval_files import (  # noqa: F401
    Jungfrau1MFilesDataRetrieval,
    PilatusFilesDataRetrieval,
    Eiger16MFilesDataRetrieval,
    RayonixMccdFilesDataRetrieval,
    Lambda1M5FilesDataRetrieval,
)

try:
    import fabio  # type: ignore  # noqa: F401
    from .data_retrieval_files import PilatusFilesDataRetrieval  # noqa: F401
except ModuleNotFoundError:
    pass

try:
    import psana  # type: ignore  # noqa: F401
    from .data_retrieval_psana import (  # noqa: F401
        CxiLclsDataRetrieval,
        CxiLclsCspadDataRetrieval,
        LclsEpix100DataRetrieval,
        MfxLclsDataRetrieval,
        MfxLclsRayonixDataRetrieval,
    )
except ModuleNotFoundError:
    pass

try:
    import asapo_consumer  # type: ignore
    from .data_retrieval_asapo import EigerAsapoDataRetrieval
except ModuleNotFoundError:
    pass

try:
    import PIL  # type: ignore  # noqa: F401
    from .data_retrieval_http import EigerHttpDataRetrieval
except ModuleNotFoundError:
    pass

from .frame_retrieval import OmFrameDataRetrieval
