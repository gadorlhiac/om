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
OM's ta Retrieval Layer package.

This package contains OM's Data Retrieval Layer, with Data Event Handlers and Data
Extraction Functions supporting several facilities. Functions and classes for different
detectors and software frameworks are implemented in separate modules that are imported
on-demand when OM starts.
"""

try:
    import fabio  # type: ignore  # noqa: F401
    import h5py  # type: ignore  # noqa: F401

    from om.data_retrieval_layer.data_retrieval_filesystem import (
        Jungfrau1MFilesDataRetrieval,
        PilatusFilesDataRetrieval,
    )
    print("OM Message: activating file-based data retrieval")
    PilatusFilesDataEventHandler = PilatusFilesDataRetrieval
    Jungfrau1MFilesDataEventHandler = Jungfrau1MFilesDataRetrieval
except ModuleNotFoundError:
    pass

try:
    import psana  # type: ignore  # noqa: F401

    from om.data_retrieval_layer.data_retrieval_psana import (
        CxiLclsCspadDataRetrieval,
        CxiLclsDataRetrieval,
        MfxLclsDataRetrieval,
        MfxLclsRayonixDataRetrieval,
    )
    print("OM Message: activating psana data retrieval")
    MfxLclsDataEventHandler = MfxLclsDataRetrieval
    MfxLclsRayonixDataEventHandler = MfxLclsRayonixDataRetrieval
    CxiLclsDataEventHandler = CxiLclsDataRetrieval
    CxiLclsCspadDataEventHandler = CxiLclsCspadDataRetrieval

except ModuleNotFoundError:
    pass
