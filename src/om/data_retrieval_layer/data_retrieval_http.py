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
Retrieval and handling of data from the http/REST interface.

This module contains Data Event Handlers and Data Retrieval classes that deal with data
retrieved from the http/REST interface.
"""
from typing import Dict

from om.abcs import data_retrieval_layer as drl_abcs
from om.data_retrieval_layer import data_event_handlers_http as deh_http
from om.data_retrieval_layer import data_sources_http as ds_http
from om.data_retrieval_layer import data_sources_generic as ds_generic
from om.utils import parameters


class EigerHttpDataRetrieval(drl_abcs.OmDataRetrievalBase):
    """
    See documentation of the `__init__` function.

    Base class: [`OmDataRetrieval`][om.abcs.data_retrieval_layer.OmDataRetrievalBase]
    """

    def __init__(self, *, monitor_parameters: parameters.MonitorParams, source: str):
        """
        Data Retrieval from Eiger http/REST interface.

        This method overrides the corresponding method of the base class: please also
        refer to the documentation of that class for more information.

        This class implements OM's Data Retrieval Layer for an Eiger detector using
        detector http/REST interface.

        * This class considers an individual data event as equivalent to the content of
          a tif file retrieved from the Eiger http/REST interface, which stores
          data related to a single detector frame.

        * Since Eiger http/REST monitor interface does not provide any detector
          distance or beam energy information, their values are retrieved from OM's
          configuration parameters (specifically, the `fallback_detector_distance_in_mm`
          and `fallback_beam_energy_in_eV` entries in the `data_retrieval_layer`
          parameter group).

        * The source string for this Data Retrieval class is the base URL of the
          'monitor' subsystem  of the Eiger detector http/REST interface:
          http://<address_of_dcu>/monitor/api/<version>
                .

        Arguments:

            monitor_parameters: A [MonitorParams]
                [om.utils.parameters.MonitorParams] object storing the OM monitor
                parameters from the configuration file.

            source: A string describing the data source.
        """

        data_sources: Dict[str, drl_abcs.OmDataSourceBase] = {
            "timestamp": ds_http.TimestampEigerHttp(
                data_source_name="timestamp", monitor_parameters=monitor_parameters
            ),
            "event_id": ds_http.EventIdEigerHttp(
                data_source_name="eventid", monitor_parameters=monitor_parameters
            ),
            "frame_id": ds_generic.FrameIdZero(
                data_source_name="frameid", monitor_parameters=monitor_parameters
            ),
            "detector_data": ds_http.EigerHttp(
                data_source_name="detector", monitor_parameters=monitor_parameters
            ),
            "beam_energy": ds_generic.FloatEntryFromConfiguration(
                data_source_name="fallback_beam_energy_in_eV",
                monitor_parameters=monitor_parameters,
            ),
            "detector_distance": ds_generic.FloatEntryFromConfiguration(
                data_source_name="fallback_detector_distance_in_mm",
                monitor_parameters=monitor_parameters,
            ),
        }

        self._data_event_handler: drl_abcs.OmDataEventHandlerBase = (
            deh_http.EigerHttpDataEventHandler(
                source=source,
                monitor_parameters=monitor_parameters,
                data_sources=data_sources,
            )
        )

    def get_data_event_handler(self) -> drl_abcs.OmDataEventHandlerBase:
        """
        Retrieves the Data Event Handler used by the class.

        This method overrides the corresponding method of the base class: please also
        refer to the documentation of that class for more information.

        Returns:

            The Data Event Handler used by the Data Retrieval class.
        """
        return self._data_event_handler
