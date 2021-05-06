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
OM monitor for x-ray emission spectroscopy.

This module contains an OM monitor for x-ray emission spectroscopy experiments.
"""
from __future__ import absolute_import, division, print_function

import sys
import time
from typing import Any, Dict, List, Tuple, Union

import h5py  # type: ignore
import numpy  # type: ignore

from om.algorithms import generic as gen_algs, xes as xes_algs
from om.processing_layer import base as process_layer_base
from om.utils import parameters, zmq_monitor


# TODO: Fix documentation for this file.


class XESMonitor(process_layer_base.OmMonitor):
    """
    See documentation for the '__init__' function.
    """

    def __init__(self, monitor_parameters):
        # type: (parameters.MonitorParams) -> None
        """
        An OM real-time monitor for x-ray emission spectroscopy experiments.

        See documentation of the constructor of the base class:
        :func:`~om.processing_layer.base.OmMonitor`.

        This monitor processes detector data frames, optionally applying detector
        calibration, dark correction and gain correction. It extracts 1d spectral
        information from the detector frame data. Additionally, it calculates the
        evolution of the hit rate over time. It broadcasts all this information over a
        network socket for visualization by other programs. Optionally, it can also
        broadcast calibrated and corrected detector data frames.
        """
        super(XESMonitor, self).__init__(monitor_parameters=monitor_parameters)

    def initialize_processing_node(self, node_rank: int, node_pool_size: int) -> None:
        """
        Initializes the OM nodes for the XES monitor.

        On the processing nodes, it initializes the correction and spectrum extraction
        algorithms, plus some internal counters.

        """

        self._hit_frame_sending_counter = 0  # type: int
        self._non_hit_frame_sending_counter = 0  # type: int

        dark_data_filename: str = self._monitor_params.get_param(
            group="correction", parameter="dark_filename", parameter_type=str
        )
        dark_data_hdf5_path: str = self._monitor_params.get_param(
            group="correction", parameter="dark_hdf5_path", parameter_type=str
        )
        mask_filename: str = self._monitor_params.get_param(
            group="correction", parameter="mask_filename", parameter_type=str
        )
        mask_hdf5_path: str = self._monitor_params.get_param(
            group="correction", parameter="mask_hdf5_path", parameter_type=str
        )
        gain_map_filename: str = self._monitor_params.get_param(
            group="correction", parameter="gain_filename", parameter_type=str
        )
        gain_map_hdf5_path: str = self._monitor_params.get_param(
            group="correction", parameter="gain_hdf5_path", parameter_type=str
        )
        self._correction = gen_algs.Correction(
            dark_filename=dark_data_filename,
            dark_hdf5_path=dark_data_hdf5_path,
            mask_filename=mask_filename,
            mask_hdf5_path=mask_hdf5_path,
            gain_filename=gain_map_filename,
            gain_hdf5_path=gain_map_hdf5_path,
        )

        self._hit_frame_sending_interval: Union[
            int, None
        ] = self._monitor_params.get_param(
            group="crystallography",
            parameter="hit_frame_sending_interval",
            parameter_type=int,
        )
        self._non_hit_frame_sending_interval: Union[
            int, None
        ] = self._monitor_params.get_param(
            group="crystallography",
            parameter="non_hit_frame_sending_interval",
        )

        xes_rotation_in_degrees: float = self._monitor_params.get_param(
            group="xes",
            parameter="rotation_in_degrees",
            parameter_type=float,
            required=True,
        )
        xes_min_row_in_pix_for_integration: int = self._monitor_params.get_param(
            group="xes",
            parameter="min_row_in_pix_for_integration",
            parameter_type=int,
            required=True,
        )
        xes_max_row_in_pix_for_integration: int = self._monitor_params.get_param(
            group="xes",
            parameter="max_row_in_pix_for_integration",
            parameter_type=int,
            required=True,
        )
        self._xes_intensity_threshold_in_ADU: int = self._monitor_params.get_param(
            group="xes",
            parameter="intensity_threshold_in_ADU",
            parameter_type=int,
            required=True,
        )
        self._xes_analysis = xes_algs.XESAnalysis(
            rotation=xes_rotation_in_degrees,
            min_row=xes_min_row_in_pix_for_integration,
            max_row=xes_max_row_in_pix_for_integration,
        )

        self._time_resolved: bool = self._monitor_params.get_param(
            group="xes",
            parameter="time_resolved",
            parameter_type=bool,
            required=True,
        )

        print("Processing node {0} starting.".format(node_rank))
        sys.stdout.flush()

    def initialize_collecting_node(self, node_rank: int, node_pool_size: int) -> None:
        """
        Initializes the OM nodes for the Crystallography monitor.

        On the processing nodes, it initializes the correction and peak finding
        algorithms, plus some internal counters. On the collecting node, this function
        initializes the data accumulation algorrithms and the storage for the
        aggregated statistics.
        """
        self._speed_report_interval: int = self._monitor_params.get_param(
            group="crystallography",
            parameter="speed_report_interval",
            parameter_type=int,
            required=True,
        )

        self._data_broadcast_interval: int = self._monitor_params.get_param(
            group="crystallography",
            parameter="data_broadcast_interval",
            parameter_type=int,
            required=True,
        )

        self._save_spectra: bool = self._monitor_params.get_param(
            group="xes",
            parameter="save_spectra",
            parameter_type=bool,
            required=True,
        )
        self._time_resolved = self._monitor_params.get_param(
            group="xes",
            parameter="time_resolved",
            parameter_type=bool,
            required=True,
        )

        self._spectra: List[numpy.ndarray] = []
        self._spectra_cumulative_sum: Union[numpy.ndarray, None] = None
        self._spectra_cumulative_sum_smoothed: Union[numpy.ndarray, None] = None
        self._cumulative_2d = None

        # self._spectra_cumulative_sum_pumped: Union[List[numpy.ndarray], None] = None
        # self._spectra_cumulative_sum_dark: Union[List[numpy.ndarray], None] = None
        # self._spectra_cumulative_sum_difference: Union[List[numpy.ndarray], None] = None
        self._cumulative_2d_pumped: Union[numpy.ndarray, None] = None
        self._cumulative_2d_dark: Union[numpy.ndarray, None] = None

        data_broadcast_url: Union[str, None] = self._monitor_params.get_param(
            group="crystallography", parameter="data_broadcast_url", parameter_type=str
        )
        if data_broadcast_url is None:
            data_broadcast_url = "tcp://{0}:12321".format(
                zmq_monitor.get_current_machine_ip()
            )

        self._data_broadcast_socket: zmq_monitor.ZmqDataBroadcaster = (
            zmq_monitor.ZmqDataBroadcaster(url=data_broadcast_url)
        )

        self._num_events: int = 0
        self._old_time: float = time.time()
        self._time: Union[float, None] = None

        self._num_events_pumped: int = 0
        self._num_events_dark: int = 0

        print("Starting the monitor...")
        sys.stdout.flush()

    def process_data(
        self, node_rank: int, node_pool_size: int, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """
        Processes a detector data frame.

        See documentation of the function in the base class:
        :func:`~om.processing_layer.base.OmMonitor.process_data`.

        This function performs calibration and correction of a detector data frame and
        extracts Bragg peak information. Finally, it prepares the Bragg peak data (and
        optionally, the detector frame data) for transmission to to the collecting
        node.
        """
        processed_data: Dict[str, Any] = {}
        corrected_detector_data: numpy.ndarray = self._correction.apply_correction(
            data=data["detector_data"]
        )

        # TODO: I think this will not work with a segmented detector.....

        # Apply a threshold
        processed_detector_data: numpy.ndarray = numpy.copy(corrected_detector_data)
        processed_detector_data[
            processed_detector_data < self._xes_intensity_threshold_in_ADU
        ] = 0

        # Mask the panel edges
        processed_detector_data[
            processed_detector_data.shape[0] // 2
            - 1 : processed_detector_data.shape[0] // 2
            + 1
        ] = 0
        processed_detector_data[
            :,
            processed_detector_data.shape[1] // 2
            - 1 : processed_detector_data.shape[1] // 2
            + 1,
        ] = 0

        xes: Dict[str, numpy.ndarray] = self._xes_analysis.generate_spectrum(
            processed_detector_data
        )

        processed_data["timestamp"] = data["timestamp"]
        processed_data["spectrum"] = xes["spectrum"]
        processed_data["beam_energy"] = data["beam_energy"]
        processed_data["data_shape"] = data["detector_data"].shape
        processed_data["detector_data"] = processed_detector_data
        if self._time_resolved:
            processed_data["optical_laser_active"] = data["optical_laser_active"]

        return (processed_data, node_rank)

    def collect_data(self, node_rank, node_pool_size, processed_data):
        # type: (int, int, Tuple[Dict[str, Any], int]) -> None
        """
        Computes statistics on aggregated data and broadcasts them via a network socket.

        This function computes aggregated statistics on data received from the
        processing nodes. It then broadcasts the results via a network socket (for
        visualization by other programs) using the MessagePack protocol.
        """
        received_data: Dict[str, Any] = processed_data[0]
        self._num_events += 1

        if self._time_resolved:
            if received_data["optical_laser_active"]:
                self._num_events_pumped += 1
            else:
                self._num_events_dark += 1

        if self._save_spectra:
            self._spectra.append(received_data["spectrum"])

        if self._cumulative_2d is None:
            self._cumulative_2d = received_data["detector_data"]
        else:
            self._cumulative_2d += (
                (received_data["detector_data"] - self._cumulative_2d * 1.0)
                / self._num_events
                * 1.0
            )

        # Calculate spectrum from cumulative 2D images
        cumulative_xes: Dict[str, numpy.ndarray] = self._xes_analysis.generate_spectrum(
            self._cumulative_2d
        )
        self._spectra_cumulative_sum = cumulative_xes["spectrum"]
        self._spectra_cumulative_sum_smoothed = cumulative_xes["spectrum_smoothed"]

        if numpy.mean(numpy.abs(self._spectra_cumulative_sum)) > 0:
            self._spectra_cumulative_sum /= numpy.mean(
                numpy.abs(self._spectra_cumulative_sum)
            )
        if numpy.mean(numpy.abs(self._spectra_cumulative_sum_smoothed)) > 0:
            self._spectra_cumulative_sum_smoothed /= numpy.mean(
                numpy.abs(self._spectra_cumulative_sum_smoothed)
            )

        spectrum_for_gui = received_data["spectrum"]

        if self._time_resolved:
            # Sum the spectra for pumped (optical_laser_active) and dark
            if self._cumulative_2d_pumped is None:
                self._cumulative_2d_pumped = received_data["detector_data"] * 0
            if self._cumulative_2d_dark is None:
                self._cumulative_2d_dark = received_data["detector_data"] * 0

            # Need to calculate a running average
            if received_data["optical_laser_active"]:
                self._cumulative_2d_pumped += (
                    (received_data["detector_data"] - self._cumulative_2d_pumped * 1.0)
                    / self._num_events_pumped
                    * 1.0
                )
            else:
                self._cumulative_2d_dark += (
                    (received_data["detector_data"] - self._cumulative_2d_dark * 1.0)
                    / self._num_events_dark
                    * 1.0
                )

            # Calculate spectrum from cumulative 2D images
            cumulative_xes_pumped: Dict[
                str, numpy.ndarray
            ] = self._xes_analysis.generate_spectrum(self._cumulative_2d_pumped)
            spectra_cumulative_sum_pumped: numpy.ndarray = cumulative_xes_pumped[
                "spectrum"
            ]

            # calculate spectrum from cumulative 2D images
            cumulative_xes_dark: Dict[
                str, numpy.ndarray
            ] = self._xes_analysis.generate_spectrum(self._cumulative_2d_dark)
            spectra_cumulative_sum_dark: numpy.ndarray = cumulative_xes_dark["spectrum"]

            # normalize spectra
            if numpy.mean(numpy.abs(spectra_cumulative_sum_pumped)) > 0:
                spectra_cumulative_sum_pumped /= numpy.mean(
                    numpy.abs(spectra_cumulative_sum_pumped)
                )
            if numpy.mean(numpy.abs(spectra_cumulative_sum_dark)) > 0:
                spectra_cumulative_sum_dark /= numpy.mean(
                    numpy.abs(spectra_cumulative_sum_dark)
                )

            spectra_cumulative_sum_difference = (
                spectra_cumulative_sum_pumped - spectra_cumulative_sum_dark
            )

        if self._num_events % self._data_broadcast_interval == 0:
            self._data_broadcast_socket.send_data(
                tag="view:omdata",
                message={
                    "timestamp": received_data["timestamp"],
                    "detector_data": self._cumulative_2d,
                    "spectrum": spectrum_for_gui,
                    "spectra_sum": self._spectra_cumulative_sum,
                    "spectra_sum_smoothed": self._spectra_cumulative_sum_smoothed,
                    "spectra_sum_pumped": spectra_cumulative_sum_pumped,
                    "spectra_sum_dark": spectra_cumulative_sum_dark,
                    "spectra_sum_difference": spectra_cumulative_sum_difference,
                    "beam_energy": received_data["beam_energy"],
                },
            )

        if self._num_events % self._speed_report_interval == 0:
            now_time: float = time.time()
            speed_report_msg: str = (
                "Processed: {0} in {1:.2f} seconds "
                "({2:.2f} Hz)".format(
                    self._num_events,
                    now_time - self._old_time,
                    (
                        float(self._speed_report_interval)
                        / float(now_time - self._old_time)
                    ),
                )
            )
            print(speed_report_msg)
            sys.stdout.flush()
            self._old_time = now_time

    def end_processing_on_processing_node(
        self, node_rank: int, node_pool_size: int
    ) -> None:
        """
        Ends processing actions on the processing nodes.

        This method overrides the corresponding method of the base class: please also
        refer to the documentation of that class for more information.

        This function prints a message on the console and ends the processing.

        Arguments:

            node_rank: The OM rank of the current node, which is an integer that
                unambiguously identifies the current node in the OM node pool.

            node_pool_size: The total number of nodes in the OM pool, including all the
                processing nodes and the collecting node.

        Returns:

            A dictionary storing information to be sent to the processing node
            (Optional: if this function returns nothing, no information is transferred
            to the processing node.

        """
        print("Processing node {0} shutting down.".format(node_rank))
        sys.stdout.flush()

    def end_processing_on_collecting_node(
        self, node_rank: int, node_pool_size: int
    ) -> None:
        """
        Ends processing on the collecting node.

        This method overrides the corresponding method of the base class: please also
        refer to the documentation of that class for more information.

        This function prints a message on the console and ends the processing.

        Arguments:

            node_rank: The OM rank of the current node, which is an integer that
                unambiguously identifies the current node in the OM node pool.

            node_pool_size: The total number of nodes in the OM pool, including all the
                processing nodes and the collecting node.
        """
        if self._save_spectra:
            print("Saving spectral data to the xes_spectra/h5 file.")
            radials_file = h5py.File("xes_spectra.h5", "w")
            radials_file.create_dataset("spectra", data=self._spectra)
            radials_file.close()
        print(
            "Processing finished. OM has processed {0} events in total.".format(
                self._num_events
            )
        )
        sys.stdout.flush()
