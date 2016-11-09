#    This file is part of OnDA.
#
#    OnDA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    OnDA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with OnDA.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from importlib import import_module
from inspect import isfunction, isclass


def import_correct_layer_module(layer, monitor_params):
    layer_paths = {
        'processing_layer': 'processing_layer.',
        'parallelization_layer': 'parallelization_layer.',
        'data_extraction_layer':
            'data_extraction_layer.',
        'instrument_layer':
            'data_extraction_layer.instrument_layer.',
    }

    if 'Backend' not in monitor_params:
        raise RuntimeError('[Backend] section is not present in the configuration file.')

    if layer not in monitor_params['Backend']:
        raise RuntimeError('Module for {0} not specified in the configuration file.'.format(layer))

    try:
        m = import_module(
            '{0}{1}'.format(
                layer_paths.get(layer, ''),
                monitor_params['Backend'][layer])
        )
    except ImportError:
        raise RuntimeError('Error when importing the {0}.  Either the {1} module does not exist, or importing it '
                           'causes an error.'.format(layer, monitor_params['Backend'][layer]))
    else:
        return m


def import_function_from_layer(function, layer):
    try:
        ret = getattr(layer, function)
    except AttributeError:
        raise RuntimeError('Error importing function {0} from layer {1}, the function does not exist,'.format(
            function, layer.__name__))
    else:
        if not isfunction(ret):
            raise RuntimeError('Error importing function {0} from layer {1}, {0} is not a function'.format(
                function, layer.__name__))
        else:
            return ret


def import_class_from_layer(cls, layer):
    try:
        ret = getattr(layer, cls)
    except AttributeError:
        raise RuntimeError('Error importing class {0} from layer {1}, {0} does not exist.'.format(
            cls, layer.__name__))
    else:
        if not isclass(ret):
            raise RuntimeError('Error importing class {0} from layer {1}, {0} is not a class.'.format(
                cls, layer.__name__))
        else:
            return ret


def import_list_from_layer(lst, layer):
    try:
        ret = getattr(layer, lst)
    except AttributeError:
        raise RuntimeError('Error importing list {0} from layer {1}, {0} does not exist.'.format(
            lst, layer.__name__))
    else:
        if not isinstance(ret, list):
            raise RuntimeError('Error importing list {0} from layer {1}, {0} is not a list.'.format(
                lst, layer.__name__))
        else:
            return ret
