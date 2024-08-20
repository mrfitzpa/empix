# -*- coding: utf-8 -*-
# Copyright 2024 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
"""``empix`` is a Python library that contains tools for analyzing electron
microscopy data that are not available in `hyperspy
<https://hyperspy.org/hyperspy-doc/current/index.html>`_.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For accessing attributes of functions.
import inspect



# For general array handling.
import numpy as np

# For interpolating data.
import scipy.interpolate

# For validating and converting objects.
import czekitout.check
import czekitout.convert

# For defining classes that support enforced validation, updatability,
# pre-serialization, and de-serialization.
import fancytypes

# For creating hyperspy signals.
import hyperspy.signals

# For azimuthally integrating 2D hyperspy signals.
import pyFAI.detectors
import pyFAI.azimuthalIntegrator

# For downsampling hyperspy signals.
import skimage.measure



# Get version of current package.
from fancytypes.version import __version__



##################################
## Define classes and functions ##
##################################

# List of public objects in package.
__all__ = ["abs_sq",
           "OptionalAzimuthalAveragingParams",
           "azimuthally_average",
           "OptionalAzimuthalIntegrationParams",
           "azimuthally_integrate",
           "OptionalAnnularAveragingParams",
           "annularly_average",
           "OptionalAnnularIntegrationParams",
           "annularly_integrate",
           "OptionalCumulative1dIntegrationParams",
           "cumulatively_integrate_1d",
           "OptionalCroppingParams",
           "crop",
           "OptionalDownsamplingParams",
           "downsample",
           "OptionalResamplingParams",
           "resample"]



def abs_sq(input_signal, title=None):
    r"""The modulus squared of a given input ``hyperspy`` signal.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.complex_signal1d.ComplexSignal1D` | :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Modulus Squared of " +
        input_signal.metadata.General.title``, where ``input_signal`` is the 
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.signal2d.Signal2D`
        The output ``hyperspy`` signal that stores the modulus squared of the
        input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_input_signal(input_signal)
    title = _check_and_convert_title_v2(title, input_signal)

    ComplexSignal = hyperspy.signals.ComplexSignal
    if isinstance(input_signal, ComplexSignal):
        output_signal = input_signal.amplitude
        output_signal *= output_signal
    else:
        output_signal = input_signal * input_signal

    output_signal.metadata.General.title = title

    return output_signal



def _check_input_signal(input_signal):
    accepted_types = (hyperspy.signals.Signal1D,
                      hyperspy.signals.Signal2D,
                      hyperspy.signals.ComplexSignal1D,
                      hyperspy.signals.ComplexSignal2D)
    kwargs = {"obj": input_signal,
              "obj_name": "input_signal",
              "accepted_types": accepted_types}
    czekitout.check.if_instance_of_any_accepted_types(**kwargs)

    return input_signal



def _check_and_convert_title_v1(ctor_params):
    title = ctor_params["title"]
    
    if title is not None:
        try:
            title = czekitout.convert.to_str_from_str_like(title, "title")
        except:
            raise TypeError(_check_and_convert_title_v1_err_msg_1)

    return title



def _check_and_convert_title_v2(title, input_signal):
    title = _check_and_convert_title_v1({"title": title})
    if title is None:
        title = _default_title(input_signal, "Modulus Squared of ", "")

    return title



def _default_title(input_signal, auto_prefix, auto_suffix):
    input_signal_title = input_signal.metadata.General.title
    title = auto_prefix + input_signal_title + auto_suffix

    return title



def _check_and_convert_center_v1(ctor_params):
    center = ctor_params["center"]
    
    if center is not None:
        try:
            center = czekitout.convert.to_pair_of_floats(center, "center")
        except:
            raise TypeError(_check_and_convert_center_v1_err_msg_1)

    return center



def _check_and_convert_radial_range_v1(ctor_params):
    radial_range = ctor_params["radial_range"]
    
    if radial_range is not None:
        try:
            radial_range = czekitout.convert.to_pair_of_floats(radial_range,
                                                               "radial_range")
        except:
            raise TypeError(_check_and_convert_radial_range_v1_err_msg_1)

        if not (0 <= radial_range[0] <= radial_range[1]):
            raise ValueError(_check_and_convert_radial_range_v1_err_msg_1)

    return radial_range



def _check_and_convert_num_bins_v1(ctor_params):
    num_bins = ctor_params["num_bins"]

    if num_bins is not None:
        try:
            num_bins = czekitout.convert.to_positive_int(num_bins, "num_bins")
        except:
            raise TypeError(_check_and_convert_num_bins_v1_err_msg_1)

    return num_bins



def _pre_serialize_center(center):
    serializable_rep = center
    
    return serializable_rep



def _pre_serialize_radial_range(radial_range):
    serializable_rep = radial_range
    
    return serializable_rep



def _pre_serialize_num_bins(num_bins):
    serializable_rep = num_bins
    
    return serializable_rep



def _pre_serialize_title(title):
    serializable_rep = title
    
    return serializable_rep



def _de_pre_serialize_center(serializable_rep):
    center = serializable_rep

    return center



def _de_pre_serialize_radial_range(serializable_rep):
    radial_range = serializable_rep

    return radial_range



def _de_pre_serialize_num_bins(serializable_rep):
    num_bins = serializable_rep

    return num_bins



def _de_pre_serialize_title(serializable_rep):
    title = serializable_rep

    return title



class OptionalAzimuthalAveragingParams(
        fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function
    :func:`empix.azimuthally_average`.

    The Python function :func:`empix.azimuthally_integrate` averages
    azimuthally a given input 2D ``hyperspy`` signal. The Python function
    assumes that the input 2D ``hyperspy`` signal samples from a mathematical
    function :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise
    continuous in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and
    :math:`u_{y}` are the horizontal and vertical coordinates in the signal
    space of the input signal, and :math:`\mathbf{m}` is a vector of integers
    representing the navigation indices of the input signal. The Python function
    approximates the azimuthal average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given the input signal. We
    define the azimuthal average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&\overline{S}_{\mathbf{m}}\left(\left.R_{xy}=r_{xy}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad=\frac{1}{2\pi r_{xy}}
        \int_{0}^{2\pi}du_{\phi}\,u_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: azimuthal_average__1

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the azimuthal averaging.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``center`` is set to ``None``, then the reference point
        :math:`\left(c_{x},c_{y}\right)`, from which the radial distance is
        defined for the azimuthal averaging, is set to the signal space
        coordinates corresponding to the center signal space pixel.  Otherwise,
        if ``center`` is set to a pair of floating-point numbers, then
        ``center[0]`` and ``center[1]`` specify :math:`c_{x}` and :math:`c_{y}`
        respectively, in the same units of the corresponding signal space axes
        of the input signal.
    radial_range : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``radial_range`` is set to ``None``, then the radial range, over
        which the azimuthal averaging is performed, is from zero to the largest
        radial distance within the signal space boundaries of the input signal
        for all azimuthal angles. Otherwise, if ``radial_range`` is set to a
        pair of floating-point numbers, then ``radial_range[0]`` and
        ``radial_range[1]`` specify the minimum and maximum radial distances of
        the radial range respectively, in the same units of the horizontal and
        vertical axes respectively of the signal space of the input signal. Note
        that in this case ``radial_range`` must satisfy
        ``0<=radial_range[0]<=radial_range[1]``. Moreover, the function
        represented by the input signal is assumed to be equal to zero
        everywhere outside of the signal space boundaries of said input signal.
    num_bins : `int` | `None`, optional
        ``num_bins`` must either be a positive integer or of the `NoneType`: if
        the former, then the dimension of the signal space of the output signal
        ``output_signal`` is set to ``num_bins``; if the latter, then the
        dimension of the signal space of ``output_signal`` is set to
        ``min(input_signal.data.shape[-2:])``, where ``input_signal`` is the
        input ``hyperspy`` signal.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Azimuthally Averaged " +
        input_signal.metadata.General.title``, where ``input_signal`` is the 
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center_v1,
         "radial_range": _check_and_convert_radial_range_v1,
         "num_bins": _check_and_convert_num_bins_v1,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "radial_range": _pre_serialize_radial_range,
         "num_bins": _pre_serialize_num_bins,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "radial_range": _de_pre_serialize_radial_range,
         "num_bins": _de_pre_serialize_num_bins,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 center=None,
                 radial_range=None,
                 num_bins=None,
                 title=None):
        ctor_params = {"center": center,
                       "radial_range": radial_range,
                       "num_bins": num_bins,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def azimuthally_average(input_signal, optional_params=None):
    r"""Average azimuthally a given input 2D ``hyperspy`` signal.

    This current Python function assumes that the input 2D ``hyperspy`` signal
    samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    azimuthal average of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given
    the input signal. We define the azimuthal average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&\overline{S}_{\mathbf{m}}\left(\left.R_{xy}=r_{xy}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad=\frac{1}{2\pi r_{xy}}
        \int_{0}^{2\pi}du_{\phi}\,u_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: azimuthal_average__2

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the azimuthal averaging.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalAzimuthalAveragingParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalAzimuthalAveragingParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalAzimuthalAveragingParams`, then the default
        values of the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.complex_signal1d.ComplexSignal1D`
        The output ``hyperspy`` signal that samples the azimuthal average of the
        input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_2D_input_signal(input_signal)
    center, radial_range, num_bins, title = \
        _check_and_convert_optional_params_v1(optional_params, input_signal)
    if title is None:
        title = _default_title(input_signal, "Azimuthally Averaged ", "")
    
    azimuthal_integrator = _construct_azimuthal_integrator(input_signal, center)

    bin_coords, output_data = \
        _apply_azimuthal_integrator_to_input_signal(azimuthal_integrator,
                                                    input_signal,
                                                    num_bins,
                                                    radial_range)
    
    metadata = {"General": {"title": title}, "Signal": dict()}
    if np.isrealobj(output_data):
        output_signal = hyperspy.signals.Signal1D(data=output_data,
                                                  metadata=metadata)
    else:
        output_signal = hyperspy.signals.ComplexSignal1D(data=output_data,
                                                         metadata=metadata)
    _update_output_signal_axes_v1(output_signal, bin_coords, input_signal)

    return output_signal



def _check_2D_input_signal(input_signal):
    accepted_types = (hyperspy.signals.Signal2D,
                      hyperspy.signals.ComplexSignal2D)
    kwargs = {"obj": input_signal,
              "obj_name": "input_signal",
              "accepted_types": accepted_types}
    czekitout.check.if_instance_of_any_accepted_types(**kwargs)

    return input_signal



def _check_and_convert_optional_params_v1(optional_params, input_signal):
    if optional_params is None:
        optional_params = OptionalAzimuthalAveragingParams()
    if not isinstance(optional_params, OptionalAzimuthalAveragingParams):
        raise TypeError(_check_and_convert_optional_params_v1_err_msg_1)

    center = optional_params.core_attrs["center"]
    center = _check_and_convert_center_v2(center, input_signal)

    radial_range = optional_params.core_attrs["radial_range"]
    radial_range = _check_and_convert_radial_range_v2(radial_range,
                                                      input_signal,
                                                      center)

    num_bins = optional_params.core_attrs["num_bins"]
    num_bins = _check_and_convert_num_bins_v2(num_bins, input_signal)

    title = optional_params.core_attrs["title"]

    return center, radial_range, num_bins, title



def _check_and_convert_center_v2(center, signal):
    h_range, v_range = _h_and_v_ranges(signal)
    
    if center is not None:
        if not ((h_range[0] < center[0] < h_range[1])
                and (v_range[0] < center[1] < v_range[1])):
            raise ValueError(_check_and_convert_center_v2_err_msg_2)
    else:
        center = ((h_range[0]+h_range[1])/2, (v_range[0]+v_range[1])/2)

    center = (center[0], center[1])

    return center


def _h_and_v_ranges(signal):
    h_scale = signal.axes_manager[-2].scale
    v_scale = signal.axes_manager[-1].scale

    n_v, n_h = signal.data.shape[-2:]

    h_offset = signal.axes_manager[-2].offset
    v_offset = signal.axes_manager[-1].offset

    h_min = min(h_offset, h_offset + n_h*h_scale)
    h_max = max(h_offset, h_offset + n_h*h_scale)
    h_range = (h_min, h_max)

    v_min = min(v_offset, v_offset + n_v*v_scale)
    v_max = max(v_offset, v_offset + n_v*v_scale)
    v_range = (v_min, v_max)

    return h_range, v_range



def _check_and_convert_radial_range_v2(radial_range, signal, center):
    if radial_range is None:
        h_range, v_range = _h_and_v_ranges(signal)
        temp_1 = min(abs(center[0] - h_range[0]), abs(center[0] - h_range[1]))
        temp_2 = min(abs(center[1] - v_range[0]), abs(center[1] - v_range[1]))
        radial_range = (0, min(temp_1, temp_2))

    # Need to multiply range by 1000 because of the units used in pyFAI's
    # azimuthal integrator.
    radial_range = (radial_range[0]*1000, radial_range[1]*1000)

    return radial_range



def _check_and_convert_num_bins_v2(num_bins, signal):
    if num_bins is None:
        if _is_1d_signal(signal):
            num_bins = signal.data.shape[-1]
        else:
            num_bins = min(signal.data.shape[-2:])

    return num_bins



def _is_1d_signal(signal):
    signal_1d_types = (hyperspy.signals.Signal1D,
                       hyperspy.signals.ComplexSignal1D)
    result = isinstance(signal, signal_1d_types)

    return result



def _construct_azimuthal_integrator(signal, center):
    detector = _construct_pyfai_detector(signal)

    h_scale = signal.axes_manager[-2].scale
    v_scale = signal.axes_manager[-1].scale

    # ``pone_1`` and ``poni_2`` are the vertical and horizontal displacements
    # of the reference point, from which to perform the azimuthal integration,
    # from the top left corner of the input signal.
    h_range, v_range = _h_and_v_ranges(signal)
    poni_1 = center[1] - v_range[0] if v_scale > 0 else v_range[1] - center[1]
    poni_2 = center[0] - h_range[0] if h_scale > 0 else h_range[1] - center[0]

    # We require ``L >> max(v_pixel_size, h_pixel_size)``.
    h_pixel_size = abs(h_scale)
    v_pixel_size = abs(v_scale)
    L = 10000 * max(v_pixel_size, h_pixel_size)

    AzimuthalIntegrator = pyFAI.azimuthalIntegrator.AzimuthalIntegrator
    azimuthal_integrator = AzimuthalIntegrator(dist=L,
                                               poni1=poni_1, 
                                               poni2=poni_2, 
                                               detector=detector)

    return azimuthal_integrator
    


def _construct_pyfai_detector(signal):
    h_pixel_size = abs(signal.axes_manager[-2].scale)
    v_pixel_size = abs(signal.axes_manager[-1].scale)
    detector = pyFAI.detectors.Detector(pixel1=v_pixel_size,
                                        pixel2=h_pixel_size)

    return detector



def _apply_azimuthal_integrator_to_input_signal(azimuthal_integrator,
                                                input_signal,
                                                num_bins,
                                                radial_range):
    navigation_dims = input_signal.data.shape[:-2]
    output_data_shape = list(navigation_dims) + [num_bins]
    output_data = np.zeros(output_data_shape, dtype=input_signal.data.dtype)

    num_patterns = int(np.prod(navigation_dims))
    for pattern_idx in range(num_patterns):
        navigation_indices = np.unravel_index(pattern_idx, navigation_dims)
        input_datasubset = input_signal.data[navigation_indices]
            
        kwargs = {"azimuthal_integrator": azimuthal_integrator,
                  "input_datasubset": input_datasubset,
                  "num_bins": num_bins,
                  "radial_range": radial_range}
        bin_coords, output_datasubset = \
            _apply_azimuthal_integrator_to_input_datasubset(**kwargs)
        output_data[navigation_indices] = output_datasubset
    
    bin_coords /= 1000  # Because of the artificial "r_mm" units.

    return bin_coords, output_data



def _apply_azimuthal_integrator_to_input_datasubset(azimuthal_integrator,
                                                    input_datasubset,
                                                    num_bins,
                                                    radial_range):
    if np.isrealobj(input_datasubset):
        bin_coords, output_datasubset = \
            azimuthal_integrator.integrate1d(input_datasubset, 
                                             npt=num_bins,
                                             radial_range=radial_range,
                                             unit="r_mm")
    else:
        bin_coords, real_output_datasubset = \
            azimuthal_integrator.integrate1d(input_datasubset.real, 
                                             npt=num_bins,
                                             radial_range=radial_range,
                                             unit="r_mm")
        bin_coords, imag_output_datasubset = \
            azimuthal_integrator.integrate1d(input_datasubset.imag, 
                                             npt=num_bins,
                                             radial_range=radial_range,
                                             unit="r_mm")

        output_datasubset = real_output_datasubset + 1j*imag_output_datasubset

    return bin_coords, output_datasubset



def _update_output_signal_axes_v1(output_signal, bin_coords, input_signal):
    if _is_1d_signal(output_signal):
        navigation_dims = input_signal.data.shape[:-1]
        axis_name = {"Å": "$r$", "1/Å": "$k$"}
    else:
        navigation_dims = input_signal.data.shape[:-2]
        axis_name = {"Å": "$r_{xy}$", "1/Å": "$k_{xy}$"}
        
    for idx in range(len(navigation_dims)):
        axis = input_signal.axes_manager[idx]
        output_signal.axes_manager[idx].update_from(axis)
        output_signal.axes_manager[idx].name = axis.name

    units = input_signal.axes_manager[-1].units
    axis = hyperspy.axes.UniformDataAxis(size=len(bin_coords),
                                         scale=bin_coords[1]-bin_coords[0],
                                         offset=bin_coords[0],
                                         units=units)
    output_signal.axes_manager[-1].update_from(axis)
    output_signal.axes_manager[-1].name = axis_name.get(units, "")

    return None



class OptionalAzimuthalIntegrationParams(
        fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function
    :func:`empix.azimuthally_integrate`.

    The Python function :func:`empix.azimuthally_integrate` integrates
    azimuthally a given input 2D ``hyperspy`` signal. The Python function
    assumes that the input 2D ``hyperspy`` signal samples from a mathematical
    function :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise
    continuous in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and
    :math:`u_{y}` are the horizontal and vertical coordinates in the signal
    space of the input signal, and :math:`\mathbf{m}` is a vector of integers
    representing the navigation indices of the input signal. The Python function
    approximates the azimuthal integral of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given the input signal. We
    define the azimuthal integral of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&S_{\mathbf{m}}\left(\left.R_{xy}=r_{xy}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad=
        \int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: azimuthal_integral__1

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the azimuthal integration.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``center`` is set to ``None``, then the reference point
        :math:`\left(c_{x},c_{y}\right)`, from which the radial distance is
        defined for the azimuthal integration, is set to the signal space
        coordinates corresponding to the center signal space pixel.  Otherwise,
        if ``center`` is set to a pair of floating-point numbers, then
        ``center[0]`` and ``center[1]`` specify :math:`c_{x}` and :math:`c_{y}`
        respectively, in the same units of the corresponding signal space axes
        of the input signal.
    radial_range : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``radial_range`` is set to ``None``, then the radial range, over
        which the azimuthal integration is performed, is from zero to the largest
        radial distance within the signal space boundaries of the input signal
        for all azimuthal angles. Otherwise, if ``radial_range`` is set to a
        pair of floating-point numbers, then ``radial_range[0]`` and
        ``radial_range[1]`` specify the minimum and maximum radial distances of
        the radial range respectively, in the same units of the horizontal and
        vertical axes respectively of the signal space of the input signal. Note
        that in this case ``radial_range`` must satisfy
        ``0<=radial_range[0]<=radial_range[1]``. Moreover, the function
        represented by the input signal is assumed to be equal to zero
        everywhere outside of the signal space boundaries of said input signal.
    num_bins : `int` | `None`, optional
        ``num_bins`` must either be a positive integer or of the `NoneType`: if
        the former, then the dimension of the signal space of the output signal
        ``output_signal`` is set to ``num_bins``; if the latter, then the
        dimension of the signal space of ``output_signal`` is set to
        ``min(input_signal.data.shape[-2:])``, where ``input_signal`` is the
        input ``hyperspy`` signal.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Azimuthally Integrated " +
        input_signal.metadata.General.title``, where ``input_signal`` is the 
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center_v1,
         "radial_range": _check_and_convert_radial_range_v1,
         "num_bins": _check_and_convert_num_bins_v1,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "radial_range": _pre_serialize_radial_range,
         "num_bins": _pre_serialize_num_bins,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "radial_range": _de_pre_serialize_radial_range,
         "num_bins": _de_pre_serialize_num_bins,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 center=None,
                 radial_range=None,
                 num_bins=None,
                 title=None):
        ctor_params = {"center": center,
                       "radial_range": radial_range,
                       "num_bins": num_bins,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def azimuthally_integrate(input_signal, optional_params=None):
    r"""Integrate azimuthally a given input 2D ``hyperspy`` signal.

    This current Python function assumes that the input 2D ``hyperspy`` signal
    samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    azimuthal integral of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given
    the input signal. We define the azimuthal integral of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&S_{\mathbf{m}}\left(\left.R_{xy}=r_{xy}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad=
        \int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: azimuthal_integral__2

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the azimuthal integration.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalAzimuthalIntegrationParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalAzimuthalIntegrationParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalAzimuthalIntegrationParams`, then the default
        values of the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.complex_signal1d.ComplexSignal1D`
        The output ``hyperspy`` signal that samples the azimuthal integral of
        the input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    if optional_params is None:
        optional_params = OptionalAzimuthalIntegrationParams()
    if not isinstance(optional_params, OptionalAzimuthalIntegrationParams):
        raise TypeError(_azimuthally_integrate_err_msg_1)
    kwargs = {"center": optional_params.core_attrs["center"],
              "radial_range": optional_params.core_attrs["radial_range"],
              "num_bins": optional_params.core_attrs["num_bins"],
              "title": optional_params.core_attrs["title"]}
    temp_optional_params = OptionalAzimuthalAveragingParams(**kwargs)
    
    output_signal = azimuthally_average(input_signal, temp_optional_params)
    title = optional_params.core_attrs["title"]
    if title is None:
        title = _default_title(input_signal, "Azimuthally Integrated ", "")
    output_signal.metadata.General.title = title

    bin_coords = _bin_coords(output_signal)
    navigation_rank = len(output_signal.data.shape) - 1

    for idx, r_xy in enumerate(bin_coords):
        multi_dim_slice = tuple([slice(None)]*navigation_rank + [idx])
        output_signal.data[multi_dim_slice] *= 2 * np.pi * r_xy

    return output_signal



def _bin_coords(output_signal):
    offset = output_signal.axes_manager[-1].offset
    scale = output_signal.axes_manager[-1].scale
    size = output_signal.axes_manager[-1].size
    bin_coords = np.arange(offset, offset + size*scale - 1e-10, scale)

    return bin_coords



class OptionalAnnularAveragingParams(
        fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function
    :func:`empix.annularly_average`.

    The Python function :func:`empix.annularly_average` averages annularly a
    given input 2D ``hyperspy`` signal. The Python function assumes that the
    input 2D ``hyperspy`` signal samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    azimuthal average of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given
    the input signal. We define the annular average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&\overline{S}_{\mathbf{m}}\left(\left.r_{xy,i}\le R<r_{xy,f}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad
        =\frac{1}{\pi\left(r_{xy,f}^{2}-r_{xy,i}^{2}\right)}
        \int_{r_{xy,i}}^{r_{xy,f}}dr_{xy}\int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: annular_average__1

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the annular averaging.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``center`` is set to ``None``, then the reference point
        :math:`\left(c_{x},c_{y}\right)`, from which the radial distance is
        defined for the annular averaging, is set to the signal space
        coordinates corresponding to the center signal space pixel.  Otherwise,
        if ``center`` is set to a pair of floating-point numbers, then
        ``center[0]`` and ``center[1]`` specify :math:`c_{x}` and :math:`c_{y}`
        respectively, in the same units of the corresponding signal space axes
        of the input signal.
    radial_range : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``radial_range`` is set to ``None``, then the radial range, over
        which the annular averaging is performed, is from zero to the largest
        radial distance within the signal space boundaries of the input signal
        for all azimuthal angles. Otherwise, if ``radial_range`` is set to a
        pair of floating-point numbers, then ``radial_range[0]`` and
        ``radial_range[1]`` specify the minimum and maximum radial distances of
        the radial range respectively, in the same units of the horizontal and
        vertical axes respectively of the signal space of the input signal. Note
        that in this case ``radial_range`` must satisfy
        ``0<=radial_range[0]<=radial_range[1]``. Moreover, the function
        represented by the input signal is assumed to be equal to zero
        everywhere outside of the signal space boundaries of said input signal.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Annularly Averaged " +
        input_signal.metadata.General.title``, where ``input_signal`` is the 
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center_v1,
         "radial_range": _check_and_convert_radial_range_v1,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "radial_range": _pre_serialize_radial_range,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "radial_range": _de_pre_serialize_radial_range,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 center=None,
                 radial_range=None,
                 title=None):
        ctor_params = {"center": center,
                       "radial_range": radial_range,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def annularly_average(input_signal, optional_params=None):
    r"""Average annularly a given input 2D ``hyperspy`` signal.

    This current Python function assumes that the input 2D ``hyperspy`` signal
    samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    annular average of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given the
    input signal. We define the annular average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&\overline{S}_{\mathbf{m}}\left(\left.r_{xy,i}\le R<r_{xy,f}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad
        =\frac{1}{\pi\left(r_{xy,f}^{2}-r_{xy,i}^{2}\right)}
        \int_{r_{xy,i}}^{r_{xy,f}}dr_{xy}\int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: annular_average__2

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the annular averaging.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalAnnularAveragingParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalAnnularAveragingParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalAnnularAveragingParams`, then the default values 
        of the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy.signal.BaseSignal` | :class:`hyperspy._signals.complex_signal.ComplexSignal`
        The output ``hyperspy`` signal that samples the annular average of the
        input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_2D_input_signal(input_signal)
    if optional_params is None:
        optional_params = OptionalAnnularAveragingParams()
    if not isinstance(optional_params, OptionalAnnularAveragingParams):
        raise TypeError(_annularly_average_err_msg_1)
    kwargs = {"center": optional_params.core_attrs["center"],
              "radial_range": optional_params.core_attrs["radial_range"],
              "num_bins": 2 * min(input_signal.data.shape[-2:]),
              "title": optional_params.core_attrs["title"]}
    temp_optional_params = OptionalAzimuthalAveragingParams(**kwargs)
    temp_signal = azimuthally_average(input_signal, temp_optional_params)

    bin_coords = _bin_coords(temp_signal)
    navigation_rank = len(temp_signal.data.shape) - 1

    for idx, r_xy in enumerate(bin_coords):
        multi_dim_slice = tuple([slice(None)]*navigation_rank + [idx])
        temp_signal.data[multi_dim_slice] *= 2 * np.pi * r_xy

    center = temp_optional_params.core_attrs["center"]
    center = _check_and_convert_center_v2(center, input_signal)

    radial_range = temp_optional_params.core_attrs["radial_range"]
    radial_range = _check_and_convert_radial_range_v2(radial_range,
                                                      input_signal,
                                                      center)
    r_xy_i, r_xy_f = (radial_range[0]/1000, radial_range[1]/1000)
    area_of_annulus = np.pi * (r_xy_f**2 - r_xy_i**2)    
    r_xy_scale = temp_signal.axes_manager[-1].scale

    output_signal = temp_signal.sum(axis=-1)
    output_signal.data *= r_xy_scale / area_of_annulus

    title = optional_params.core_attrs["title"]
    if title is None:
        title = _default_title(input_signal, "Annularly Averaged ", "")
    output_signal.metadata.General.title = title

    return output_signal



class OptionalAnnularIntegrationParams(
        fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function
    :func:`empix.annularly_integrate`.

    The Python function :func:`empix.annularly_integrate` integrates annularly a
    given input 2D ``hyperspy`` signal. The Python function assumes that the
    input 2D ``hyperspy`` signal samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    azimuthal average of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given
    the input signal. We define the annular average of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&S_{\mathbf{m}}\left(\left.r_{xy,i}\le R<r_{xy,f}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad
        =\int_{r_{xy,i}}^{r_{xy,f}}dr_{xy}\int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: annular_integral__1

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the annular integration.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``center`` is set to ``None``, then the reference point
        :math:`\left(c_{x},c_{y}\right)`, from which the radial distance is
        defined for the annular integration, is set to the signal space
        coordinates corresponding to the center signal space pixel.  Otherwise,
        if ``center`` is set to a pair of floating-point numbers, then
        ``center[0]`` and ``center[1]`` specify :math:`c_{x}` and :math:`c_{y}`
        respectively, in the same units of the corresponding signal space axes
        of the input signal.
    radial_range : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``radial_range`` is set to ``None``, then the radial range, over
        which the annular integration is performed, is from zero to the largest
        radial distance within the signal space boundaries of the input signal
        for all azimuthal angles. Otherwise, if ``radial_range`` is set to a
        pair of floating-point numbers, then ``radial_range[0]`` and
        ``radial_range[1]`` specify the minimum and maximum radial distances of
        the radial range respectively, in the same units of the horizontal and
        vertical axes respectively of the signal space of the input signal. Note
        that in this case ``radial_range`` must satisfy
        ``0<=radial_range[0]<=radial_range[1]``. Moreover, the function
        represented by the input signal is assumed to be equal to zero
        everywhere outside of the signal space boundaries of said input signal.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Annularly Integrated " +
        input_signal.metadata.General.title``, where ``input_signal`` is the 
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center_v1,
         "radial_range": _check_and_convert_radial_range_v1,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "radial_range": _pre_serialize_radial_range,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "radial_range": _de_pre_serialize_radial_range,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 center=None,
                 radial_range=None,
                 title=None):
        ctor_params = {"center": center,
                       "radial_range": radial_range,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def annularly_integrate(input_signal, optional_params=None):
    r"""Integrate annularly a given input 2D ``hyperspy`` signal.

    This current Python function assumes that the input 2D ``hyperspy`` signal
    samples from a mathematical function
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` which is piecewise continuous
    in :math:`u_{x}` and :math:`u_{y}`, where :math:`u_{x}` and :math:`u_{y}`
    are the horizontal and vertical coordinates in the signal space of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    annular integral of :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` given the
    input signal. We define the annular integral of
    :math:`F_{\mathbf{m}}\left(u_{x},u_{y}\right)` as

    .. math ::
	&S_{\mathbf{m}}\left(\left.r_{xy,i}\le R<r_{xy,f}\right|
        0\le U_{\phi}<2\pi;c_{x},c_{y}\right)\\&\quad
        =\int_{r_{xy,i}}^{r_{xy,f}}dr_{xy}\int_{0}^{2\pi}du_{\phi}\,r_{xy}
        F_{\mathbf{m}}\left(c_{x}+r_{xy}\cos\left(u_{\phi}\right),
        c_{y}+r_{xy}\sin\left(u_{\phi}\right)\right),
        :label: annular_integral__2

    where :math:`\left(c_{x},c_{y}\right)` is the reference point from which the
    radial distance :math:`r_{xy}` is defined for the annular integration.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalAnnularIntegrationParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalAnnularIntegrationParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalAnnularIntegrationParams`, then the default values
        of the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy.signal.BaseSignal` | :class:`hyperspy._signals.complex_signal.ComplexSignal`
        The output ``hyperspy`` signal that samples the annular integral of the
        input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_2D_input_signal(input_signal)
    if optional_params is None:
        optional_params = OptionalAnnularIntegrationParams()
    if not isinstance(optional_params, OptionalAnnularIntegrationParams):
        raise TypeError(_annularly_integrate_err_msg_1)
    kwargs = {"center": optional_params.core_attrs["center"],
              "radial_range": optional_params.core_attrs["radial_range"],
              "num_bins": 2 * min(input_signal.data.shape[-2:]),
              "title": optional_params.core_attrs["title"]}
    temp_optional_params = OptionalAzimuthalIntegrationParams(**kwargs)
    
    integrated_azimuthally_signal = azimuthally_integrate(input_signal,
                                                          temp_optional_params)
    r_xy_scale = integrated_azimuthally_signal.axes_manager[-1].scale
    output_signal = integrated_azimuthally_signal.sum(axis=-1)
    output_signal.data *= r_xy_scale

    title = optional_params.core_attrs["title"]
    if title is None:
        title = _default_title(input_signal, "Annularly Integrated ", "")
    output_signal.metadata.General.title = title

    return output_signal



def _check_and_convert_limits_v1(ctor_params):
    limits = ctor_params["limits"]
    
    if limits is not None:
        try:
            limits = czekitout.convert.to_pair_of_floats(limits, "limits")
        except:
            raise TypeError(_check_and_convert_limits_v1_err_msg_1)

    return limits



def _check_and_convert_normalize(ctor_params):
    normalize = czekitout.convert.to_bool(ctor_params["normalize"], "normalize")

    return normalize



def _pre_serialize_limits(limits):
    serializable_rep = limits
    
    return serializable_rep



def _pre_serialize_normalize(normalize):
    serializable_rep = normalize
    
    return serializable_rep



def _de_pre_serialize_limits(serializable_rep):
    limits = serializable_rep

    return limits



def _de_pre_serialize_normalize(serializable_rep):
    normalize = serializable_rep

    return normalize



class OptionalCumulative1dIntegrationParams(
        fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function
    :func:`empix.cumulatively_integrate_1d`.

    The Python function :func:`empix.cumulatively_integrate_1d` integrates
    cumulatively a given input 1D ``hyperspy`` signal. The Python function
    assumes that the input 1D ``hyperspy`` signal samples from a mathematical
    function :math:`F_{\mathbf{m}}\left(u\right)` which is piecewise continuous
    in :math:`u`, where :math:`u` is the signal space coordinate of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    cumulative integral of :math:`F_{\mathbf{m}}\left(u\right)` given the input
    signal. We define the cumulative integral of
    :math:`F_{\mathbf{m}}\left(u\right)` as

    .. math ::
	\text{CDF}_{\text{1D}}\left(u\right)&=\int_{u_{i}}^{u}du^{\prime}\,
        F_{\mathbf{m}}\left(u^{\prime}\right),
        \\&\quad u\in\left[\min\left(u_{i},u_{f}\right),
        \max\left(u_{i},u_{f}\right)\right],
        :label: cumulative_integral_1d__1

    where :math:`u_i` and :math:`u_f` specify the interval over which cumulative
    integration is performed, the interval being
    :math:`\left[\min\left(u_{i},u_{f}\right),
    \max\left(u_{i},u_{f}\right)\right]`.

    Parameters
    ----------
    limits : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``limits`` is set to ``None``, then the cumulative integration is
        performed over the entire input signal, with :math:`u_i<u_f`. Otherwise,
        if ``limits`` is set to a pair of floating-point numbers, then
        ``limits[0]`` and ``limits[1]`` are :math:`u_i` and :math:`u_f`
        respectively, in the same units of the signal space coordinate
        :math:`u`. Note that the function represented by the input signal is
        assumed to be equal to zero everywhere outside of the bounds of said
        input signal.
    num_bins : `int` | `None`, optional
        ``num_bins`` must either be a positive integer or of the `NoneType`: if
        the former, then the dimension of the signal space of the output signal
        ``output_signal`` is set to ``num_bins``; if the latter, then the
        dimension of the signal space of ``output_signal`` is set to
        ``input_signal.data[-1]``, where ``input_signal`` is the input
        ``hyperspy`` signal.
    normalize : `bool`, optional
        If ``normalize`` is set to ``True``, then the cumulative integral is
        normalized such that
        :math:`\text{CDF}_{\text{1D}}\left(u=u_f\right)`. Otherwise, if
        ``normalize`` is set to ``False``, then the cumulative integral is not
        normalized.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"CDF("+
        input_signal.metadata.General.title+")"``, where ``input_signal`` is the
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of 
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"limits": _check_and_convert_limits_v1,
         "num_bins": _check_and_convert_num_bins_v1,
         "normalize": _check_and_convert_normalize,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"limits": _pre_serialize_limits,
         "num_bins": _pre_serialize_num_bins,
         "normalize": _pre_serialize_normalize,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"limits": _de_pre_serialize_limits,
         "num_bins": _de_pre_serialize_num_bins,
         "normalize": _de_pre_serialize_normalize,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 limits=None,
                 num_bins=None,
                 normalize=False,
                 title=None):
        ctor_params = {"limits": limits,
                       "num_bins": num_bins,
                       "normalize": normalize,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def cumulatively_integrate_1d(input_signal, optional_params=None):
    r"""Integrate cumulatively a given input 1D ``hyperspy`` signal.

    This current Python function assumes that the input 1D ``hyperspy`` signal
    samples from a mathematical
    function :math:`F_{\mathbf{m}}\left(u\right)` which is piecewise continuous
    in :math:`u`, where :math:`u` is the signal space coordinate of the input
    signal, and :math:`\mathbf{m}` is a vector of integers representing the
    navigation indices of the input signal. The Python function approximates the
    cumulative integral of :math:`F_{\mathbf{m}}\left(u\right)` given the input
    signal. We define the cumulative integral of
    :math:`F_{\mathbf{m}}\left(u\right)` as

    .. math ::
	\text{CDF}_{\text{1D}}\left(u\right)&=\int_{u_{i}}^{u}du^{\prime}\,
        F_{\mathbf{m}}\left(u^{\prime}\right),
        \\&\quad u\in\left[\min\left(u_{i},u_{f}\right),
        \max\left(u_{i},u_{f}\right)\right],
        :label: cumulative_integral_1d__2

    where :math:`u_i` and :math:`u_f` specify the interval over which cumulative
    integration is performed, the interval being
    :math:`\left[\min\left(u_{i},u_{f}\right),
    \max\left(u_{i},u_{f}\right)\right]`.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.complex_signal1d.ComplexSignal1D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalCumulative1dIntegrationParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalCumulative1dIntegrationParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalCumulative1dIntegrationParams`, then the default 
        values of the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal1d.Signal1D` | :class:`hyperspy._signals.complex_signal1d.ComplexSignal1D`
        The output ``hyperspy`` signal that samples the annular integral of the
        input signal ``input_signal``. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_1D_input_signal(input_signal)
    limits, num_bins, normalize, title = \
        _check_and_convert_optional_params_v2(optional_params, input_signal)
    if title is None:
        title = _default_title(input_signal, "CDF(", ")")

    u_coords = _u_coords_1d(input_signal)
    beg_u_coord_idx, end_u_coord_idx = _beg_and_end_u_coord_indices(u_coords,
                                                                    limits)

    navigation_dims = input_signal.data.shape[:-1]
    output_data_shape = list(navigation_dims) + [num_bins]
    output_data = np.zeros(output_data_shape, dtype=input_signal.data.dtype)

    num_patterns = int(np.prod(navigation_dims))
    for pattern_idx in range(num_patterns):
        navigation_indices = np.unravel_index(pattern_idx, navigation_dims)
        input_datasubset = input_signal.data[navigation_indices]
            
        kwargs = {"input_datasubset": input_datasubset,
                  "u_coords": u_coords,
                  "beg_u_coord_idx": beg_u_coord_idx,
                  "end_u_coord_idx": end_u_coord_idx,
                  "limits": limits,
                  "num_bins": num_bins,
                  "normalize": normalize}
        bin_coords, output_datasubset = \
            _cumulatively_integrate_input_datasubset(**kwargs)
        output_data[navigation_indices] = output_datasubset
        
    metadata = {"General": {"title": title}, "Signal": dict()}
    if np.isrealobj(output_data):
        output_signal = hyperspy.signals.Signal1D(data=output_data,
                                                  metadata=metadata)
    else:
        output_signal = hyperspy.signals.ComplexSignal1D(data=output_data,
                                                         metadata=metadata)
    _update_output_signal_axes_v1(output_signal, bin_coords, input_signal)

    return output_signal



def _check_1D_input_signal(input_signal):
    accepted_types = (hyperspy.signals.Signal1D,
                      hyperspy.signals.ComplexSignal1D)
    kwargs = {"obj": input_signal,
              "obj_name": "input_signal",
              "accepted_types": accepted_types}
    czekitout.check.if_instance_of_any_accepted_types(**kwargs)

    return input_signal



def _check_and_convert_optional_params_v2(optional_params, input_signal):
    if optional_params is None:
        optional_params = OptionalCumulative1dIntegrationParams()
    if not isinstance(optional_params, OptionalCumulative1dIntegrationParams):
        raise TypeError(_check_and_convert_optional_params_v2_err_msg_1)

    limits = optional_params.core_attrs["limits"]
    limits = _check_and_convert_limits_v2(limits, input_signal)

    num_bins = optional_params.core_attrs["num_bins"]
    num_bins = _check_and_convert_num_bins_v2(num_bins, input_signal)

    normalize = optional_params.core_attrs["normalize"]

    title = optional_params.core_attrs["title"]

    return limits, num_bins, normalize, title



def _check_and_convert_limits_v2(limits, signal):
    if limits is None:
        u_coords = _u_coords_1d(signal)
        u_i = np.amin(u_coords)
        u_f = np.amax(u_coords)
        limits = (u_i, u_f)

    return limits



def _u_coords_1d(signal):
    offset = signal.axes_manager[-1].offset
    scale = signal.axes_manager[-1].scale
    size = signal.axes_manager[-1].size
    u_coords = offset + scale * np.arange(size)

    return u_coords



def _beg_and_end_u_coord_indices(u_coords, limits):
    du = u_coords[1] - u_coords[0]
    
    idx_1 = np.abs(u_coords-min(limits)).argmin()
    idx_1 = max(idx_1-1, 0) if du > 0 else min(idx_1+1, u_coords.size-1)
    
    idx_2 = np.abs(u_coords-max(limits)).argmin()
    idx_2 = min(idx_2+1, u_coords.size-1) if du > 0 else max(idx_2-1, 0)
    
    beg_u_coord_idx = min(idx_1, idx_2)
    end_u_coord_idx = max(idx_1, idx_2)

    return beg_u_coord_idx, end_u_coord_idx



def _cumulatively_integrate_input_datasubset(input_datasubset,
                                             u_coords,
                                             beg_u_coord_idx,
                                             end_u_coord_idx,
                                             limits,
                                             num_bins,
                                             normalize):
    du = u_coords[1] - u_coords[0]
    x = u_coords[beg_u_coord_idx:end_u_coord_idx+1]
    y = input_datasubset[beg_u_coord_idx:end_u_coord_idx+1]

    if du < 0:
        x = x[::-1]
        y = y[::-1]

    F = scipy.interpolate.interp1d(x, y,
                                   kind="cubic", copy=False,
                                   bounds_error=False, fill_value=0,
                                   assume_sorted=True)

    u_i, u_f = limits
    bin_coords = np.linspace(u_i, u_f, num_bins)
    F_data = F(bin_coords)

    output_datasubset = abs(du) * np.cumsum(F_data)
    
    if normalize:
        output_datasubset /= output_datasubset[-1]

    return bin_coords, output_datasubset



def _check_and_convert_window_dims_v1(ctor_params):
    window_dims = ctor_params["window_dims"]
    
    if window_dims is not None:
        try:
            window_dims = \
                czekitout.convert.to_pair_of_positive_ints(window_dims,
                                                           "window_dims")
        except:
            raise TypeError(_check_and_convert_window_dims_v1_err_msg_1)

    return window_dims



def _check_and_convert_pad_mode(ctor_params):
    pad_mode = ctor_params["pad_mode"]
    
    if pad_mode is not None:
        try:
            pad_mode = czekitout.convert.to_str_from_str_like(pad_mode,
                                                              "pad_mode")
        except:
            raise TypeError(_check_and_convert_pad_mode_err_msg_1)

        accepted_values = ("no-padding", "wrap", "zeros")
        if pad_mode not in accepted_values:
            raise ValueError(_check_and_convert_pad_mode_err_msg_1)

    return pad_mode



def _check_and_convert_apply_symmetric_mask(ctor_params):
    apply_symmetric_mask = \
        czekitout.convert.to_bool(ctor_params["apply_symmetric_mask"],
                                  "apply_symmetric_mask")

    return apply_symmetric_mask



def _pre_serialize_window_dims(window_dims):
    serializable_rep = window_dims
    
    return serializable_rep



def _pre_serialize_pad_mode(pad_mode):
    serializable_rep = pad_mode
    
    return serializable_rep



def _pre_serialize_apply_symmetric_mask(apply_symmetric_mask):
    serializable_rep = apply_symmetric_mask
    
    return serializable_rep



def _de_pre_serialize_window_dims(serializable_rep):
    window_dims = serializable_rep

    return window_dims



def _de_pre_serialize_pad_mode(serializable_rep):
    pad_mode = serializable_rep

    return pad_mode



def _de_pre_serialize_apply_symmetric_mask(serializable_rep):
    apply_symmetric_mask = serializable_rep

    return apply_symmetric_mask



class OptionalCroppingParams(fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function :func:`empix.crop`.

    The Python function :func:`empix.crop` applies a series of optional
    transformations to a given input 2D ``hyperspy`` signal. Let us denote the
    input 2D ``hyperspy`` signal by :math:`F_{\mathbf{m}; l_x, l_y}`, where
    :math:`l_x` and :math:`l_y` are integers indexing the sampled horizontal and
    vertical coordinates respectively in the signal space of the input signal,
    and :math:`\mathbf{m}` is a vector of integers representing the navigation
    indices of the input signal. The Python function effectively does the
    following:

    1. Copies the input signal and optionally pads the copy along the horizontal
    and vertical axes in signal space according to the parameter ``pad_mode``;

    2. Constructs a cropping window in the signal space of the (optionally
    padded) copy of the input signal, with the cropping window dimensions being
    determined by the parameter ``window_dims``;

    3. Shifts the center of the cropping window to coordinates determined by the
    parameter ``center``;

    4. Shifts the center of the cropping window again to the coordinates of the
    pixel closest to the aforementioned coordinates in the previous step;

    5. Crops the (optionally padded) copy of the input signal along the
    horizontal and vertical dimensions of the signal space according to the
    placement of the cropping window in the previous two steps;

    6. Optionally applies a symmetric mask to the cropped signal resulting from
    the previous step according to the parameter ``apply_symmetric_mask``.

    See the description below of the optional parameters for more details.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``center`` is set to ``None``, then the center of the cropping window
        is set to the signal space coordinates corresponding to the center
        signal space pixel of the original input signal. Otherwise, if
        ``center`` is set to a pair of floating-point numbers, then
        ``center[0]`` and ``center[1]`` specify the horizontal and vertical
        signal space coordinates of the center of the cropping window prior to
        the subpixel shift to the nearest pixel, in the same units of the
        corresponding axes of the input signal
    window_dims : `array_like` (`int`, shape=(2,)) | `None`, optional
        If ``window_dims`` is set to ``None``, then the dimensions of the
        cropping window are set to the dimensions of the signal space of the
        input signal.  Otherwise, if ``window_dims`` is set to a pair of
        positive integers, then ``window_dims[0]`` and ``window_dims[1]``
        specify the horizontal and vertical dimensions of the cropping window in
        units of pixels.
    pad_mode : ``"no-padding"`` | ``"wrap"`` | ``"zeros"``, optional
        If ``pad_mode`` is set to ``"no-padding"``, then no padding is performed
        prior to the crop. If ``pad_mode`` is set to ``"wrap"``, then the copy
        of the input signal is effectively padded along the horizontal and
        vertical axes in signal space by tiling the copy both horizontally and
        vertically in signal space such that the cropping window lies completely
        within the signal space boundaries of the resulting padded signal upon
        performing the crop. If ``pad_mode`` is set to ``"zeros"``, then the
        copy of the input signal is effectively padded with zeros such that the
        cropping window lies completely within the signal space boundaries of
        the resulting padded signal upon performing the crop.
    apply_symmetric_mask : `bool`, optional
        If ``apply_symmetric_mask`` is set to ``True`` and ``pad_mode`` to
        ``"zeros"``, then for every signal space pixel in the cropped signal
        that has a value of zero due to padding and a corresponding pixel with
        coordinates equal to the former after a rotation of 180 degrees about
        the center of the cropped signal, the latter i.e. the aforementioned
        corresponding pixel is effectively set to zero. The effective procedure
        is equivalent to applying a symmetric mask. Otherwise, no mask is
        effectively applied after cropping.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Cropped "+
        input_signal.metadata.General.title``, where ``input_signal`` is the
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center_v1,
         "window_dims": _check_and_convert_window_dims_v1,
         "pad_mode": _check_and_convert_pad_mode,
         "apply_symmetric_mask": _check_and_convert_apply_symmetric_mask,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "window_dims": _pre_serialize_window_dims,
         "pad_mode": _pre_serialize_pad_mode,
         "apply_symmetric_mask": _pre_serialize_apply_symmetric_mask,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "window_dims": _de_pre_serialize_window_dims,
         "pad_mode": _de_pre_serialize_pad_mode,
         "apply_symmetric_mask": _de_pre_serialize_apply_symmetric_mask,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 center=None,
                 window_dims=None,
                 pad_mode="no-padding",
                 apply_symmetric_mask=False,
                 title=None):
        ctor_params = {"center": center,
                       "window_dims": window_dims,
                       "pad_mode": pad_mode,
                       "apply_symmetric_mask": apply_symmetric_mask,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def crop(input_signal, optional_params=None):
    r"""Crop a given input 2D ``hyperspy`` signal.

    This current Python function applies a series of optional transformations to
    a given input 2D ``hyperspy`` signal. Let us denote the input 2D
    ``hyperspy`` signal by :math:`F_{\mathbf{m}; l_x, l_y}`, where :math:`l_x`
    and :math:`l_y` are integers indexing the sampled horizontal and vertical
    coordinates respectively in the signal space of the input signal, and
    :math:`\mathbf{m}` is a vector of integers representing the navigation
    indices of the input signal. The Python function effectively does the
    following:

    1. Copies the input signal and optionally pads the copy along the horizontal
    and vertical axes in signal space according to the parameter ``pad_mode``;

    2. Constructs a cropping window in the signal space of the (optionally
    padded) copy of the input signal, with the cropping window dimensions being
    determined by the parameter ``window_dims``;

    3. Shifts the center of the cropping window to coordinates determined by the
    parameter ``center``;

    4. Shifts the center of the cropping window again to the coordinates of the
    pixel closest to the aforementioned coordinates in the previous step;

    5. Crops the (optionally padded) copy of the input signal along the
    horizontal and vertical dimensions of the signal space according to the
    placement of the cropping window in the previous two steps;

    6. Optionally applies a symmetric mask to the cropped signal resulting from
    the previous step according to the parameter ``apply_symmetric_mask``.

    See the description below of the optional parameters for more details.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalCroppingParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalCroppingParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalCroppingParams`, then the default values of the
        optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The output ``hyperspy`` signal that results from the applied
        transformations, described above. Note that except for the title, the
        metadata of the output signal is determined from the metadata of the
        input signal.

    """
    _check_2D_input_signal(input_signal)
    center, window_dims, pad_mode, apply_symmetric_mask, title = \
        _check_and_convert_optional_params_v3(optional_params, input_signal)
    if title is None:
        title = _default_title(input_signal, "Cropped ", "")

    temp_cropping_params = _temp_cropping_params(input_signal,
                                                 center,
                                                 window_dims,
                                                 pad_mode,
                                                 apply_symmetric_mask)

    navigation_dims = input_signal.data.shape[:-2]
    navigation_indices = np.unravel_index(0, navigation_dims)
    input_datasubset = input_signal.data[navigation_indices]
    output_datasubset = _crop_datasubset(input_datasubset, temp_cropping_params)
    output_data_shape = list(navigation_dims) + list(output_datasubset.shape)
    output_data = np.zeros(output_data_shape, dtype=input_signal.data.dtype)
    
    if np.prod(output_data.shape) == 0:
        raise ValueError(_crop_err_msg_1)

    output_data[navigation_indices] = output_datasubset
    num_patterns = int(np.prod(navigation_dims))
    for pattern_idx in range(1, num_patterns):
        navigation_indices = np.unravel_index(pattern_idx, navigation_dims)
        input_datasubset = input_signal.data[navigation_indices]
            
        kwargs = {"input_datasubset": input_datasubset,
                  "temp_cropping_params": temp_cropping_params}
        output_datasubset = _crop_datasubset(**kwargs)
        output_data[navigation_indices] = output_datasubset
        
    metadata = {"General": {"title": title}, "Signal": dict()}
    if np.isrealobj(output_data):
        output_signal = hyperspy.signals.Signal2D(data=output_data,
                                                  metadata=metadata)
    else:
        output_signal = hyperspy.signals.ComplexSignal2D(data=output_data,
                                                         metadata=metadata)

    output_axes_offsets = temp_cropping_params["output_axes_offsets"]
    _update_output_signal_axes_v2(output_signal,
                                  input_signal,
                                  output_axes_offsets)

    return output_signal



def _check_and_convert_optional_params_v3(optional_params, input_signal):
    if optional_params is None:
        optional_params = OptionalCroppingParams()
    if not isinstance(optional_params, OptionalCroppingParams):
        raise TypeError(_check_and_convert_optional_params_v3_err_msg_1)

    center = optional_params.core_attrs["center"]
    center = _check_and_convert_center_v3(center, input_signal)

    window_dims = optional_params.core_attrs["window_dims"]
    window_dims = _check_and_convert_window_dims_v2(window_dims, input_signal)

    pad_mode = optional_params.core_attrs["pad_mode"]

    apply_symmetric_mask = optional_params.core_attrs["apply_symmetric_mask"]

    title = optional_params.core_attrs["title"]

    return center, window_dims, pad_mode, apply_symmetric_mask, title



def _check_and_convert_center_v3(center, signal):
    if center is None:
        n_v, n_h = signal.data.shape[-2:]
    
        h_scale = signal.axes_manager[-2].scale
        v_scale = signal.axes_manager[-1].scale

        h_offset = signal.axes_manager[-2].offset
        v_offset = signal.axes_manager[-1].offset

        center = (h_offset + h_scale*((n_h-1)//2),
                  v_offset + v_scale*((n_v-1)//2))

    return center



def _check_and_convert_window_dims_v2(window_dims, signal):
    if window_dims is None:
        n_v, n_h = signal.data.shape[-2:]
        window_dims = (n_h, n_v)

    return window_dims



def _temp_cropping_params(input_signal,
                          center,
                          window_dims,
                          pad_mode,
                          apply_symmetric_mask):
    navigation_dims = input_signal.data.shape[:-2]
    navigation_indices = np.unravel_index(0, navigation_dims)
    input_datasubset = input_signal.data[navigation_indices]

    center_in_pixel_coords = _center_in_pixel_coords(input_signal, center)
    
    pad_width_1, multi_dim_slice_1 = \
        _pad_width_1_and_multi_dim_slice_1(input_datasubset,
                                           center_in_pixel_coords,
                                           window_dims,
                                           pad_mode)
    pad_width_2, multi_dim_slice_2 = \
        _pad_width_2_and_multi_dim_slice_2(input_datasubset,
                                           center_in_pixel_coords,
                                           window_dims,
                                           pad_mode)
    mode = "wrap" if pad_mode == "wrap" else "constant"
    symmetric_mask = _symmetric_mask(input_datasubset,
                                     center_in_pixel_coords,
                                     window_dims,
                                     pad_mode,
                                     apply_symmetric_mask)
    output_axes_offsets = _offsets_after_cropping(input_signal,
                                                  center_in_pixel_coords,
                                                  window_dims,
                                                  pad_mode)

    temp_cropping_params = {"pad_width_1": pad_width_1,
                            "pad_width_2": pad_width_2,
                            "multi_dim_slice_1": multi_dim_slice_1,
                            "multi_dim_slice_2": multi_dim_slice_2,
                            "mode": mode,
                            "symmetric_mask": symmetric_mask,
                            "output_axes_offsets": output_axes_offsets}

    return temp_cropping_params



def _center_in_pixel_coords(input_signal, center):
    h_scale = input_signal.axes_manager[-2].scale
    v_scale = input_signal.axes_manager[-1].scale

    h_offset = input_signal.axes_manager[-2].offset
    v_offset = input_signal.axes_manager[-1].offset

    center_in_pixel_coords = np.round(((center[0] - h_offset) / h_scale,
                                       (center[1] - v_offset) / v_scale))

    return center_in_pixel_coords



def _pad_width_1_and_multi_dim_slice_1(input_datasubset,
                                       center_in_pixel_coords,
                                       window_dims,
                                       pad_mode):
    shift = (center_in_pixel_coords[::-1]
             - ((np.array(input_datasubset.shape)-1) // 2))

    pad_width_1 = [(0, 0), (0, 0)]
    multi_dim_slice_1 = [slice(None), slice(None)]

    mode = "wrap" if pad_mode == "wrap" else "constant"
    
    for idx in (0, 1):
        s = int(np.sign(shift[idx]+0.5))
        n = input_datasubset.shape[idx]
        c = (n-1)//2
        L = window_dims[::-1][idx]
        p = (L-1)//2
        if pad_mode != "no-padding":
            pad_width_1[idx] = (0, int(s * shift[idx]))[::s]
        if s < 0:
            k = 0 if pad_mode != "no-padding" else max((L-p)-(n-c), 0)
            end = max(n + pad_width_1[idx][0] + shift[idx] + k, 0)
            multi_dim_slice_1[idx] = slice(None, int(end))
        else:
            k = 0 if pad_mode != "no-padding" else max(p-c, 0)
            start = max(shift[idx] - k, 0)
            multi_dim_slice_1[idx] = slice(int(start), None)
    multi_dim_slice_1 = tuple(multi_dim_slice_1)

    return pad_width_1, multi_dim_slice_1



def _pad_width_2_and_multi_dim_slice_2(input_datasubset,
                                       center_in_pixel_coords,
                                       window_dims,
                                       pad_mode):
    pad_width_2 = [(0, 0), (0, 0)]
    multi_dim_slice_2 = [slice(None), slice(None)]

    shift = (center_in_pixel_coords[::-1]
             - ((np.array(input_datasubset.shape)-1) // 2))

    dims_diff = window_dims[::-1] - np.array(input_datasubset.shape)
    
    for idx in (0, 1):
        c = (input_datasubset.shape[idx]-1)//2
        L = window_dims[::-1][idx]

        i = (min(shift[idx], 0)
             if ((pad_mode == "no-padding") and dims_diff[idx] < 0)
             else 0)
            
        start = int(max((c+1) - (L//2) - (L%2) + i, 0))
        stop = int(max((c+1) - (L//2) - (L%2) + i + L, 0))

        if dims_diff[idx] > 0:
            temp = np.array([dims_diff[idx]//2]*2)
            if dims_diff[idx]%2 == 1:
                temp += np.array([(input_datasubset.shape[idx]+1)%2,
                                  (window_dims[::-1][idx]+1)%2])
            temp = (int(temp[0]), int(temp[1]))
            pad_width_2[idx] = (0, 0) if pad_mode == "no-padding" else temp

            if pad_mode != "no-padding":
                start = None
                stop = None
                        
        multi_dim_slice_2[idx] = slice(start, stop)

    multi_dim_slice_2 = tuple(multi_dim_slice_2)

    return pad_width_2, multi_dim_slice_2



def _symmetric_mask(input_datasubset,
                    center_in_pixel_coords,
                    window_dims,
                    pad_mode,
                    apply_symmetric_mask):
    if (pad_mode == "zeros") and apply_symmetric_mask:
        symmetric_mask = np.ones(window_dims[::-1], dtype=bool)
        multi_dim_slice = [slice(None), slice(None)]
        
        for idx in range(2):
            n = input_datasubset.shape[idx]
            m = window_dims[::-1][idx]
        
            a = m // 2
            b = ((m-1) // 2) - (m%2)
            c = center_in_pixel_coords[::-1][idx]
            c_L = c - (n%2)
            c_R = c + 1
            d = min(a, c_L+1, n-c_R)
        
            start = b - d + 1
            stop = start + 2*d + (m%2)
            multi_dim_slice[idx] = slice(int(start), int(stop))
        
        multi_dim_slice = tuple(multi_dim_slice)
        symmetric_mask[multi_dim_slice] = False
    else:
        symmetric_mask = None

    return symmetric_mask



def _offsets_after_cropping(input_signal,
                            center_in_pixel_coords,
                            window_dims,
                            pad_mode):
    navigation_dims = input_signal.data.shape[:-2]
    navigation_indices = np.unravel_index(0, navigation_dims)
    num_axes = len(input_signal.data.shape)
    input_datasubset = input_signal.data[navigation_indices]

    shift = (center_in_pixel_coords[::-1]
             - ((np.array(input_datasubset.shape)-1) // 2))

    scales = [input_signal.axes_manager[idx].scale for idx in range(num_axes)]
    offsets = [input_signal.axes_manager[idx].offset for idx in range(num_axes)]
    adjusted_center = [offsets[idx] + center_in_pixel_coords[idx]*scales[idx]
                       for idx in (-2, -1)]

    for idx in (-2, -1):
        n = input_datasubset.shape[::-1][idx]
        c = (n-1)//2
        L = window_dims[idx]
        p = (L-1)//2
        if pad_mode != "no-padding":
            m = p
        else:
            k = max(p-c, 0)
            m = min(p, c+min(shift[::-1][idx], k))
        offsets[idx] = adjusted_center[idx] - m*scales[idx]

    return offsets



def _crop_datasubset(input_datasubset, temp_cropping_params):
    pad_width_1 = temp_cropping_params["pad_width_1"]
    pad_width_2 = temp_cropping_params["pad_width_2"]
    multi_dim_slice_1 = temp_cropping_params["multi_dim_slice_1"]
    multi_dim_slice_2 = temp_cropping_params["multi_dim_slice_2"]
    mode = temp_cropping_params["mode"]
    symmetric_mask = temp_cropping_params["symmetric_mask"]

    shifted_datasubset = np.pad(input_datasubset,
                                pad_width_1,
                                mode=mode)[multi_dim_slice_1]
    cropped_datasubset = np.pad(shifted_datasubset,
                                pad_width_2,
                                mode=mode)[multi_dim_slice_2]

    if symmetric_mask is not None:
        cropped_datasubset *= (~symmetric_mask)

    return cropped_datasubset



def _update_output_signal_axes_v2(output_signal,
                                  input_signal,
                                  output_axes_offsets):
    num_axes = len(input_signal.data.shape)
        
    for idx, output_axis_offset in enumerate(output_axes_offsets):
        input_axis = \
            input_signal.axes_manager[idx]
        output_axis_size = \
            output_signal.axes_manager[idx].size
        new_output_axis = \
            hyperspy.axes.UniformDataAxis(size=output_axis_size,
                                          scale=input_axis.scale,
                                          offset=output_axis_offset,
                                          units=input_axis.units)
        
        output_signal.axes_manager[idx].update_from(new_output_axis)
        output_signal.axes_manager[idx].name = input_axis.name

    return None



def _check_and_convert_block_dims(ctor_params):
    block_dims = ctor_params["block_dims"]
    block_dims = czekitout.convert.to_pair_of_positive_ints(block_dims,
                                                           "block_dims")
    
    return block_dims



def _check_and_convert_padding_const(ctor_params):
    padding_const = ctor_params["padding_const"]
    padding_const = czekitout.convert.to_float(padding_const, "padding_const")

    return padding_const



def _check_and_convert_downsample_mode(ctor_params):
    downsample_mode = ctor_params["downsample_mode"]
    downsample_mode = czekitout.convert.to_str_from_str_like(downsample_mode,
                                                             "downsample_mode")
    
    accepted_values = ("sum", "mean", "median", "amin", "amax")
    if downsample_mode not in accepted_values:
        raise ValueError(_check_and_convert_downsample_mode_err_msg_1)

    return downsample_mode



def _pre_serialize_block_dims(block_dims):
    serializable_rep = block_dims
    
    return serializable_rep



def _pre_serialize_padding_const(padding_const):
    serializable_rep = padding_const
    
    return serializable_rep



def _pre_serialize_downsample_mode(downsample_mode):
    serializable_rep = downsample_mode
    
    return serializable_rep



def _de_pre_serialize_block_dims(serializable_rep):
    block_dims = serializable_rep

    return block_dims



def _de_pre_serialize_padding_const(serializable_rep):
    padding_const = serializable_rep

    return padding_const



def _de_pre_serialize_downsample_mode(serializable_rep):
    downsample_mode = serializable_rep

    return downsample_mode



class OptionalDownsamplingParams(fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function 
    :func:`empix.downsample`.

    The Python function :func:`empix.downsample` copies a given input 2D
    ``hyperspy`` signal and downsamples the copy along the axes in signal space.
    The Python function effectively does the following: 

    1. Groups the pixels of the copy of the input signal into so-called
    downsampling blocks along the axes in signal space, with dimensions
    determined by the parameter ``block_dims``, padding the copy with a constant
    value of ``padding_const`` in the case that either the horizontal or
    vertical dimensions of the signal space of the original input signal are not
    divisible by the corresponding dimensions of the downsampling blocks.

    2. For each downsampling block, the Python function calls a ``numpy``
    function determined by the parameter ``downsample_mode``, wherein the input
    is the array data of the downsampling block, and the output is the value of
    the corresponding pixel of the downsampled signal.

    Parameters
    ----------
    block_dims : `array_like` (`int`, shape=(2,)), optional
        ``block_dims[0]`` and ``block_dims[1]`` specify the horizontal and
        vertical dimensions of the downsampling blocks in units of pixels.
    padding_const : `float`, optional
        ``padding_const`` is the padding constant to be applied in the case that
        either the horizontal or vertical dimensions of the signal space of the
        original input signal are not divisible by the corresponding dimensions
        of the downsampling blocks.
    downsample_mode : ``"sum"`` | ``"mean"`` | ``"median"`` | ``"amin"`` | ``"amax"``, optional
        ``downsample_mode == numpy_func.__name__`` where ``numpy_func`` is the 
        ``numpy`` function to be applied to the downsampling blocks.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Downsampled "+
        input_signal.metadata.General.title``, where ``input_signal`` is the
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"block_dims": _check_and_convert_block_dims,
         "padding_const": _check_and_convert_padding_const,
         "downsample_mode": _check_and_convert_downsample_mode,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"block_dims": _pre_serialize_block_dims,
         "padding_const": _pre_serialize_padding_const,
         "downsample_mode": _pre_serialize_downsample_mode,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"block_dims": _de_pre_serialize_block_dims,
         "padding_const": _de_pre_serialize_padding_const,
         "downsample_mode": _de_pre_serialize_downsample_mode,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 block_dims=(2, 2),
                 padding_const=0,
                 downsample_mode="sum",
                 title=None):
        ctor_params = {"block_dims": block_dims,
                       "padding_const": padding_const,
                       "downsample_mode": downsample_mode,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def downsample(input_signal, optional_params=None):
    r"""Downsample a given input 2D ``hyperspy`` signal.

    This current Python function copies a given input 2D
    ``hyperspy`` signal and downsamples the copy along the axes in signal space.
    The Python function effectively does the following: 

    1. Groups the pixels of the copy of the input signal into so-called
    downsampling blocks along the axes in signal space, with dimensions
    determined by the parameter ``block_dims``, padding the copy with a constant
    value of ``padding_const`` in the case that either the horizontal or
    vertical dimensions of the signal space of the original input signal are not
    divisible by the corresponding dimensions of the downsampling blocks.

    2. For each downsampling block, the Python function calls a ``numpy``
    function determined by the parameter ``downsample_mode``, wherein the input
    is the array data of the downsampling block, and the output is the value of
    the corresponding pixel of the downsampled signal.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalDownsamplingParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalDownsamplingParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalDownsamplingParams`, then the default values of
        the optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The output ``hyperspy`` signal that results from the downsampling. Note
        that except for the title, the metadata of the output signal is
        determined from the metadata of the input signal.

    """
    _check_2D_input_signal(input_signal)
    if optional_params is None:
        optional_params = OptionalDownsamplingParams()
    if not isinstance(optional_params, OptionalDownsamplingParams):
        raise TypeError(_downsample_err_msg_1)
    title = optional_params.core_attrs["title"]
    if title is None:
        title = _default_title(input_signal, "Downsampled ", "")

    navigation_dims = input_signal.data.shape[:-2]
    navigation_indices = np.unravel_index(0, navigation_dims)
    input_datasubset = input_signal.data[navigation_indices]
    output_datasubset = _downsample_datasubset(input_datasubset,
                                               optional_params)
    output_data_shape = list(navigation_dims) + list(output_datasubset.shape)
    output_data = np.zeros(output_data_shape, dtype=input_signal.data.dtype)

    output_data[navigation_indices] = output_datasubset
    num_patterns = int(np.prod(navigation_dims))
    for pattern_idx in range(1, num_patterns):
        navigation_indices = np.unravel_index(pattern_idx, navigation_dims)
        input_datasubset = input_signal.data[navigation_indices]
            
        kwargs = {"input_datasubset": input_datasubset,
                  "optional_params": optional_params}
        output_datasubset = _downsample_datasubset(**kwargs)
        output_data[navigation_indices] = output_datasubset
        
    metadata = {"General": {"title": title}, "Signal": dict()}
    if np.isrealobj(output_data):
        output_signal = hyperspy.signals.Signal2D(data=output_data,
                                                  metadata=metadata)
    else:
        output_signal = hyperspy.signals.ComplexSignal2D(data=output_data,
                                                         metadata=metadata)

    _update_output_signal_axes_v3(output_signal, input_signal, optional_params)

    return output_signal



def _downsample_datasubset(input_datasubset, optional_params):
    downsample_mode = optional_params.core_attrs["downsample_mode"]
    if downsample_mode == "sum":
        func = np.sum
    elif downsample_mode == "mean":
        func = np.mean
    elif downsample_mode == "median":
        func = np.median
    elif downsample_mode == "amin":
        func = np.amin
    else:
        func = np.amax
    
    kwargs = {"block_size": optional_params.core_attrs["block_dims"][::-1],
              "cval": optional_params.core_attrs["padding_const"],
              "func": func}

    downsampled_datasubset = skimage.measure.block_reduce(input_datasubset,
                                                          **kwargs)

    return downsampled_datasubset



def _update_output_signal_axes_v3(output_signal, input_signal, optional_params):
    num_axes = len(input_signal.data.shape)

    sizes = [output_signal.axes_manager[idx].scale for idx in range(num_axes)]
    scales = [input_signal.axes_manager[idx].scale for idx in range(num_axes)]
    offsets = [input_signal.axes_manager[idx].offset for idx in range(num_axes)]
    units = [input_signal.axes_manager[idx].units for idx in range(num_axes)]
    names = [input_signal.axes_manager[idx].name for idx in range(num_axes)]

    for idx in (-2, -1):
        L = optional_params.core_attrs["block_dims"][idx]
        offsets[idx] += 0.5*(L-1)*scales[idx]
        scales[idx] *= L
        
    for idx in range(num_axes):
        new_output_axis = hyperspy.axes.UniformDataAxis(size=sizes[idx],
                                                        scale=scales[idx],
                                                        offset=offsets[idx],
                                                        units=units[idx])
        output_signal.axes_manager[idx].update_from(new_output_axis)
        output_signal.axes_manager[idx].name = names[idx]

    return None



def _check_and_convert_new_signal_space_sizes_v1(ctor_params):
    new_signal_space_sizes = ctor_params["new_signal_space_sizes"]
    
    if new_signal_space_sizes is not None:
        try:
            kwargs = {"obj": new_signal_space_sizes,
                      "obj_name": "new_signal_space_sizes"}
            new_signal_space_sizes = \
                czekitout.convert.to_pair_of_positive_ints(**kwargs)
        except:
            err_msg = _check_and_convert_new_signal_space_sizes_v1_err_msg_1
            raise TypeError(err_msg)

    return new_signal_space_sizes



def _check_and_convert_new_signal_space_scales_v1(ctor_params):
    new_signal_space_scales = ctor_params["new_signal_space_scales"]

    if new_signal_space_scales is not None:
        err_msg = _check_and_convert_new_signal_space_scales_v1_err_msg_1
        try:
            kwargs = {"obj": new_signal_space_scales,
                      "obj_name": "new_signal_space_scales"}
            new_signal_space_scales = \
                czekitout.convert.to_pair_of_floats(**kwargs)
        except:
            raise TypeError(err_msg)

        if np.prod(new_signal_space_scales) == 0:
            raise ValueError(err_msg)

    return new_signal_space_scales



def _check_and_convert_new_signal_space_offsets_v1(ctor_params):
    new_signal_space_offsets = ctor_params["new_signal_space_offsets"]

    if new_signal_space_offsets is not None:
        try:
            kwargs = {"obj": new_signal_space_offsets,
                      "obj_name": "new_signal_space_offsets"}
            new_signal_space_offsets = \
                czekitout.convert.to_pair_of_floats(**kwargs)
        except:
            err_msg = _check_and_convert_new_signal_space_offsets_v1_err_msg_1
            raise TypeError(err_msg)

    return new_signal_space_offsets



def _check_and_convert_spline_degrees(ctor_params):
    spline_degrees = ctor_params["spline_degrees"]
    spline_degrees = \
        czekitout.convert.to_pair_of_positive_ints(spline_degrees,
                                                   "spline_degrees")

    if (spline_degrees[0] > 5) or (spline_degrees[1] > 5):
        raise ValueError(_check_and_convert_spline_degrees_err_msg_1)

    return spline_degrees



def _check_and_convert_interpolate_polar_cmpnts(ctor_params):
    interpolate_polar_cmpnts = ctor_params["interpolate_polar_cmpnts"]
    interpolate_polar_cmpnts = \
        czekitout.convert.to_bool(interpolate_polar_cmpnts,
                                  "interpolate_polar_cmpnts")

    return interpolate_polar_cmpnts



def _pre_serialize_new_signal_space_sizes(new_signal_space_sizes):
    serializable_rep = new_signal_space_sizes
    
    return serializable_rep



def _pre_serialize_new_signal_space_scales(new_signal_space_scales):
    serializable_rep = new_signal_space_scales
    
    return serializable_rep



def _pre_serialize_new_signal_space_offsets(new_signal_space_offsets):
    serializable_rep = new_signal_space_offsets
    
    return serializable_rep



def _pre_serialize_spline_degrees(spline_degrees):
    serializable_rep = spline_degrees
    
    return serializable_rep



def _pre_serialize_interpolate_polar_cmpnts(interpolate_polar_cmpnts):
    serializable_rep = interpolate_polar_cmpnts
    
    return serializable_rep



def _de_pre_serialize_new_signal_space_sizes(serializable_rep):
    new_signal_space_sizes = serializable_rep

    return new_signal_space_sizes



def _de_pre_serialize_new_signal_space_scales(serializable_rep):
    new_signal_space_scales = serializable_rep

    return new_signal_space_scales



def _de_pre_serialize_new_signal_space_offsets(serializable_rep):
    new_signal_space_offsets = serializable_rep

    return new_signal_space_offsets



def _de_pre_serialize_spline_degrees(serializable_rep):
    spline_degrees = serializable_rep

    return spline_degrees



def _de_pre_serialize_interpolate_polar_cmpnts(serializable_rep):
    interpolate_polar_cmpnts = serializable_rep

    return interpolate_polar_cmpnts



class OptionalResamplingParams(fancytypes.PreSerializableAndUpdatable):
    r"""The set of optional parameters for the function :func:`empix.resample`.

    The Python function :func:`empix.resample` copies a given input 2D
    ``hyperspy`` signal and resamples the copy along the axes in signal space by
    interpolating the original input signal using bivariate spines. Effectively,
    :func:`empix.resample` resamples the input signal.

    Parameters
    ----------
    new_signal_space_sizes : array_like` (`int`, shape=(2,)) | `None`, optional
        If ``new_signal_space_sizes`` is set to ``None``, then
        ``output_signal.data.shape`` will be equal to
        ``input_signal.data.shape``, where ``input_signal`` is the input signal,
        and ``output_signal`` is the output signal to result from the
        resampling. Otherwise, if ``new_signal_space_sizes`` is set to a pair of
        positive integers, then ``output_signal.data.shape[-2]`` and
        ``output_signal.data.shape[-1]`` will be equal to
        ``new_signal_space_sizes[0]`` and ``new_signal_space_sizes[1]``
        respectively.
    new_signal_space_scales : `array_like` (`float`, shape=(2,)) | `None`, optional
        Continuing from above, if ``new_signal_space_scales`` is set to
        ``None``, then ``output_signal.axes_manager[-2].scale`` and
        ``output_signal.axes_manager[-1].scale`` will be equal to
        ``input_signal.axes_manager[-2].scale`` and
        ``input_signal.axes_manager[-1].scale`` respectively. If
        ``new_signal_space_scales`` is set to a pair of non-zero floating-point
        numbers, then ``output_signal.axes_manager[-2].scale`` and
        ``output_signal.axes_manager[-1].scale`` will be equal to
        ``new_signal_space_scales[0]`` and ``new_signal_space_scales[1]``
        respectively. Otherwise, an error is raised.
    new_signal_space_offsets : `array_like` (`float`, shape=(2,)) | `None`, optional
        Continuing from above, if ``new_signal_space_offsets`` is set to
        ``None``, then ``output_signal.axes_manager[-2].offset`` and
        ``output_signal.axes_manager[-1].offset`` will be equal to
        ``input_signal.axes_manager[-2].offset`` and
        ``input_signal.axes_manager[-1].offset`` respectively. Otherwise, if
        ``new_signal_space_offsets`` is set to a pair of floating-point numbers,
        then ``output_signal.axes_manager[-2].offset`` and
        ``output_signal.axes_manager[-1].offset`` will be equal to
        ``new_signal_space_offsets[0]`` and ``new_signal_space_offsets[1]``
        respectively.
    spline_degrees : `array_like` (`int`, shape=(2,)), optional
        ``spline_degrees[0]`` and ``spline_degrees[1]`` are the horizontal and
        vertical degrees of the bivariate splines used to interpolate the input
        signal. Note that ``spline_degrees`` is expected to satisfy both
        ``1<=spline_degrees[0]<=5`` and ``1<=spline_degrees[1]<=5``.
    interpolate_polar_cmpnts : `bool`, optional
        If ``interpolate_polar_cmpnts`` is set to ``True``, then the polar
        components of the input signal are separately interpolated. Otherwise,
        if ``interpolate_polar_cmpnts`` is set to ``False``, then the real and
        imaginary components of the input signal are separately interpolated.
        Note that if the input signal is real-valued, then this parameter is
        effectively ignored.
    title : `str` | `None`, optional
        If ``title`` is set to ``None``, then the title of the output signal
        ``output_signal`` is set to ``"Resampled "+
        input_signal.metadata.General.title``, where ``input_signal`` is the
        input ``hyperspy`` signal.  Otherwise, if ``title`` is a `str`, then the
        ``output_signal.metadata.General.title`` is set to the value of
        ``title``.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"new_signal_space_sizes": \
         _check_and_convert_new_signal_space_sizes_v1,
         "new_signal_space_scales": \
         _check_and_convert_new_signal_space_scales_v1,
         "new_signal_space_offsets": \
         _check_and_convert_new_signal_space_offsets_v1,
         "spline_degrees": _check_and_convert_spline_degrees,
         "interpolate_polar_cmpnts": \
         _check_and_convert_interpolate_polar_cmpnts,
         "title": _check_and_convert_title_v1}

    _pre_serialization_funcs = \
        {"new_signal_space_sizes": _pre_serialize_new_signal_space_sizes,
         "new_signal_space_scales": _pre_serialize_new_signal_space_scales,
         "new_signal_space_offsets": _pre_serialize_new_signal_space_offsets,
         "spline_degrees": _pre_serialize_spline_degrees,
         "interpolate_polar_cmpnts": _pre_serialize_interpolate_polar_cmpnts,
         "title": _pre_serialize_title}

    _de_pre_serialization_funcs = \
        {"new_signal_space_sizes": _de_pre_serialize_new_signal_space_sizes,
         "new_signal_space_scales": _de_pre_serialize_new_signal_space_scales,
         "new_signal_space_offsets": _de_pre_serialize_new_signal_space_offsets,
         "spline_degrees": _de_pre_serialize_spline_degrees,
         "interpolate_polar_cmpnts": _de_pre_serialize_interpolate_polar_cmpnts,
         "title": _de_pre_serialize_title}

    def __init__(self,
                 new_signal_space_sizes=None,
                 new_signal_space_scales=None,
                 new_signal_space_offsets=None,
                 spline_degrees=(3, 3),
                 interpolate_polar_cmpnts=True,
                 title=None):
        ctor_params = {"new_signal_space_sizes": new_signal_space_sizes,
                       "new_signal_space_scales": new_signal_space_scales,
                       "new_signal_space_offsets": new_signal_space_offsets,
                       "spline_degrees": spline_degrees,
                       "interpolate_polar_cmpnts": interpolate_polar_cmpnts,
                       "title": title}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def resample(input_signal, optional_params=None):
    r"""Resample a given input 2D ``hyperspy`` signal via interpolation.

    This current Python function copies a given input 2D ``hyperspy`` signal and
    resamples the copy along the axes in signal space by interpolating the
    original input signal using bivariate spines. Effectively,
    :func:`empix.resample` resamples the input signal.

    Parameters
    ----------
    input_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The input ``hyperspy`` signal.
    optional_params : :class:`empix.OptionalResamplingParams` | `None`, optional
        The set of optional parameters. See the documentation for the class
        :class:`empix.OptionalResamplingParams` for details. If
        ``optional_params`` is set to ``None``, rather than an instance of
        :class:`empix.OptionalResamplingParams`, then the default values of the
        optional parameters are chosen.

    Returns
    -------
    output_signal : :class:`hyperspy._signals.signal2d.Signal2D` | :class:`hyperspy._signals.complex_signal2d.ComplexSignal2D`
        The output ``hyperspy`` signal that results from the resampling. Note
        that except for the title, the metadata of the output signal is
        determined from the metadata of the input signal.

    """
    _check_2D_input_signal(input_signal)
    optional_params = _check_and_convert_optional_params_v4(optional_params,
                                                            input_signal)
    title = optional_params.core_attrs["title"]
    if title is None:
        title = _default_title(input_signal, "Resampled ", "")

    temp_resampling_params = _temp_resampling_params(input_signal,
                                                     optional_params)

    navigation_dims = input_signal.data.shape[:-2]
    navigation_indices = np.unravel_index(0, navigation_dims)
    input_datasubset = input_signal.data[navigation_indices]
    output_datasubset = _resample_datasubset(input_datasubset,
                                             temp_resampling_params)
    output_data_shape = list(navigation_dims) + list(output_datasubset.shape)
    output_data = np.zeros(output_data_shape, dtype=input_signal.data.dtype)

    output_data[navigation_indices] = output_datasubset
    num_patterns = int(np.prod(navigation_dims))
    for pattern_idx in range(1, num_patterns):
        navigation_indices = np.unravel_index(pattern_idx, navigation_dims)
        input_datasubset = input_signal.data[navigation_indices]
            
        kwargs = {"input_datasubset": input_datasubset,
                  "temp_resampling_params": temp_resampling_params}
        output_datasubset = _resample_datasubset(**kwargs)
        output_data[navigation_indices] = output_datasubset
        
    metadata = {"General": {"title": title}, "Signal": dict()}
    if np.isrealobj(output_data):
        output_signal = hyperspy.signals.Signal2D(data=output_data,
                                                  metadata=metadata)
    else:
        output_signal = hyperspy.signals.ComplexSignal2D(data=output_data,
                                                         metadata=metadata)

    _update_output_signal_axes_v4(output_signal, input_signal, optional_params)

    return output_signal



def _check_and_convert_optional_params_v4(optional_params, input_signal):
    if optional_params is None:
        optional_params = OptionalResamplingParams()
    if not isinstance(optional_params, OptionalResamplingParams):
        raise TypeError(_check_and_convert_optional_params_v4_err_msg_1)

    new_signal_space_sizes = \
        optional_params.core_attrs["new_signal_space_sizes"]
    new_signal_space_sizes = \
        _check_and_convert_new_signal_space_sizes_v2(new_signal_space_sizes,
                                                     input_signal)

    new_signal_space_scales = \
        optional_params.core_attrs["new_signal_space_scales"]
    new_signal_space_scales = \
        _check_and_convert_new_signal_space_scales_v2(new_signal_space_scales,
                                                      input_signal)

    new_signal_space_offsets = \
        optional_params.core_attrs["new_signal_space_offsets"]
    new_signal_space_offsets = \
        _check_and_convert_new_signal_space_offsets_v2(new_signal_space_offsets,
                                                       input_signal)

    core_attr_subset = {"new_signal_space_sizes": new_signal_space_sizes,
                        "new_signal_space_scales": new_signal_space_scales,
                        "new_signal_space_offsets": new_signal_space_offsets}
    optional_params.update(core_attr_subset)

    return optional_params



def _check_and_convert_new_signal_space_sizes_v2(new_signal_space_sizes,
                                                 signal):
    if new_signal_space_sizes is None:
        n_v, n_h = signal.data.shape[-2:]
        new_signal_space_sizes = (n_h, n_v)

    return new_signal_space_sizes



def _check_and_convert_new_signal_space_scales_v2(new_signal_space_scales,
                                                  signal):
    if new_signal_space_scales is None:
        new_signal_space_scales = (signal.axes_manager[-2].scale,
                                   signal.axes_manager[-1].scale)

    return new_signal_space_scales



def _check_and_convert_new_signal_space_offsets_v2(new_signal_space_offsets,
                                                   signal):
    if new_signal_space_offsets is None:
        new_signal_space_offsets = (signal.axes_manager[-2].offset,
                                    signal.axes_manager[-1].offset)

    return new_signal_space_offsets



def _temp_resampling_params(input_signal, optional_params):
    old_sizes = [input_signal.axes_manager[idx].size for idx in (-2, -1)]
    old_scales = [input_signal.axes_manager[idx].scale for idx in (-2, -1)]
    old_offsets = [input_signal.axes_manager[idx].offset for idx in (-2, -1)]

    new_sizes = optional_params.core_attrs["new_signal_space_sizes"]
    new_scales = optional_params.core_attrs["new_signal_space_scales"]
    new_offsets = optional_params.core_attrs["new_signal_space_offsets"]

    h_old = np.sign(old_scales[0]) * (old_offsets[0]
                                      + old_scales[0]*np.arange(old_sizes[0]))
    v_old = np.sign(old_scales[1]) * (old_offsets[1]
                                      + old_scales[1]*np.arange(old_sizes[1]))
    
    h_new = np.sign(old_scales[0]) * (new_offsets[0]
                                      + new_scales[0]*np.arange(new_sizes[0]))
    v_new = np.sign(old_scales[1]) * (new_offsets[1]
                                      + new_scales[1]*np.arange(new_sizes[1]))
    s_h_new = int(np.sign(h_new[1]-h_new[0]))
    s_v_new = int(np.sign(v_new[1]-v_new[0]))
    h_new = np.sort(h_new)
    v_new = np.sort(v_new)

    spline_degrees = optional_params.core_attrs["spline_degrees"]
    interpolate_polar_cmpnts = \
        optional_params.core_attrs["interpolate_polar_cmpnts"]

    temp_resampling_params = \
        {"h_old": h_old,
         "v_old": v_old,
         "h_new": h_new,
         "v_new": v_new,
         "s_h_new": int(np.sign(h_new[1]-h_new[0])),
         "s_v_new": int(np.sign(v_new[1]-v_new[0])),
         "spline_degrees": spline_degrees,
         "interpolate_polar_cmpnts": interpolate_polar_cmpnts}

    return temp_resampling_params



def _resample_datasubset(input_datasubset, temp_resampling_params):
    kwargs = {"x": temp_resampling_params["v_old"],
              "y": temp_resampling_params["h_old"],
              "z": None,
              "bbox": [None, None, None, None],
              "kx": temp_resampling_params["spline_degrees"][1],
              "ky": temp_resampling_params["spline_degrees"][0],
              "s": 0}

    v_new = temp_resampling_params["v_new"]
    h_new = temp_resampling_params["h_new"]

    if np.isrealobj(input_datasubset):
        kwargs["z"] = input_datasubset
        f = scipy.interpolate.RectBivariateSpline(**kwargs)
        resampled_datasubset = f(v_new, h_new)
    else:
        if temp_resampling_params["interpolate_polar_cmpnts"]:
            kwargs["z"] = np.abs(input_datasubset)
            f = scipy.interpolate.RectBivariateSpline(**kwargs)
            mag = f(v_new, h_new)
            kwargs["z"] = np.angle(input_datasubset)
            f = scipy.interpolate.RectBivariateSpline(**kwargs)
            angle = f(v_new, h_new)
            resampled_datasubset = mag * np.exp(1j*angle)
        else:
            kwargs["z"] = np.real(input_datasubset)
            f = scipy.interpolate.RectBivariateSpline(**kwargs)
            real_part = f(v_new, h_new)
            kwargs["z"] = np.imag(input_datasubset)
            f = scipy.interpolate.RectBivariateSpline(**kwargs)
            imag_part = f(v_new, h_new)
            resampled_datasubset = real_part + 1j*imag_part

    s_h_new = temp_resampling_params["s_h_new"]
    s_v_new = temp_resampling_params["s_v_new"]
    resampled_datasubset[:, :] = resampled_datasubset[::s_v_new, ::s_h_new]

    return resampled_datasubset



def _update_output_signal_axes_v4(output_signal, input_signal, optional_params):
    num_axes = len(input_signal.data.shape)

    sizes = [output_signal.axes_manager[idx].scale for idx in range(num_axes)]
    scales = [input_signal.axes_manager[idx].scale for idx in range(num_axes)]
    offsets = [input_signal.axes_manager[idx].offset for idx in range(num_axes)]
    units = [input_signal.axes_manager[idx].units for idx in range(num_axes)]
    names = [input_signal.axes_manager[idx].name for idx in range(num_axes)]

    for idx in (-2, -1):
        scales[idx] = \
              optional_params.core_attrs["new_signal_space_scales"][idx]
        offsets[idx] = \
              optional_params.core_attrs["new_signal_space_offsets"][idx]
        
    for idx in range(num_axes):
        new_output_axis = hyperspy.axes.UniformDataAxis(size=sizes[idx],
                                                        scale=scales[idx],
                                                        offset=offsets[idx],
                                                        units=units[idx])
        output_signal.axes_manager[idx].update_from(new_output_axis)
        output_signal.axes_manager[idx].name = names[idx]

    return None



###########################
## Define error messages ##
###########################

_check_and_convert_title_v1_err_msg_1 = \
    ("The object ``title`` must be `NoneType` or a `str`.")

_check_and_convert_center_v1_err_msg_1 = \
    ("The object ``center`` must be `NoneType` or a pair of real numbers.")
_check_and_convert_center_v2_err_msg_2 = \
    ("The object ``center`` must specify a point within the boundaries of the "
     "input ``hyperspy`` signal.")

_check_and_convert_radial_range_v1_err_msg_1 = \
    ("The object ``radial_range`` must be `NoneType` or a pair of non-negative "
     "real numbers satisfying ``0<=radial_range[0]<=radial_range[1]``.")

_check_and_convert_num_bins_v1_err_msg_1 = \
    ("The object ``num_bins`` must be `NoneType` or a positive `int`.")

_check_and_convert_optional_params_v1_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalAzimuthalAveragingParams`.")

_azimuthally_integrate_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalAzimuthalIntegrationParams`.")

_annularly_average_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalAnnularAveragingParams`.")

_annularly_integrate_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalAnnularIntegrationParams`.")

_check_and_convert_optional_params_v2_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalCumulative1dIntegrationParams`.")

_check_and_convert_window_dims_v1_err_msg_1 = \
    ("The object ``window_dims`` must be `NoneType` or a pair of positive "
     "integers.")

_check_and_convert_pad_mode_err_msg_1 = \
    ("The object ``pad_mode`` must be either ``'no-padding'``, ``'wrap'``, or "
     "``'zeros'``.")

_check_and_convert_optional_params_v3_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalCroppingParams`.")

_crop_err_msg_1 = \
    ("The object ``optional_params`` specifies a crop that yields an output "
     "``hyperspy`` signal with zero elements.")

_check_and_convert_optional_block_reduce_params_err_msg_1 = \
    ("The key ``'{}'`` in the dictionary ``optional_block_reduce_params`` is "
     "invalid: the only accepted keys are ``'block_size'``, ``'func'``, "
     "``'cval'``, and ``'func_kwargs'``.")

_check_and_convert_optional_block_reduce_params_err_msg_2 = \
    ("The object ``optional_block_reduce_params`` specifies an invalid set of "
     "optional parameters for the function "
     ":func:`skimage.measure.block_reduce`: see the traceback above for "
     "further details.")

_check_and_convert_downsample_mode_err_msg_1 = \
    ("The object ``downsample_mode`` must be either ``'sum'``, ``'mean'``, "
     "``'median'``, ``'amin'``, or ``'amax'``.")

_downsample_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalDownsamplingParams`.")

_check_and_convert_new_signal_space_sizes_v1_err_msg_1 = \
    ("The object ``new_signal_space_sizes`` must be `NoneType` or a pair of "
     "positive integers.")

_check_and_convert_new_signal_space_scales_v1_err_msg_1 = \
    ("The object ``new_signal_space_scales`` must be `NoneType` or a pair of "
     "non-zero real numbers.")

_check_and_convert_new_signal_space_offsets_v1_err_msg_1 = \
    ("The object ``new_signal_space_offsets`` must be `NoneType` or a pair of "
     "real numbers.")

_check_and_convert_spline_degrees_err_msg_1 = \
    ("The object ``spline_degrees`` must be a pair of positive integers where "
     "each integer is less than or equal to ``5``.")

_check_and_convert_optional_params_v4_err_msg_1 = \
    ("The object ``optional_params`` must be `NoneType` or an instance of the "
     "class `OptionalResamplingParams`.")
