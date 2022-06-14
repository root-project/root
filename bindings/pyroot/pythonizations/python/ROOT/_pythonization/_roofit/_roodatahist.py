# Authors:
# * Jonas Rembser 06/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _kwargs_to_roocmdargs, _dict_to_std_map, cpp_signature


class RooDataHist(object):
    r"""Constructor of RooDataHist takes a RooCmdArg as argument also supports keyword arguments.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    dh = ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(x), ROOT.RooFit.Import("SampleA", histo))

    # With keyword arguments:
    dh = ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(x), Import=("SampleA", histo))
    \endcode
    """

    @cpp_signature(
        [
            "RooDataHist(std::string_view name, std::string_view title, const RooArgList& vars, RooCategory& indexCat, std::map<std::string,TH1*> histMap, Double_t initWgt=1.0) ;",
            "RooDataHist(std::string_view name, std::string_view title, const RooArgList& vars, RooCategory& indexCat, std::map<std::string,RooDataHist*> dhistMap, Double_t wgt=1.0) ;",
        ]
    )
    def __init__(self, *args, **kwargs):
        r"""The RooDataHist constructor is pythonized with the command argument pythonization and for converting python dict to std::map.
        The keywords must correspond to the CmdArg of the constructor function.
        The instances in dict must correspond to the template argument in std::map of the constructor.
        """
        # Redefinition of `RooDataHist` constructor for keyword arguments and converting python dict to std::map.
        if len(args) > 4 and isinstance(args[4], dict):
            args = list(args)
            args[4] = _dict_to_std_map(args[4], {"std::string": ["RooDataHist*", "TH1*"]})

        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    def plotOn(self, *args, **kwargs):
        # Redefinition of `RooDataHist.plotOn` for keyword arguments.
        # The keywords must correspond to the CmdArg of the `plotOn` function.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOn(*args, **kwargs)

    @property
    def shape(self):
        shape = []

        for i_var, var in enumerate(self.get()):
            shape.append(var.numBins())

        def get_len_from_shape():
            n_size_check = 1
            for n in shape:
                n_size_check = n_size_check * n
            return n_size_check

        # Cross check that the product of the shape values is equal to the full
        # length.
        assert get_len_from_shape() == len(self)

        return tuple(shape)

    def __len__(self):
        return self.numEntries()

    def _to_array(self, buffer):
        # Helper to create a numpy array from a raw array pointer.
        #
        # Note: The data is copied.
        #
        # Args:
        #     buffer (cppyy.LowLevelView):
        #         The pointer to the beginning of the array data, usually
        #         obtained from a C++ function that returns a `double *`.
        #
        # Returns:
        #     numpy.ndarray
        import numpy as np

        # Check if buffer is a nullptr by using implicit conversion from
        # `nullptr` to Bool (for some reason comparing with ROOT.nullptr
        # doesn't work).
        if not buffer:
            return None
        a = np.copy(np.frombuffer(buffer, dtype=np.float64, count=len(self)))
        return a.reshape(self.shape)

    def _var_is_category(self):
        # Returns a list with a Bool for each dimension of the histogram that
        # flags whether the variable in this dimension is a RooAbsCategory.
        import ROOT

        out = []

        for i_var, var in enumerate(self.get()):
            if isinstance(var, ROOT.RooAbsReal):
                out.append(False)
            elif isinstance(var, ROOT.RooAbsCategory):
                out.append(True)
            else:
                raise TypeError("Vaiables in RooDataHist should either be RooAbsReal or RooAbsCategory.")

        return out

    def _weight_error_low(self):
        # Returns the low weight errors as numpy arrays.
        return self._to_array(self.wgtErrLoArray())

    def _weight_error_high(self):
        # Returns the high weight errors as numpy arrays.
        return self._to_array(self.wgtErrHiArray())

    def _weight_error(self):
        # Returns the low and high weight errors as numpy arrays.
        return self.weight_error_low(), self.weight_error_high()

    def _weights_squared_sum(self):
        # Returns the sum of squared weights that were used to fill each bin.
        return self._to_array(self.sumW2Array())

    @staticmethod
    def from_numpy(hist_weights, variables, bins=None, ranges=None, weights_squared_sum=None, name=None, title=None):
        r"""Create a RooDataHist from numpy arrays.

        Note: The argument stucture was inspired by numpy.histogramdd.

        Args:
            hist_weights (numpy.ndarray): The multidimensional histogram bin
                                          weights.
            bins (list): The bin specification, where each element is either:
                           * a numpy array describing the monotonically
                             increasing bin edges along each dimension.
                           * a scalar value for the number of bins (in this
                             case, the corresponding item in the `ranges`
                             argument must be filled)
                           * `None` for a category dimension or if you want to
                              use the default binning of the RooFit variable
            variables (RooArgSet, or list/tuple of RooAbsArgs):
                Specification of the variables in the RooDataHist, will be
                forwarded to the RooDataHist constructor. Both real values and
                categories are supported.
            ranges (list): An optional list specifying the variable range
                           limits. Each element is either:
                             * `None` if a full bin edges array is given to
                               `bins` or for a category dimension
                             * a tuple with two values corresponding to the
                               minimum and maximum values
            weights_squared_sum (numpy.ndarray):
                The sum of squared weights of the original samples that were
                used to fill the histogram. If the input weights are from a
                weighted histogram, this parameter is no longer optional.
            name (str): Name of the RooDataSet, `None` is equivalent to an
                        empty string.
            title (str): Title of the RooDataSet, `None` is equivalent to an
                         empty string.

        Returns:
            RooDataHist
        """
        import ROOT
        import numpy as np

        name = "" if name is None else name
        title = "" if title is None else title

        n_dim = len(variables)

        if bins is None:
            bins = [None] * n_dim

        if ranges is None:
            ranges = [None] * n_dim

        # name for internal binning that is created for the RooDataHist to adapt
        binning_name = "_roodataset_from_numpy_binning"

        def set_binning(var, bins, ranges):
            if bins is None:
                binning = var.getBinning().clone()
            elif np.isscalar(bins):
                if ranges is None:
                    raise ValueError("If a scalar number of bins is given, you must also provide an explicit range.")
                binning = ROOT.RooUniformBinning(ranges[0], ranges[1], bins)
            else:
                binning = ROOT.RooBinning(len(bins) - 1, bins)

            var.setBinning(binning, binning_name)

        for i_var, var in enumerate(variables):
            if isinstance(var, ROOT.RooAbsReal):
                set_binning(var, bins[i_var], ranges[i_var])

        datahist = ROOT.RooDataHist(name, title, variables, binning_name)

        if len(datahist) != len(hist_weights):
            raise ValueError("Length of hist_weights array doesn't match the size of the RooDataHist.")

        if weights_squared_sum is None:
            if not np.allclose(hist_weights, hist_weights.round()):
                raise ValueError(
                    "Your input histogram has non-integer weights! You must also provide weights_squared_sum to privide the complete information to RooDataHist.from_numpy()."
                )
        else:
            if len(datahist) != len(weights_squared_sum):
                raise ValueError("Length of weights_squared_sum array doesn't match the size of the RooDataHist.")

        for i_bin in range(len(datahist)):
            if not weights_squared_sum is None:
                # some reverse-computation that can't be avoided with the current C++ RooDataHist interface
                wgt_err = np.sqrt(weights_squared_sum[i_bin])
            else:
                wgt_err = -1

            datahist.set(i_bin, hist_weights[i_bin], wgt_err)

        return datahist

    def to_numpy(self):
        r"""Converts the weights and bin edges of a RooDataHist to numpy arrays.

        Note: The output stucture was inspired by numpy.histogramdd.

        Returns:
            weight (numpy.ndarray): The weights for each histrogram bin.
            bin_edges (list): A list of `n_dim` arrays describing the bin edges
                              for each dimension. For dimensions of category
                              types, the list element is `None`.
        """
        import ROOT
        import numpy as np

        bin_edges = []

        var_is_category = self._var_is_category()

        for i_var, var in enumerate(self.get()):

            if var_is_category[i_var]:
                bin_edges.append(None)
            else:
                binning = self.getBinnings()[i_var]
                bin_edges.append(
                    np.copy(np.frombuffer(binning.array(), dtype=np.float64, count=binning.numBoundaries()))
                )

        return self._to_array(self.weightArray()), bin_edges
