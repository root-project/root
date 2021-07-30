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


from ._utils import _kwargs_to_roocmdargs, cpp_signature


class RooDataSet(object):
    r"""Some member functions of RooDataSet that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooDataSet() constructor and RooDataSet::plotOnXY.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    dxy = ROOT.RooDataSet("dxy", "dxy", ROOT.RooArgSet(x, y), ROOT.RooFit.StoreError(ROOT.RooArgSet(x, y)))

    # With keyword arguments:
    dxy = ROOT.RooDataSet("dxy", "dxy", ROOT.RooArgSet(x, y), StoreError=(ROOT.RooArgSet(x, y)))
    \endcode
    """

    @cpp_signature(
        "RooDataSet(std::string_view name, std::string_view title, const RooArgSet& vars, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg(),"
        "    const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),"
        "    const RooCmdArg& arg6=RooCmdArg(),const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg()) ;"
    )
    def __init__(self, *args, **kwargs):
        r"""The RooDataSet constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the constructor.
        """
        # Redefinition of `RooDataSet` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    @cpp_signature(
        "RooPlot *RooDataSet::plotOnXY(RooPlot* frame,"
        "    const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
        "    const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
        "    const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),"
        "    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;"
    )
    def plotOnXY(self, *args, **kwargs):
        r"""The RooDataSet::plotOnXY() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooDataSet.plotOnXY` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOnXY(*args, **kwargs)

    @staticmethod
    def from_numpy(data, variables, name=None, title=None, weight_name=None, clip_to_limits=True):
        """Create a RooDataSet from a dictionary of numpy arrays.
        Args:
            data (dict): Dictionary with strings as keys and numpy arrays as
                         values, to be imported into the RooDataSet.
            variables (RooArgSet, or list/tuple of RooAbsArgs):
                Specification of the variables in the RooDataSet, will be
                forwarded to the RooDataSet constructor. Both real values and
                categories are supported.
            name (str): Name of the RooDataSet, `None` is equivalent to an
                        empty string.
            title (str): Title of the RooDataSet, `None` is equivalent to an
                         empty string.
            weight_name (str): Key of the array in `data` that will be used for
                               the dataset weights.
            clip_to_limits (bool): When entries are added to a RooDataSet, the
                                   standard RooFit behavior is to clip the
                                   values to the limits specified by the
                                   binning of the RooFit variables. To save
                                   computational cost, you can disable this
                                   clipping if not necessary in your workflow.

        Returns:
            RooDataSet
        """
        import ROOT
        import numpy as np
        import ctypes

        name = "" if name is None else name
        title = "" if title is None else title

        if weight_name is None:
            dataset = ROOT.RooDataSet(name, title, variables)
        else:
            dataset = ROOT.RooDataSet(name, title, variables, weight_name)

        for real in dataset.store().realStoreList():
            vec = real.data()
            arg = real.bufArg()
            arr = data[arg.GetName()]

            if clip_to_limits:
                arr = np.clip(arr, a_min=arg.getMin(), a_max=arg.getMax())

            arr = arr if arr.dtype == np.float64 else np.array(arr, dtype=np.float64)
            vec.resize(len(arr))

            beg = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            void_p = ctypes.cast(beg, ctypes.c_voidp).value + 8 * len(arr)
            end = ctypes.cast(void_p, ctypes.POINTER(ctypes.c_double))
            ROOT.std.copy(beg, end, vec.begin())

        for cat in dataset.store().catStoreList():
            vec = cat.data()
            arr = data[cat.bufArg().GetName()]
            arr = arr if arr.dtype == np.int32 else np.array(arr, dtype=np.int32)
            vec.resize(len(arr))

            beg = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            void_p = ctypes.cast(beg, ctypes.c_voidp).value + 4 * len(arr)
            end = ctypes.cast(void_p, ctypes.POINTER(ctypes.c_int))
            ROOT.std.copy(beg, end, vec.begin())

        dataset.store().recomputeSumWeight()

        n_entries = None

        for real in dataset.store().realStoreList():
            if n_entries is None:
                n_entries = real.size()
            assert n_entries == real.size()

        for cat in dataset.store().catStoreList():
            assert n_entries == cat.size()

        return dataset

    def to_numpy(self, copy=True, compute_derived_weight=False):
        """Export a RooDataSet to a dictinary of numpy arrays.

        Args:
            copy (bool): If False, the data will not be copied. Use with
                         caution, as the numpy arrays and the RooAbsData now
                         own the same memory. If the dataset uses a
                         RooTreeDataStore, there will always be a copy and the
                         copy argument is ignored.
            compute_derived_weight (bool): Sometimes, the weight variable is
                not stored in the dataset, but it is a derived variable like a
                RooFormulaVar. If the compute_derived_weight is True, the
                weights will be computed in this case and also stored in the
                output. Switched off by default because the computation is
                relatively expensive.

        Returns:
            dict: A dictionary with the variable or weight names as keys and
                  the numpy arrays as values.
        """
        import ROOT
        import numpy as np

        data = {}

        if isinstance(self.store(), ROOT.RooVectorDataStore):
            for name, array in self.store().to_numpy(copy=copy).items():
                data[name] = array
        elif isinstance(self.store(), ROOT.RooTreeDataStore):
            # first create a VectorDataStore so we can read arrays
            store = self.store()
            variables = store.get()
            store_name = store.GetName()
            tmp_store = ROOT.RooVectorDataStore(store, variables, store_name)
            for name, array in tmp_store.to_numpy(copy=copy).items():
                data[name] = array
        else:
            raise RuntimeError(
                "Exporting RooDataSet to numpy arrays failed. The data store type "
                + self.store().__class__.__name__
                + " is not supported."
            )

        # Special case where the weight is a derived variable (e.g. a RooFormulaVar).
        # We don't want to miss putting the weight in the output arrays, so we
        # are forced to iterate over the dataset to compute the weight.
        if compute_derived_weight:
            # Check if self.weightVar() is not a nullptr by using implicit
            # conversion from `nullptr` to Bool.
            if self.weightVar():
                wgt_var_name = self.weightVar().GetName()
                if not wgt_var_name in data:
                    weight_array = np.zeros(self.numEntries(), dtype=np.float64)
                    for i in range(self.numEntries()):
                        self.get(i)
                        weight_array[i] = self.weight()
                    data[wgt_var_name] = weight_array

        return data

    @staticmethod
    def from_pandas(df, variables, name=None, title=None, weight_name=None, clip_to_limits=True):
        """Create a RooDataSet from a pandas DataFrame.
        Args:
            df (pandas.DataFrame): Pandas DataFrame to import.
            variables (RooArgSet, or list/tuple of RooAbsArgs):
                Specification of the variables in the RooDataSet, will be
                forwarded to the RooDataSet constructor. Both real values and
                categories are supported.
            name (str): Name of the RooDataSet, `None` is equivalent to an
                        empty string.
            title (str): Title of the RooDataSet, `None` is equivalent to an
                         empty string.
            weight_name (str): Key of the array in `data` that will be used for
                               the dataset weights.
            clip_to_limits (bool): When entries are added to a RooDataSet, the
                                   standard RooFit behavior is to clip the
                                   values to the limits specified by the
                                   binning of the RooFit variables. To save
                                   computational cost, you can disable this
                                   clipping if not necessary in your workflow.

        Returns:
            RooDataSet
        """
        import ROOT

        data = {}
        for column in df:
            data[column] = df[column].values
        return ROOT.RooDataSet.from_numpy(
            data, variables=variables, name=name, title=title, weight_name=weight_name, clip_to_limits=clip_to_limits
        )

    def to_pandas(self, compute_derived_weight=False):
        """Export a RooDataSet to a pandas DataFrame.

        Args:
            compute_derived_weight (bool): Sometimes, the weight variable is
                not stored in the dataset, but it is a derived variable like a
                RooFormulaVar. If the compute_derived_weight is True, the
                weights will be computed in this case and also stored in the
                output. Switched off by default because the computation is
                relatively expensive.

        Note:
            Pandas copies the data from the numpy arrays when creating a
            DataFrame. That's why we can disable copying in the to_numpy call.

        Returns:
            pandas.DataFrame: A dataframe with the variable or weight names as
                              column names and the a row for each variable or
                              weight in the dataset.
        """
        import ROOT
        import pandas as pd

        return pd.DataFrame(self.to_numpy(copy=False))
