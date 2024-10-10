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

    __cpp_name__ = 'RooDataSet'

    @cpp_signature(
        "RooDataSet(std::string_view name, std::string_view title, const RooArgSet& vars, const RooCmdArg& arg1, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},const RooCmdArg& arg5={},"
        "    const RooCmdArg& arg6={},const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;"
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
        "    const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "    const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "    const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;"
    )
    def plotOnXY(self, *args, **kwargs):
        r"""The RooDataSet::plotOnXY() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooDataSet.plotOnXY` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._plotOnXY(*args, **kwargs)

    @staticmethod
    def from_numpy(data, variables, name=None, title=None, weight_name=None):
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
            dataset = ROOT.RooDataSet(name, title, variables, WeightVar=weight_name)

        def log_warning(s):
            """Log a string to the RooFit message log for the WARNING level on
            the DataHandling topic.
            """
            log = ROOT.RooMsgService.instance().log(dataset, ROOT.RooFit.WARNING, ROOT.RooFit.DataHandling)
            b = bytes(s, "utf-8")
            log.write(b, len(b))
            log.write("\n", 1)

        range_mask = np.ones_like(list(data.values())[0], dtype=bool)

        def in_range(arr, variable):
            # For categories, we need to check whether the elements of the
            # array are in the set of category state indices
            if variable.isCategory():
                return np.isin(arr, [state.second for state in variable])

            return (arr >= variable.getMin()) & (arr <= variable.getMax())

        # Get a mask that filters out all entries that are outside the variable definition range
        range_mask = np.logical_and.reduce([in_range(data[v.GetName()], v) for v in variables])
        # If all entries are in the range, we don't need a mask
        if range_mask.all():
            range_mask = None

        def select_range_and_change_type(arr, dtype):
            if range_mask is not None:
                arr = arr[range_mask]
            arr = arr if arr.dtype == dtype else np.array(arr, dtype=dtype)
            # Make sure that the array is contiguous so we can std::copy() it.
            # In the implementation of ascontiguousarray(), no copy is done if
            # the array is already contiguous, which is exactly what we want.
            return np.ascontiguousarray(arr)

        def copy_to_dataset(store_list, np_type, c_type, type_size_in_bytes):
            for real in store_list:
                vec = real.data()
                arg = real.bufArg()
                arr = select_range_and_change_type(data[arg.GetName()], np_type)

                vec.resize(len(arr))

                # The next part works because arr is guaranteed to be C-contiguous
                beg = arr.ctypes.data_as(ctypes.POINTER(c_type))
                n_bytes = type_size_in_bytes * len(arr)
                void_p = ctypes.cast(beg, ctypes.c_voidp).value + n_bytes
                end = ctypes.cast(void_p, ctypes.POINTER(c_type))
                ROOT.std.copy(beg, end, vec.begin())

        copy_to_dataset(dataset.store().realStoreList(), np.float64, ctypes.c_double, 8)
        copy_to_dataset(dataset.store().catStoreList(), np.int32, ctypes.c_int, 4)

        dataset.store().recomputeSumWeight()

        n_entries = None

        for real in dataset.store().realStoreList():
            if n_entries is None:
                n_entries = real.size()
            assert n_entries == real.size()

        for cat in dataset.store().catStoreList():
            assert n_entries == cat.size()

        if range_mask is not None:
            n_out_of_range = len(range_mask) - range_mask.sum()
            log_warning("RooDataSet.from_numpy({0}) Ignored {1} out-of-range events".format(name, n_out_of_range))

        return dataset

    def to_numpy(self, copy=True):
        """Export a RooDataSet to a dictionary of numpy arrays.

        Args:
            copy (bool): If False, the data will not be copied. Use with
                         caution, as the numpy arrays and the RooAbsData now
                         own the same memory. If the dataset uses a
                         RooTreeDataStore, there will always be a copy and the
                         copy argument is ignored.

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

        return data

    @staticmethod
    def from_pandas(df, variables, name=None, title=None, weight_name=None):
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

        Returns:
            RooDataSet
        """
        import ROOT

        data = {}
        for column in df:
            data[column] = df[column].values
        return ROOT.RooDataSet.from_numpy(data, variables=variables, name=name, title=title, weight_name=weight_name)

    def to_pandas(self):
        """Export a RooDataSet to a pandas DataFrame.

        Args:

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
