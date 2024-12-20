import importlib
import types
import sys
import os
from functools import partial

import cppyy

import cppyy.ll

from ._application import PyROOTApplication
from ._numbadeclare import _NumbaDeclareDecorator

from ._pythonization import pythonization


class PyROOTConfiguration(object):
    """Class for configuring PyROOT"""

    def __init__(self):
        self.IgnoreCommandLineOptions = True
        self.ShutDown = True
        self.DisableRootLogon = False
        self.StartGUIThread = True


class _gROOTWrapper(object):
    """Internal class to manage lookups of gROOT in the facade.
    This wrapper calls _finalSetup on the facade when it
    receives a lookup, unless the lookup is for SetBatch.
    This allows to evaluate the command line parameters
    before checking if batch mode is on in _finalSetup
    """

    def __init__(self, facade):
        self.__dict__["_facade"] = facade
        self.__dict__["_gROOT"] = cppyy.gbl.ROOT.GetROOT()

    def __getattr__(self, name):
        if name != "SetBatch" and self._facade.__dict__["gROOT"] != self._gROOT:
            self._facade._finalSetup()
        return getattr(self._gROOT, name)

    def __setattr__(self, name, value):
        return setattr(self._gROOT, name, value)


def _create_rdf_experimental_distributed_module(parent):
    """
    Create the ROOT.RDF.Experimental.Distributed python module.

    This module will be injected into the ROOT.RDF namespace.

    Arguments:
        parent: The ROOT.RDF namespace. Needed to define __package__.

    Returns:
        types.ModuleType: The ROOT.RDF.Experimental.Distributed submodule.
    """
    import DistRDF

    return DistRDF.create_distributed_module(parent)


def _subimport(name):
    # type: (str) -> types.ModuleType
    """
    Import and return the Python module with the input name.

    Helper function for the __reduce__ method of the ROOTFacade class.
    """
    return importlib.import_module(name)


class ROOTFacade(types.ModuleType):
    """Facade class for ROOT module"""

    def __init__(self, module, is_ipython):
        types.ModuleType.__init__(self, module.__name__)

        self.module = module

        self.__all__ = module.__all__
        self.__name__ = module.__name__
        self.__file__ = module.__file__
        self.__cached__ = module.__cached__
        self.__path__ = module.__path__
        self.__doc__ = module.__doc__
        self.__package__ = module.__package__
        self.__loader__ = module.__loader__

        # Inject gROOT global
        self.gROOT = _gROOTWrapper(self)

        # Expose some functionality from CPyCppyy extension module
        self._cppyy_exports = [
            "nullptr",
            "bind_object",
            "as_cobject",
            "addressof",
            "SetMemoryPolicy",
            "kMemoryHeuristics",
            "kMemoryStrict",
            "SetOwnership",
        ]
        for name in self._cppyy_exports:
            setattr(self, name, getattr(cppyy._backend, name))
        # For backwards compatibility
        self.MakeNullPointer = partial(self.bind_object, 0)
        self.BindObject = self.bind_object
        self.AsCObject = self.as_cobject

        # Initialize configuration
        self.PyConfig = PyROOTConfiguration()

        # @pythonization decorator
        self.pythonization = pythonization

        self._is_ipython = is_ipython

        # Redirect lookups to temporary helper methods
        # This lets the user do some actions before all the machinery is in place:
        # - Set batch mode in gROOT
        # - Set options in PyConfig
        self.__class__.__getattr__ = self._getattr
        self.__class__.__setattr__ = self._setattr

    def AddressOf(self, obj):
        # Return an indexable buffer of length 1, whose only element
        # is the address of the object.
        # The address of the buffer is the same as the address of the
        # address of the object

        # addr is the address of the address of the object
        addr = self.addressof(instance=obj, byref=True)

        # Check for 64 bit as suggested here:
        # https://docs.python.org/3/library/platform.html#cross-platform
        out_type = "Long64_t*" if sys.maxsize > 2**32 else "Int_t*"

        # Create a buffer (LowLevelView) from address
        return cppyy.ll.cast[out_type](addr)

    def _fallback_getattr(self, name):
        # Try:
        # - in the global namespace
        # - in the ROOT namespace
        # - in gROOT (ROOT lists such as list of files,
        #   memory mapped files, functions, geometries ecc.)
        # The first two attempts allow to lookup
        # e.g. ROOT.ROOT.Math as ROOT.Math

        # Note that hasattr caches the lookup for getattr
        if hasattr(cppyy.gbl, name):
            return getattr(cppyy.gbl, name)
        elif hasattr(cppyy.gbl.ROOT, name):
            return getattr(cppyy.gbl.ROOT, name)
        else:
            res = self.gROOT.FindObject(name)
            if res:
                return res
        raise AttributeError("Failed to get attribute {} from ROOT".format(name))

    def _register_converters_and_executors(self):

        converter_aliases = {
            "Long64_t": "long long",
            "Long64_t ptr": "long long ptr",
            "Long64_t&": "long long&",
            "const Long64_t&": "const long long&",
            "ULong64_t": "unsigned long long",
            "ULong64_t ptr": "unsigned long long ptr",
            "ULong64_t&": "unsigned long long&",
            "const ULong64_t&": "const unsigned long long&",
            "Float16_t": "float",
            "const Float16_t&": "const float&",
            "Double32_t": "double",
            "Double32_t&": "double&",
            "const Double32_t&": "const double&",
        }

        executor_aliases = {
            "Long64_t": "long long",
            "Long64_t&": "long long&",
            "Long64_t ptr": "long long ptr",
            "ULong64_t": "unsigned long long",
            "ULong64_t&": "unsigned long long&",
            "ULong64_t ptr": "unsigned long long ptr",
            "Float16_t": "float",
            "Float16_t&": "float&",
            "Double32_t": "double",
            "Double32_t&": "double&",
        }

        from libROOTPythonizations import CPyCppyyRegisterConverterAlias, CPyCppyyRegisterExecutorAlias

        for name, target in converter_aliases.items():
            CPyCppyyRegisterConverterAlias(name, target)

        for name, target in executor_aliases.items():
            CPyCppyyRegisterExecutorAlias(name, target)

    def _finalSetup(self):
        # Prevent this method from being re-entered through the gROOT wrapper
        self.__dict__["gROOT"] = cppyy.gbl.ROOT.GetROOT()

        # Make sure the interpreter is initialized once gROOT has been initialized
        cppyy.gbl.TInterpreter.Instance()

        # Setup interactive usage from Python
        self.__dict__["app"] = PyROOTApplication(self.PyConfig, self._is_ipython)
        if not self.gROOT.IsBatch() and self.PyConfig.StartGUIThread:
            self.app.init_graphics()

        # Set memory policy to kUseHeuristics.
        # This restores the default in PyROOT which was changed
        # by new Cppyy
        self.SetMemoryPolicy(self.kMemoryHeuristics)

        # Redirect lookups to cppyy's global namespace
        self.__class__.__getattr__ = self._fallback_getattr
        self.__class__.__setattr__ = lambda self, name, val: setattr(cppyy.gbl, name, val)

        # Register custom converters and executors
        self._register_converters_and_executors()

        # Run rootlogon if exists
        self._run_rootlogon()

    def _getattr(self, name):
        # Special case, to allow "from ROOT import gROOT" w/o starting the graphics
        if name == "__path__":
            raise AttributeError(name)

        self._finalSetup()

        return getattr(self, name)

    def _setattr(self, name, val):
        self._finalSetup()

        return setattr(self, name, val)

    def _execute_rootlogon_module(self, file_path):
        """Execute the 'rootlogon.py' module found at the given 'file_path'"""
        # Could also have used execfile, but import is likely to give fewer surprises
        module_name = "rootlogon"

        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    def _run_rootlogon(self):
        # Run custom logon file (must be after creation of ROOT globals)
        hasargv = hasattr(sys, "argv")
        # -n disables the reading of the logon file, just like with root
        if hasargv and not "-n" in sys.argv and not self.PyConfig.DisableRootLogon:
            file_path_home = os.path.expanduser("~/.rootlogon.py")
            file_path_local = os.path.join(os.getcwd(), ".rootlogon.py")
            if os.path.exists(file_path_home):
                self._execute_rootlogon_module(file_path_home)
            elif os.path.exists(file_path_local):
                self._execute_rootlogon_module(file_path_local)
            else:
                # If the .py version of rootlogon exists, the .C is ignored (the user can
                # load the .C from the .py, if so desired).
                # System logon, user logon, and local logon (skip Rint.Logon)
                name = ".rootlogon.C"
                logons = [
                    os.path.join(str(self.TROOT.GetEtcDir()), "system" + name),
                    os.path.expanduser(os.path.join("~", name)),
                ]
                if logons[-1] != os.path.join(os.getcwd(), name):
                    logons.append(name)
                for rootlogon in logons:
                    if os.path.exists(rootlogon):
                        self.TApplication.ExecuteFile(rootlogon)

    def __reduce__(self):
        # type: () -> types.ModuleType
        """
        Reduction function of the ROOT facade to customize the (pickle)
        serialization step.

        Defines the ingredients needed for a correct serialization of the
        facade, that is a function that imports a Python module and the name of
        that module, which corresponds to this facade's __name__ attribute. This
        method helps serialization tools like `cloudpickle`, especially used in
        distributed environments, that always need to include information about
        the ROOT module in the serialization step. For example, the following
        snippet would not work without this method::

            import ROOT
            import cloudpickle

            def foo():
                return ROOT.TH1F()

            cloudpickle.loads(cloudpickle.dumps(foo))

        In particular, it would raise::

            TypeError: cannot pickle 'ROOTFacade' object
        """
        return _subimport, (self.__name__,)

    # Inject version as __version__ property in ROOT module
    @property
    def __version__(self):
        return self.gROOT.GetVersion()

    # Overload VecOps namespace
    # The property gets the C++ namespace, adds the pythonizations and
    # eventually deletes itself so that following calls go directly
    # to the C++ namespace. This mechanic ensures that we pythonize the
    # namespace lazily.
    @property
    def VecOps(self):
        ns = self._fallback_getattr("VecOps")
        try:
            from ._pythonization._rvec import _AsRVec

            ns.AsRVec = _AsRVec
        except:
            raise Exception("Failed to pythonize the namespace VecOps")
        del type(self).VecOps
        return ns

    # Overload RDF namespace
    @property
    def RDF(self):
        self._finalSetup()
        ns = self._fallback_getattr("RDF")
        try:
            from ._pythonization._rdataframe import _MakeNumpyDataFrame

            # Provide a FromCSV factory method that uses keyword arguments instead of the ROptions config struct.
            # In Python, the RCsvDS::ROptions struct members are available without the leading 'f' and in camelCase,
            # e.g. fDelimiter --> delimiter.
            # We need to keep the parameters of the old FromCSV signature for backward compatibility.
            ns._FromCSV = ns.FromCSV
            def MakeCSVDataFrame(
                    fileName, readHeaders = True, delimiter = ',', linesChunkSize = -1, colTypes = {}, **kwargs):
                options = ns.RCsvDS.ROptions()
                options.fHeaders = readHeaders
                options.fDelimiter = delimiter
                options.fLinesChunkSize = linesChunkSize
                options.fColumnTypes = colTypes
                for key, val in kwargs.items():
                    structMemberName = 'f' + key[0].upper() + key[1:]
                    if hasattr(options, structMemberName):
                        setattr(options, structMemberName, val)
                return ns._FromCSV(fileName, options)
            ns.FromCSV = MakeCSVDataFrame

            # Make a copy of the arrays that have strides to make sure we read the correct values
            # TODO a cleaner fix
            def MakeNumpyDataFrameCopy(np_dict):
                import numpy

                for key in np_dict.keys():
                    if (np_dict[key].__array_interface__["strides"]) is not None:
                        np_dict[key] = numpy.copy(np_dict[key])
                return _MakeNumpyDataFrame(np_dict)

            ns.FromNumpy = MakeNumpyDataFrameCopy

            # make a RDataFrame from a Pandas dataframe
            def MakePandasDataFrame(df):
                np_dict = {}
                for key in df.columns:
                    np_dict[key] = df[key].to_numpy()
                return _MakeNumpyDataFrame(np_dict)

            ns.FromPandas = MakePandasDataFrame

            try:
                # Inject Experimental.Distributed package into namespace RDF if available
                ns.Experimental.Distributed = _create_rdf_experimental_distributed_module(ns.Experimental)
            except ImportError:
                pass
        except:
            raise Exception("Failed to pythonize the namespace RDF")
        del type(self).RDF
        return ns

    # Overload RooFit namespace
    @property
    def RooFit(self):
        from ._pythonization._roofit import pythonize_roofit_namespace

        ns = self._fallback_getattr("RooFit")
        try:
            pythonize_roofit_namespace(ns)
        except:
            raise Exception("Failed to pythonize the namespace RooFit")
        del type(self).RooFit
        return ns

    # Overload TMVA namespace
    @property
    def TMVA(self):
        # this line is needed to import the pythonizations in _tmva directory
        from ._pythonization import _tmva

        ns = self._fallback_getattr("TMVA")
        hasRDF = "dataframe" in self.gROOT.GetConfigFeatures()
        if hasRDF:
            try:
                from ._pythonization._tmva import inject_rbatchgenerator, _AsRTensor, SaveXGBoost

                inject_rbatchgenerator(ns)
                ns.Experimental.AsRTensor = _AsRTensor
                ns.Experimental.SaveXGBoost = SaveXGBoost
            except:
                raise Exception("Failed to pythonize the namespace TMVA")
        del type(self).TMVA
        return ns

    # Create and overload Numba namespace
    @property
    def Numba(self):
        cppyy.cppdef("namespace Numba {}")
        ns = self._fallback_getattr("Numba")
        ns.Declare = staticmethod(_NumbaDeclareDecorator)
        del type(self).Numba
        return ns

    @property
    def NumbaExt(self):
        import numba

        if not hasattr(numba, "version_info") or numba.version_info < (0, 54):
            raise Exception("NumbaExt requires Numba version 0.54 or higher")

        import cppyy.numba_ext

        # Return something as it is a property function
        return self

    # Get TPyDispatcher for programming GUI callbacks
    @property
    def TPyDispatcher(self):
        cppyy.include("ROOT/TPyDispatcher.h")
        tpd = cppyy.gbl.TPyDispatcher
        type(self).TPyDispatcher = tpd
        return tpd
