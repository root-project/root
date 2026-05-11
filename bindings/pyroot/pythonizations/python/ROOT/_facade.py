import importlib
import os
import sys
import types
from functools import partial


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

    @property
    def _gROOT(self):
        gROOT = self.__dict__.get("_gROOT")
        if gROOT is None:
            gROOT = self._facade._cppyy.gbl.ROOT.GetROOT()
            self.__dict__["_gROOT"] = gROOT
        return gROOT

    def __getattr__(self, name):
        if name != "SetBatch" and self._facade.__dict__["gROOT"] is not self._gROOT:
            self._facade._finalSetup()
        return getattr(self._gROOT, name)

    def __setattr__(self, name, value):
        return setattr(self._gROOT, name, value)


class LiveProxy:
    """Generic adapter that forwards all access to a dynamically-resolved object.

    Mimics the C++ gPad/gDirectory/gVirtualX macro semantics: every attribute
    access re-evaluates the resolver to get the current underlying object.
    """

    __slots__ = ("_resolve",)

    def __init__(self, resolver):
        # resolver: zero-arg callable returning the current underlying object
        object.__setattr__(self, "_resolve", resolver)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __setattr__(self, name, value):
        setattr(self._resolve(), name, value)

    def __bool__(self):
        import ROOT

        return self._resolve() != ROOT.nullptr

    def __repr__(self):
        return repr(self._resolve())

    def __str__(self):
        return str(self._resolve())

    def __cast_cpp__(self):
        return self._resolve()


class TDirectoryPythonAdapter(LiveProxy):
    """LiveProxy specialization for `gDirectory`.

    Unlike `gPad` and `gVirtualX`, which on the C++ side are simple macros
    around a static function returning a pointer, `gDirectory` is a macro that
    constructs a `ROOT::Internal::TDirectoryAtomicAdapter`, which is itself an
    adapter that lazily resolves to the current `TDirectory *` (see
    TDirectory.h). That C++ adapter also defines custom equality against
    `TDirectory *`, which we mirror here via `IsEqual` so that comparisons with
    concrete directory pointers behave as users expect.

    Note: the C++ adapter also overloads `operator=`, but Python has no
    equivalent for non-attribute assignment, so that aspect is not reproduced.
    """

    def __init__(self):
        def resolver():
            import ROOT

            return ROOT.TDirectory.CurrentDirectory().load()

        super().__init__(resolver)

    def __eq__(self, other):
        import ROOT

        if other is self:
            return True
        cd = self._resolve()
        if cd == ROOT.nullptr:
            return other == ROOT.nullptr
        return cd.IsEqual(other)

    def __ne__(self, other):
        import ROOT

        if other is self:
            return False
        cd = self._resolve()
        if cd == ROOT.nullptr:
            return other != ROOT.nullptr
        return not cd.IsEqual(other)


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

        self.__all__ = module.__all__
        self.__name__ = module.__name__
        self.__file__ = module.__file__
        self.__spec__ = module.__spec__
        self.__path__ = module.__path__
        self.__doc__ = module.__doc__
        self.__package__ = module.__package__
        self.__loader__ = module.__loader__

        # Inject gROOT global
        self.gROOT = _gROOTWrapper(self)

        # Manually inject the three special global functors. On the C++ side
        # these are preprocessor macros (gDirectory expands to a
        # TDirectoryAtomicAdapter, gPad and gVirtualX expand to static accessor
        # calls). We reproduce their semantics here using LiveProxy.
        self.gDirectory = TDirectoryPythonAdapter()

        def _gclient_resolver():
            import ROOT

            return ROOT.TGClient.Instance()

        def _gpad_resolver():
            import ROOT

            return ROOT.TVirtualPad.Pad()

        def _gvirtualx_resolver():
            import ROOT

            return ROOT.TVirtualX.Instance()

        self.gClient = LiveProxy(_gclient_resolver)
        self.gPad = LiveProxy(_gpad_resolver)
        self.gVirtualX = LiveProxy(_gvirtualx_resolver)

        # Initialize configuration
        self.PyConfig = PyROOTConfiguration()

        self._is_ipython = is_ipython

        # Redirect lookups to temporary helper methods
        # This lets the user do some actions before all the machinery is in place:
        # - Set batch mode in gROOT
        # - Set options in PyConfig
        self.__class__.__getattr__ = self._getattr
        self.__class__.__setattr__ = self._setattr

    def SetHeuristicMemoryPolicy(self, enabled):
        import textwrap
        import warnings

        msg = """ROOT.SetHeuristicMemoryPolicy() is deprecated and will be removed in ROOT 6.44.
        Since ROOT 6.40, the heuristic memory policy is disabled by default, and with
        ROOT 6.44 it won't be possible to re-enable it with ROOT.SetHeuristicMemoryPolicy(),
        which was only meant to be used during a transition period and will be removed.
        """
        warnings.warn(textwrap.dedent(msg), FutureWarning, stacklevel=0)
        return self._cppyy._backend.SetHeuristicMemoryPolicy(enabled)

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
        return self._cppyy.ll.cast[out_type](addr)

    def _fallback_getattr(self, name):
        # Try:
        # - in the global namespace
        # - in the ROOT namespace
        # - in gROOT (ROOT lists such as list of files,
        #   memory mapped files, functions, geometries ecc.)
        # The first two attempts allow to lookup
        # e.g. ROOT.ROOT.Math as ROOT.Math

        # Note that hasattr caches the lookup for getattr
        if hasattr(self._cppyy.gbl, name):
            return getattr(self._cppyy.gbl, name)
        elif hasattr(self._cppyy.gbl.ROOT, name):
            return getattr(self._cppyy.gbl.ROOT, name)
        else:
            res = self.gROOT.FindObject(name)
            if res:
                return res
        raise AttributeError("Failed to get attribute {} from ROOT".format(name))

    def _register_converters_and_executors(self):
        converter_aliases = {
            "Float16_t": "float",
            "const Float16_t&": "const float&",
            "Double32_t": "double",
            "Double32_t&": "double&",
            "const Double32_t&": "const double&",
        }

        executor_aliases = {
            "Float16_t": "float",
            "Float16_t&": "float&",
            "Double32_t": "double",
            "Double32_t&": "double&",
        }

        from ROOT.libROOTPythonizations import CPyCppyyRegisterConverterAlias, CPyCppyyRegisterExecutorAlias

        for name, target in converter_aliases.items():
            CPyCppyyRegisterConverterAlias(name, target)

        for name, target in executor_aliases.items():
            CPyCppyyRegisterExecutorAlias(name, target)

    def _finalSetup(self):
        """
        Perform the final ROOT initialization.

        This method is intentionally deferred and is the *only* place where
        cppyy is imported and the C++ runtime is initialized. Delaying this
        step avoids importing the heavy-weight cppyy machinery unless it is
        actually required (for example, when accessing C++ ROOT symbols),
        allowing Python-only ROOT submodules to be imported with minimal
        overhead.
        """
        import cppyy
        import cppyy.ll
        import cppyy.types

        from ._application import PyROOTApplication
        from ._pythonization import _register_pythonizations, pythonization

        # signal policy: don't abort interpreter in interactive mode
        cppyy._backend.SetGlobalSignalPolicy(not cppyy.gbl.ROOT.GetROOT().IsBatch())

        self.__dict__["_cppyy"] = cppyy

        # Expose some functionality from CPyCppyy extension module
        cppyy_exports = [
            "nullptr",
            "bind_object",
            "as_cobject",
            "addressof",
            "SetImplicitSmartPointerConversion",
            "SetOwnership",
        ]
        for name in cppyy_exports:
            self.__dict__[name] = getattr(cppyy._backend, name)
        # For backwards compatibility
        self.__dict__["MakeNullPointer"] = partial(cppyy._backend.bind_object, 0)
        self.__dict__["BindObject"] = cppyy._backend.bind_object
        self.__dict__["AsCObject"] = cppyy._backend.as_cobject

        # Trigger the addition of the pythonizations
        _register_pythonizations()

        # Prevent this method from being re-entered through the gROOT wrapper
        self.__dict__["gROOT"] = self._cppyy.gbl.ROOT.GetROOT()

        # Make sure the interpreter is initialized once gROOT has been initialized
        self._cppyy.gbl.TInterpreter.Instance()

        # Setup interactive usage from Python
        self.__dict__["app"] = PyROOTApplication(self.PyConfig, self._is_ipython)
        if not self.gROOT.IsBatch() and self.PyConfig.StartGUIThread:
            self.app.init_graphics(self._cppyy.gbl.gEnv, self._cppyy.gbl.gSystem)

        # The automatic conversion of ordinary objects to smart pointers is
        # disabled for ROOT because it can cause trouble with overload
        # resolution. If a function has overloads for both ordinary objects and
        # smart pointers, then the implicit conversion to smart pointers can
        # result in the smart pointer overload being hit, even though there
        # would be an overload for the regular object. Since PyROOT didn't have
        # this feature before 6.32 anyway, disabling it was the safest option.
        self.SetImplicitSmartPointerConversion(False)

        # Redirect lookups to cppyy's global namespace
        self.__class__.__getattr__ = self._fallback_getattr
        self.__class__.__setattr__ = lambda self, name, val: setattr(self._cppyy.gbl, name, val)

        # Register custom converters and executors
        self._register_converters_and_executors()

        # Run rootlogon if exists
        self._run_rootlogon()

        # @pythonization decorator
        self.pythonization = pythonization

    def _getattr(self, name):
        # Special case, to allow "from ROOT import gROOT" w/o starting the graphics
        if name == "__path__":
            raise AttributeError(name)

        self._finalSetup()

        return getattr(self, name)

    def _setattr(self, name, val):
        # Setting attributes of ROOT will also define variables in the C++
        # runtime, so we generally require _finalSetup() in this method.
        #
        # Don't setup on submodule imports like `import ROOT._distrdf`,
        # implying a setattr(ROOT, "_distdf", <Module instance>) inside Pythons
        # importlib.
        if isinstance(val, types.ModuleType):
            self.__dict__[name] = val
            return

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
        if hasargv and "-n" not in sys.argv and not self.PyConfig.DisableRootLogon:
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
        from ._pythonization._rvec import _AsRVec

        ns.AsRVec = _AsRVec
        del type(self).VecOps
        return ns

    # Overload RDF namespace
    @property
    def RDF(self):
        self._finalSetup()
        ns = self._fallback_getattr("RDF")

        def MakeCSVDataFrame(fileName, readHeaders=True, delimiter=",", linesChunkSize=-1, colTypes={}, **kwargs):
            options = ns.RCsvDS.ROptions()
            options.fHeaders = readHeaders
            options.fDelimiter = delimiter
            options.fLinesChunkSize = linesChunkSize
            options.fColumnTypes = colTypes
            for key, val in kwargs.items():
                structMemberName = "f" + key[0].upper() + key[1:]
                if hasattr(options, structMemberName):
                    setattr(options, structMemberName, val)
            return ns._FromCSV(fileName, options)

        if hasattr(ns, "FromCSV"):
            # Provide a FromCSV factory method that uses keyword arguments instead of the ROptions config struct.
            # In Python, the RCsvDS::ROptions struct members are available without the leading 'f' and in camelCase,
            # e.g. fDelimiter --> delimiter.
            # We need to keep the parameters of the old FromCSV signature for backward compatibility.
            ns._FromCSV = ns.FromCSV
            ns.FromCSV = MakeCSVDataFrame

        # Make a copy of the arrays that have strides to make sure we read the correct values
        # TODO a cleaner fix
        def MakeNumpyDataFrameCopy(np_dict):
            import numpy

            from ._pythonization._rdataframe import _MakeNumpyDataFrame

            for key in np_dict.keys():
                if (np_dict[key].__array_interface__["strides"]) is not None:
                    np_dict[key] = numpy.copy(np_dict[key])
            return _MakeNumpyDataFrame(np_dict)

        ns.FromNumpy = MakeNumpyDataFrameCopy

        # make a RDataFrame from a Pandas dataframe
        def MakePandasDataFrame(df):
            from ._pythonization._rdataframe import _MakeNumpyDataFrame

            np_dict = {}
            for key in df.columns:
                np_dict[key] = df[key].to_numpy()
            return _MakeNumpyDataFrame(np_dict)

        ns.FromPandas = MakePandasDataFrame

        try:
            # Inject Pythonizations to interact between local and distributed RDF package
            from ._pythonization._rdf_namespace import (
                _create_distributed_module,
                _fromspec,
                _rungraphs,
                _variationsfor,
            )

            ns.Distributed = _create_distributed_module(ns)
            # Inject the experimental package which shows a warning before usage
            ns.Experimental.Distributed = _create_distributed_module(ns, True)
            ns.RunGraphs = _rungraphs(ns.Distributed.RunGraphs, ns.RunGraphs)
            ns.Experimental.VariationsFor = _variationsfor(ns.Distributed.VariationsFor, ns.Experimental.VariationsFor)
            ns.Experimental.FromSpec = _fromspec(ns.Distributed.FromSpec, ns.Experimental.FromSpec)
        except ImportError:
            # _rdf_namespace submodule not available (expected for dataframe=OFF)
            pass

        del type(self).RDF
        return ns

    @property
    def RDataFrame(self):
        """
        Dispatch between the local and distributed RDataFrame depending on
        input arguments.
        """
        local_rdf = self.__getattr__("RDataFrame")
        try:
            import ROOT._distrdf

            from ._pythonization._rdf_namespace import _rdataframe

            return _rdataframe(local_rdf, ROOT._distrdf.RDataFrame)
        except ImportError:
            # _distrdf submodule not available (expected for dataframe=OFF)
            return local_rdf

    # Overload RooFit namespace
    @property
    def RooFit(self):
        ns = self._fallback_getattr("RooFit")
        try:
            from ._pythonization._roofit import pythonize_roofit_namespace
        except ImportError:
            # _roofit submodule not available (expected for roofit=OFF)
            del type(self).RooFit
            return ns

        pythonize_roofit_namespace(ns)

        del type(self).RooFit
        return ns

    # Overload TMVA namespace
    @property
    def TMVA(self):
        ns = self._fallback_getattr("TMVA")
        try:
            # This line is needed to import the pythonizations in _tmva directory.
            # The comment suppresses linter errors about unused imports.
            from ._pythonization import _tmva  # noqa: F401
            from ._pythonization._tmva._rtensor import _AsRTensor
            from ._pythonization._tmva._sofie._parser._keras.parser import PyKeras
            from ._pythonization._tmva._tree_inference import SaveXGBoost

            setattr(ns.Experimental.SOFIE, "PyKeras", PyKeras)

            ns.Experimental.AsRTensor = _AsRTensor
            ns.Experimental.SaveXGBoost = SaveXGBoost
        except ImportError:
            # _tmva submodule not available (expected for tmva=OFF)
            pass
        del type(self).TMVA
        return ns

    # Create and overload Numba namespace
    @property
    def Numba(self):
        from ._numbadeclare import _NumbaDeclareDecorator

        self._cppyy.cppdef("namespace Numba {}")
        ns = self._fallback_getattr("Numba")
        ns.Declare = staticmethod(_NumbaDeclareDecorator)
        del type(self).Numba
        return ns

    @property
    def NumbaExt(self):
        import numba

        if not hasattr(numba, "version_info") or numba.version_info < (0, 54):
            raise Exception("NumbaExt requires Numba version 0.54 or higher")

        # The comment in the next line suppresses linter errors about unused imports
        import cppyy.numba_ext  # noqa: F401

        # Return something as it is a property function
        return self

    # Get TPyDispatcher for programming GUI callbacks
    @property
    def TPyDispatcher(self):
        self._cppyy.include("ROOT/TPyDispatcher.h")
        tpd = self._cppyy.gbl.TPyDispatcher
        type(self).TPyDispatcher = tpd
        return tpd

    # Create the uhi namespace
    @property
    def uhi(self):
        uhi_module = types.ModuleType("uhi")
        uhi_module.__file__ = "<module ROOT>"
        uhi_module.__package__ = self
        from ._pythonization._uhi import _add_module_level_uhi_helpers

        _add_module_level_uhi_helpers(uhi_module)
        return uhi_module

    @property
    def Experimental(self):
        ns = self._fallback_getattr("Experimental")

        from ._pythonization._ml_dataloader import _inject_dataloader_api

        _inject_dataloader_api(ns.ML)

        return ns
