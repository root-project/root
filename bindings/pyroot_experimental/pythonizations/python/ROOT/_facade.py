import types
import sys
import os
from functools import partial

import libcppyy as cppyy_backend
from cppyy import gbl as gbl_namespace
from cppyy import cppdef
from libROOTPythonizations import gROOT, CreateBufferFromAddress

from ._application import PyROOTApplication


def _NumbaDeclareDecorator(input_types, return_type, name=None):
    # Check for cfunc in numba
    try:
        from numba import cfunc
    except:
        raise Exception('Failed to import cfunc from numba')

    # Check input and return types
    typemap = {
            'float': 'float32',
            'double': 'float64',
            'int': 'int32',
            'unsigned int': 'uint32',
            'long': 'int64',
            'unsigned long': 'uint64',
            'bool': 'boolean'
            }

    for t in input_types:
        if not t in typemap:
            raise Exception('Input type {} is not supported for jitting with numba. Valid types are {}'.format(
                t, list(typemap.keys())))

    if not return_type in typemap:
        raise Exception('Return type {} is not supported for jitting with numba. Valid types are {}'.format(
            return_type, list(typemap.keys())))

    # Define inner decorator without arguments
    def inner(func, input_types=input_types, return_type=return_type, name=name):
        # Build signature for numba
        # We checked above that all types are in the typemap
        numba_signature = "{RETURN_TYPE}({INPUT_TYPES})".format(
                RETURN_TYPE=typemap[return_type],
                INPUT_TYPES=','.join([typemap[t] for t in input_types]))

        # Compile the Python callable with numba
        try:
            cppfunc = cfunc(numba_signature, nopython=True)(func)
        except:
            raise Exception('Failed to jit Python callable {PYCALLABLE} with numba.cfunc("{SIGNATURE}", nopython=True)'.format(
                PYCALLABLE=func, SIGNATURE=numba_signature))
        func.__numba_cfunc__ = cppfunc

        # Get address of jitted function
        address = cppfunc.address

        # Infer name of the C++ wrapper function
        if not name:
            name = func.__name__

        # Build C++ wrapper for jitting with cling
        code = """\
namespace Numba {{
{RETURN_TYPE} {FUNC_NAME}({INPUT_SIGNATURE}) {{
    auto funcptr = reinterpret_cast<{RETURN_TYPE}(*)({INPUT_TYPES})>({FUNC_PTR});
    return funcptr({INPUT_ARGS});
}}
}}""".format(
                RETURN_TYPE=return_type,
                FUNC_NAME=name,
                INPUT_SIGNATURE=', '.join(['{} x_{}'.format(t, i) for i, t in enumerate(input_types)]),
                INPUT_TYPES=', '.join(input_types),
                FUNC_PTR=address,
                INPUT_ARGS=', '.join(['x_{}'.format(i) for i in range(len(input_types))]))

        # Jit wrapper code
        err = gbl_namespace.gInterpreter.Declare(code)
        if not err:
            raise Exception('Failed to jit wrapper code with cling:\n{}'.format(code))
        func.__cpp_wrapper__ = code

        return func

    return inner


class PyROOTConfiguration(object):
    """Class for configuring PyROOT"""

    def __init__(self):
        self.IgnoreCommandLineOptions = False
        self.ShutDown = True
        self.DisableRootLogon = False


class _gROOTWrapper(object):
    """Internal class to manage lookups of gROOT in the facade.
       This wrapper calls _finalSetup on the facade when it
       receives a lookup, unless the lookup is for SetBatch.
       This allows to evaluate the command line parameters
       before checking if batch mode is on in _finalSetup
    """

    def __init__(self, facade):
        self.__dict__['_facade'] = facade
        self.__dict__['_gROOT'] = gROOT

    def __getattr__( self, name ):
        if name != 'SetBatch' and self._facade.__dict__['gROOT'] != self._gROOT:
            self._facade._finalSetup()
        return getattr(self._gROOT, name)

    def __setattr__(self, name, value):
        return setattr(self._gROOT, name, value)


class ROOTFacade(types.ModuleType):
    """Facade class for ROOT module"""

    def __init__(self, module, is_ipython):
        types.ModuleType.__init__(self, module.__name__)

        self.module = module
        # Importing all will be customised later
        self.module.__all__ = []

        self.__doc__  = module.__doc__
        self.__name__ = module.__name__
        self.__file__ = module.__file__

        # Inject gROOT global
        self.gROOT = _gROOTWrapper(self)

        # Expose some functionality from CPyCppyy extension module
        self._cppyy_exports = [ 'nullptr', 'bind_object', 'as_cobject', 'addressof',
                                'SetMemoryPolicy', 'kMemoryHeuristics', 'kMemoryStrict',
                                'SetOwnership' ]
        for name in self._cppyy_exports:
            setattr(self, name, getattr(cppyy_backend, name))
        # For backwards compatibility
        self.MakeNullPointer = partial(self.bind_object, 0)
        self.BindObject = self.bind_object
        self.AsCObject = self.as_cobject

        # Initialize configuration
        self.PyConfig = PyROOTConfiguration()

        self._is_ipython = is_ipython

        # Redirect lookups to temporary helper methods
        # This lets the user do some actions before all the machinery is in place:
        # - Set batch mode in gROOT
        # - Set options in PyConfig
        self.__class__.__getattr__ = self._getattr
        self.__class__.__setattr__ = self._setattr

        # Setup import hook
        self._set_import_hook()

    def AddressOf(self, obj):
        # Return an indexable buffer of length 1, whose only element
        # is the address of the object.
        # The address of the buffer is the same as the address of the
        # address of the object

        # addr is the address of the address of the object
        addr = self.addressof(instance = obj, byref = True)

        # Create a buffer (LowLevelView) from address
        return CreateBufferFromAddress(addr)

    def _set_import_hook(self):
        # This hook allows to write e.g:
        # from ROOT.A import a
        # instead of the longer:
        # from ROOT import A
        # from A import a
        try:
            import __builtin__
        except ImportError:
            import builtins as __builtin__  # name change in p3
        _orig_ihook = __builtin__.__import__
        def _importhook(name, *args, **kwds):
            if name[0:5] == 'ROOT.':
                try:
                    sys.modules[name] = getattr(self, name[5:])
                except Exception:
                    pass
            return _orig_ihook(name, *args, **kwds)
        __builtin__.__import__ = _importhook

    def _handle_import_all(self):
        # Called if "from ROOT import *" is executed in the app.
        # Customises lookup in Python's main module to also
        # check in C++'s global namespace

        # Get caller module (jump over the facade frames)
        num_frame = 2
        frame = sys._getframe(num_frame).f_globals['__name__']
        while frame == 'ROOT._facade':
            num_frame += 1
            frame = sys._getframe(num_frame).f_globals['__name__']
        caller = sys.modules[frame]

        # Install the hook
        cppyy_backend._set_cpp_lazy_lookup(caller.__dict__)

    def _fallback_getattr(self, name):
        # Try:
        # - in the global namespace
        # - in the ROOT namespace
        # - in gROOT (ROOT lists such as list of files,
        #   memory mapped files, functions, geometries ecc.)
        # The first two attempts allow to lookup
        # e.g. ROOT.ROOT.Math as ROOT.Math

        if name == '__all__':
            self._handle_import_all()
            # Make the attributes of the facade be injected in the
            # caller module
            raise AttributeError()
        # Note that hasattr caches the lookup for getattr
        elif hasattr(gbl_namespace, name):
            return getattr(gbl_namespace, name)
        elif hasattr(gbl_namespace.ROOT, name):
            return getattr(gbl_namespace.ROOT, name)
        else:
            res = gROOT.FindObject(name)
            if res:
                return res
        raise AttributeError("Failed to get attribute {} from ROOT".format(name))

    def _finalSetup(self):
        # Prevent this method from being re-entered through the gROOT wrapper
        self.__dict__['gROOT'] = gROOT

        # Setup interactive usage from Python
        self.__dict__['app'] = PyROOTApplication(self.PyConfig, self._is_ipython)
        if not self.gROOT.IsBatch():
            self.app.init_graphics()

        # Set memory policy to kUseHeuristics.
        # This restores the default in PyROOT which was changed
        # by new Cppyy
        self.SetMemoryPolicy(self.kMemoryHeuristics)

        # Redirect lookups to cppyy's global namespace
        self.__class__.__getattr__ = self._fallback_getattr
        self.__class__.__setattr__ = lambda self, name, val: setattr(gbl_namespace, name, val)

        # Run rootlogon if exists
        self._run_rootlogon()

    def _getattr(self, name):
        # Special case, to allow "from ROOT import gROOT" w/o starting the graphics
        if name == '__path__':
            raise AttributeError(name)

        self._finalSetup()

        return getattr(self, name)

    def _setattr(self, name, val):
        self._finalSetup()

        return setattr(self, name, val)

    def _run_rootlogon(self):
        # Run custom logon file (must be after creation of ROOT globals)
        hasargv = hasattr(sys, 'argv')
        # -n disables the reading of the logon file, just like with root
        if hasargv and not '-n' in sys.argv and not self.PyConfig.DisableRootLogon:
            file_path = os.path.expanduser('~/.rootlogon.py')
            if os.path.exists(file_path):
                # Could also have used execfile, but import is likely to give fewer surprises
                module_name = 'rootlogon'
                if sys.version_info >= (3,5):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    import imp
                    imp.load_module(module_name, open(file_path, 'r'), file_path, ('.py','r',1))
                    del imp
            else:
                # If the .py version of rootlogon exists, the .C is ignored (the user can
                # load the .C from the .py, if so desired).
                # System logon, user logon, and local logon (skip Rint.Logon)
                name = '.rootlogon.C'
                logons = [
                    os.path.join(str(self.TROOT.GetEtcDir()), 'system' + name),
                    os.path.expanduser(os.path.join('~', name))
                    ]
                if logons[-1] != os.path.join(os.getcwd(), name):
                    logons.append(name)
                for rootlogon in logons:
                    if os.path.exists(rootlogon):
                        self.TApplication.ExecuteFile(rootlogon)

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
        ns = self._fallback_getattr('VecOps')
        try:
            from libROOTPythonizations import AsRVec
            ns.AsRVec = AsRVec
        except:
            raise Exception('Failed to pythonize the namespace VecOps')
        del type(self).VecOps
        return ns

    # Overload RDF namespace
    @property
    def RDF(self):
        ns = self._fallback_getattr('RDF')
        try:
            from libROOTPythonizations import MakeNumpyDataFrame
            ns.MakeNumpyDataFrame = MakeNumpyDataFrame
        except:
            raise Exception('Failed to pythonize the namespace RDF')
        del type(self).RDF
        return ns

    # Overload TMVA namespace
    @property
    def TMVA(self):
        ns = self._fallback_getattr('TMVA')
        try:
            from libROOTPythonizations import AsRTensor
            ns.Experimental.AsRTensor = AsRTensor
        except:
            raise Exception('Failed to pythonize the namespace TMVA')
        del type(self).TMVA
        return ns

    # Create and overload Numba namespace
    @property
    def Numba(self):
        cppdef('namespace Numba {}')
        ns = self._fallback_getattr('Numba')
        ns.Declare = _NumbaDeclareDecorator
        del type(self).Numba
        return ns
