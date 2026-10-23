# Author: Giacomo Parolini CERN 04/2025

r"""
\pythondoc ROOT::Experimental::RFile

`ROOT::Experimental::RFile` is a modern, minimalistic interface to read and write ROOT files.
PyROOT enhances `RFile` with pythonic interfaces:

- **Factory methods & construction**: `RFile` cannot be instantiated directly via `RFile()`.
  Use one of the factory methods:
  - `ROOT.Experimental.RFile.Open(filename)` to open an existing file for reading.
  - `ROOT.Experimental.RFile.Recreate(filename)` to create a new file (or overwrite an existing one).
  - `ROOT.Experimental.RFile.Update(filename)` to open an existing file for reading and writing.

- **Context Manager**: `RFile` instances can be used in a Python `with` statement. The file
  will be automatically flushed and closed when exiting the context:
  \code{.py}
  import ROOT

  with ROOT.Experimental.RFile.Recreate("example.root") as f:
      hist = ROOT.TH1D("h", "Example", 10, 0, 10)
      hist.FillRandom("gaus", 1000)
      f.Put("hist", hist)
  # The file is flushed and closed here
  \endcode

- **Writing Objects (`Put`)**:
  - `file.Put(name, obj)` automatically detects the C++ type of `obj` via its `__cpp_name__`.
    Python `str` instances are automatically converted to `std::string`.
  - `file.Put[type](name, obj)` allows explicit template specialization when needed (e.g. for
    `std::vector` or custom types):
    \code{.py}
    f.Put["std::vector<int>"]("my_vec", [1, 2, 3])
    \endcode

- **Reading Objects (`Get`)**:
  - `file.Get(name)` reads the object with its concrete type automatically inferred from
    the key metadata. Returns `None` if no object exists with that name.
  - `file.Get[type](name)` reads the object explicitly cast to the specified type.
  - Objects retrieved via `Get()` are independently owned by the caller and survive after
    the file is closed:
    \code{.py}
    with ROOT.Experimental.RFile.Open("example.root") as f:
        hist = f.Get("hist")

    # hist remains valid outside the context
    print(hist.GetEntries())
    \endcode

- **Inspecting Keys (`GetKeyInfo`)**:
  - `file.GetKeyInfo(path)` returns the `RKeyInfo` metadata object (containing `GetPath()`,
    `GetClassName()`, `GetCycle()`) or `None` if the path does not exist.

- **Listing Keys (`ListKeys`)**:
  - `file.ListKeys(basePath="", **kwargs)` returns an iterable of `RKeyInfo` objects:
    - `listObjects`: Include leaf objects (default `True`).
    - `listDirs`: Include directory keys (default `False`).
    - `listRecursive`: Search subdirectories recursively (default `True`).
  \code{.py}
  with ROOT.Experimental.RFile.Open("example.root") as f:
      for key in f.ListKeys():
          print(f"{key.GetPath()} ({key.GetClassName()})")
  \endcode

\endpythondoc
"""

from . import pythonization


class _RFile_Get:
    r"""
    \ingroup Py_RFile
    \brief Wrapper allowing access to objects in an RFile via the `Get()` method.

    Supports both non-templated calls `rfile.Get("name")` and templated calls
    `rfile.Get[CppType]("name")`.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, namecycle):
        r"""
        \ingroup Py_RFile
        \brief Retrieve an object from the RFile by name or namecycle.

        The concrete C++ type is automatically deduced from key metadata.

        \param namecycle Path/namecycle of the object inside the RFile.
        \return The deserialized ROOT object, or None if the key does not exist.
        """
        import ROOT

        key = self._rfile.GetKeyInfo(namecycle)
        if key:
            obj = ROOT.Experimental.Internal.RFile_GetObjectFromKey(self._rfile, key)
            return ROOT._cppyy.bind_object(obj, key.GetClassName())
        # No key
        return None

    def __getitem__(self, template_arg):
        r"""
        \ingroup Py_RFile
        \brief Access templated `Get` overload with explicit C++ type specialization.

        \param template_arg C++ class or type name to cast the retrieved object to.
        \return Callable wrapper accepting the object path.
        """

        def getitem_wrapper(namecycle):
            obj = self._rfile._OriginalGet[template_arg](namecycle)
            return obj if obj else None

        return getitem_wrapper


class _RFile_Put:
    r"""
    \ingroup Py_RFile
    \brief Wrapper allowing writing objects to an RFile via the `Put()` method.

    Supports both non-templated calls `rfile.Put("name", obj)` and templated calls
    `rfile.Put[CppType]("name", obj)`.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, name, obj):
        r"""
        \ingroup Py_RFile
        \brief Write an object into the RFile under the given path.

        Automatically determines the C++ class name from `obj.__cpp_name__`.
        Python `str` instances are automatically converted to `std::string`.

        \param name Path in the RFile where the object should be stored.
        \param obj The object to write.
        \exception TypeError If the type of `obj` is not supported by ROOT I/O.
        """
        objType = type(obj)
        if isinstance(obj, str):
            # special case: automatically convert python str to std::string
            className = "std::string"
        elif not hasattr(objType, "__cpp_name__"):
            raise TypeError(f"type {objType} is not supported by ROOT I/O")
        else:
            className = objType.__cpp_name__
        self._rfile.Put[className](name, obj)

    def __getitem__(self, template_arg):
        r"""
        \ingroup Py_RFile
        \brief Access templated `Put` overload with explicit C++ type specialization.

        \param template_arg Explicit C++ type name string or type object.
        \return The original templated Put method bound with `template_arg`.
        """
        return self._rfile._OriginalPut[template_arg]


def _RFileExit(obj, exc_type, exc_val, exc_tb):
    r"""
    \ingroup Py_RFile
    \brief Context manager exit callback that closes the RFile.

    Signature and return value conform to Python context manager specification
    (see https://docs.python.org/3/library/stdtypes.html#typecontextmanager).

    \param obj The RFile instance being closed.
    \param exc_type Exception type if an exception occurred in the with-block, else None.
    \param exc_val Exception value, else None.
    \param exc_tb Traceback, else None.
    \return False to propagate any raised exceptions.
    """
    obj.Close()
    return False


def _RFileOpen(original):
    r"""
    \ingroup Py_RFile
    \brief Decorator for factory methods (Recreate, Open, Update) to inject pythonized methods.

    \param original The original C++ factory method.
    \return A wrapper that installs pythonized `Get` and `Put` on the returned instance.
    """

    def rfile_open_wrapper(klass, *args):
        rfile = original(*args)
        rfile._OriginalGet = rfile.Get
        rfile.Get = _RFile_Get(rfile)
        rfile._OriginalPut = rfile.Put
        rfile.Put = _RFile_Put(rfile)
        return rfile

    return rfile_open_wrapper


def _RFileInit(rfile):
    r"""
    \ingroup Py_RFile
    \brief Prevent direct instantiation of RFile via constructor.

    \param rfile The uninitialized RFile instance.
    \exception NotImplementedError Direct construction is disallowed; use Open, Recreate, or Update.
    """
    raise NotImplementedError("RFile can only be created via Recreate, Open or Update")


def _GetKeyInfo(rfile, path):
    r"""
    \ingroup Py_RFile
    \brief Retrieve metadata about a key at the specified path.

    \param rfile The RFile instance.
    \param path Path of the object inside the RFile.
    \return An RKeyInfo instance if the key exists, otherwise None.
    """
    key = rfile._OriginalGetKeyInfo(path)
    if key.has_value():
        return key.value()
    return None


def _ListKeys(rfile, basePath="", **kwargs):
    r"""
    \ingroup Py_RFile
    \brief Return an iterable over keys of objects and/or directories in the RFile.

    \param rfile The RFile instance.
    \param basePath Path in the file to start listing from (default: root of file).
    \param kwargs Keyword arguments:
                  - `listObjects` (bool): Include keys of non-directory objects (default: True).
                  - `listDirs` (bool): Include keys of directory objects (default: False).
                  - `listRecursive` (bool): Recurse into subdirectories (default: True).
    \return An iterable yielding `RKeyInfo` instances.

    Example:
    \code{.py}
    # Iterate over all objects in the RFile
    for key in file.ListKeys():
        print(f"{key.GetPath()};{key.GetCycle()} of type {key.GetClassName()}")

    # Iterate over all objects and directories
    for key in file.ListKeys("", listDirs=True):
        print(f"{key.GetPath()} ({key.GetClassName()})")

    # Iterate non-recursively inside a subdirectory
    for key in file.ListKeys("a/b", listRecursive=False):
        print(key.GetPath())
    \endcode
    """
    from ROOT.Experimental import RFile

    listObjects = kwargs.get('listObjects', True)
    listDirs = kwargs.get('listDirs', False)
    listRecursive = kwargs.get('listRecursive', True)

    flags = (listObjects * RFile.kListObjects) | (listDirs * RFile.kListDirs) | (listRecursive * RFile.kListRecursive)
    iter = rfile._OriginalListKeys(basePath, flags)
    return iter


@pythonization("RFile", ns="ROOT::Experimental")
def pythonize_rfile(klass):
    r"""
    \ingroup Py_RFile
    \brief Register Pythonization behaviors for ROOT::Experimental::RFile.

    \param klass The ROOT::Experimental::RFile class proxy.
    """
    # Explicitly prevent to create a RFile via ctor
    klass.__init__ = _RFileInit

    # Pythonize factory methods
    klass.Open = classmethod(_RFileOpen(klass.Open))
    klass.Update = classmethod(_RFileOpen(klass.Update))
    klass.Recreate = classmethod(_RFileOpen(klass.Recreate))

    # Pythonization for __enter__ and __exit__ methods
    # These make RFile usable in a `with` statement as a context manager
    klass.__enter__ = lambda rfile: rfile
    klass.__exit__ = _RFileExit
    klass._OriginalGetKeyInfo = klass.GetKeyInfo
    klass.GetKeyInfo = _GetKeyInfo
    klass._OriginalListKeys = klass.ListKeys
    klass.ListKeys = _ListKeys
