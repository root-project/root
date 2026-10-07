# Author: Giacomo Parolini CERN 04/2025

r"""
\pythondoc RFile

RFile is a modern, minimal interface to ROOT files with a focus on simplicity.
It allows to perform the basic actions of getting objects from a file, putting objects into it and query the file
for metadata.

Here is a typical use of RFile:

\code{.py}
# Reading an object from a file. Note that Get will return None if no object is stored under the given path.
with ROOT.Experimental.RFile.Open("myfile.root") as file:
    myHisto = file.Get("myHisto")
    # file will be closed upon exiting the with statement, but all objects retrieved from it remain valid.

print(f"{myHisto.GetName()} has {myHisto.GetNbins()} bins.")

# Writing an object to a new file
with ROOT.Experimental.RFile.Recreate("myfile2.root") as file:
    file.Put("myHisto", myHisto)
    # file will be written upon exiting the with statement, or explicitly with:
    # file.Flush()
\endcode

The Put method will raise an error if an object is already present at the given path; if you want to overwrite an
existing object use the Overwrite method instead:

\code{.py}
file.Overwrite("myHisto", myOtherHisto)
\endcode

You can iterate the metadata of all objects inside an RFile using the ListKeys method:
\code{.py}
for key in file.ListKeys():
    print(key.GetBaseName())
    # We can use the key to get information about the object without loading it from storage
    if key.GetClassName() == "TH1D":
        histo = file.Get(key.GetPath())
        histos.append(histo)
\endcode

ListKeys is recursive by default. If you want non-recursive behavior, you can specify it with a keyword argument:
\code{.py}
for key in file.ListKeys("", recursive=False):
    # this will only print the top-level objects.
    print(key.GetBaseName())
\endcode

See the documentation of ListKeys for more options.

## Directories and paths
Objects in an RFile are organized in a hierarchical structure represented by their "path". A path is the string you
pass to Get or Put (like in the examples above) and it uses '/' as the directory separator. A directory is useful to
group multiple related objects and it can be used alongside ListKeys:

\code{.py}
# lists all objects under the "myDir" directory. 
for key in file.ListKeys("myDir"):
    # will print something like "myDir/myObjName"
    print(key.GetPath())
\endcode

\endpythondoc
"""

from . import pythonization


class _RFile_Get:
    """
    Allow access to objects through the method Get().
    This is pythonized to allow Get() to be called both with and without a template argument.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, namecycle):
        """
        Non-templated Get()
        """
        import ROOT

        key = self._rfile.GetKeyInfo(namecycle)
        if key:
            obj = ROOT.Experimental.Internal.RFile_GetObjectFromKey(self._rfile, key)
            return ROOT._cppyy.bind_object(obj, key.GetClassName())
        # No key
        return None

    def __getitem__(self, template_arg):
        """
        Templated Get()
        """

        def getitem_wrapper(namecycle):
            obj = self._rfile._OriginalGet[template_arg](namecycle)
            return obj if obj else None

        return getitem_wrapper


class _RFile_Put:
    """
    Allow writing objects through the method Put().
    This is pythonized to allow Put() to be called both with and without a template argument.
    """

    def __init__(self, rfile):
        self._rfile = rfile

    def __call__(self, name, obj):
        """
        Non-templated Put()
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
        """
        Templated Put()
        """
        return self._rfile._OriginalPut[template_arg]


def _RFileExit(obj, exc_type, exc_val, exc_tb):
    """
    Close the RFile object.
    Signature and return value are imposed by Python, see
    https://docs.python.org/3/library/stdtypes.html#typecontextmanager.
    """
    obj.Close()
    return False


def _RFileOpen(original):
    """
    Pythonization for the factory methods (Recreate, Open, Update)
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
    """
    Prevent the creation of RFile through constructor (must use a factory method)
    """
    raise NotImplementedError("RFile can only be created via Recreate, Open or Update")


def _GetKeyInfo(rfile, path):
    key = rfile._OriginalGetKeyInfo(path)
    if key.has_value():
        return key.value()
    return None


def _ListKeys(rfile, basePath="", **kwargs):
    """
    Returns an iterable over all keys of objects and/or directories written into this RFile starting at path
    `basePath` (defaulting to include the content of all subdirectories).
    By default, keys referring to directories are not returned: only those referring to leaf objects are.
    If `basePath` is the path of a leaf object, only `basePath` itself will be returned.
    If it is the path of a directory, it will not be included in the listing.

    You can specify what to list via the keyword arguments:
    - if `listObjects == True`, the listing will include keys of non-directory objects (default);
    - if `listDirs == True`, the listing will include keys of directory objects;
    - if `listRecursive == True`, the listing will recurse on all subdirectories of `basePath` (default),
    otherwise it will only list immediate children of `basePath`.
        Example usage:
    ~~~{.py}
    for key in file.ListKeys():
        # iterate over all objects in the RFile
        print(f"{key.GetPath()};{key.GetCycle()} of type {key.GetClassName()}")

    for key in file.ListKeys("", listDirs=True):
        # iterate over all objects and directories in the RFile
        print(f"{key.GetPath()};{key.GetCycle()} of type {key.GetClassName()}")

    for key in file.ListKeys("a/b", listRecursive=False):
        # iterate over all objects that are immediate children of directory "a/b"
        print(f"{key.GetPath()};{key.GetCycle()} of type {key.GetClassName()}")

    for key in file.ListKeys("foo", listDirs=True, listObjects=False):
        # iterate over all directories under directory "foo", recursively
        print(key.GetPath())
    ~~~
    """
    from ROOT.Experimental import RFile

    listObjects = kwargs['listObjects'] if 'listObjects' in kwargs else True
    listDirs = kwargs['listDirs'] if 'listDirs' in kwargs else False
    listRecursive = kwargs['listRecursive'] if 'listRecursive' in kwargs else True

    flags = (listObjects * RFile.kListObjects) | (listDirs * RFile.kListDirs) | (listRecursive * RFile.kListRecursive)
    iter = rfile._OriginalListKeys(basePath, flags)
    return iter


@pythonization("RFile", ns="ROOT::Experimental")
def pythonize_rfile(klass):
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
