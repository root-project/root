from __future__ import print_function
import os, py, sys, subprocess
from contextlib import contextmanager

currpath = py.path.local(__file__).dirpath()


def setup_make(targetname):
    if sys.platform == 'win32':
        popen = subprocess.Popen([sys.executable, "make_dict_win32.py", targetname], cwd=str(currpath),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        popen = subprocess.Popen(["make", targetname+"Dict.so"], cwd=str(currpath),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = popen.communicate()
    if popen.returncode:
        raise OSError("'make' failed:\n%s" % (stdout,))

if sys.hexversion >= 0x3000000:
    pylong = int
    pyunicode = str
    maxvalue = sys.maxsize
else:
    pylong = long
    pyunicode = unicode
    maxvalue = sys.maxint

IS_WINDOWS = 0
if 'win32' in sys.platform:
    import platform
    if '64' in platform.architecture()[0]:
        IS_WINDOWS = 64
        maxvalue = 2**31-1
    else:
        IS_WINDOWS = 32

IS_MAC_ARM = 0
IS_MAC_X86 = 0
if 'darwin' in sys.platform:
    import platform
    if 'arm64' in platform.machine():
        IS_MAC_ARM = 64
        os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"
    else:
        IS_MAC_X86 = 1

IS_MAC = IS_MAC_ARM or IS_MAC_X86
IS_LINUX = not (IS_WINDOWS or IS_MAC) 

def _register_root_error_counter():

    import cppyy

    # already registered
    if hasattr(cppyy.gbl, "rootErrorCount"):
        return

    cppyy.cppdef("""
    auto originalErrorHandler = ::GetErrorHandler();

    int &rootErrorCount()
    {
        static int count = 0;
        return count;
    }

    void handleErrorWithException(int severity,
                              bool abort,
                              const char * location,
                              const char * msg)
    {
        originalErrorHandler(severity, abort, location, msg);
        rootErrorCount()++;
    }

    ::SetErrorHandler(handleErrorWithException);
    """)

@contextmanager
def no_root_errors():
    """Context manager to ensure no new ROOT errors occur."""
    import cppyy

    _register_root_error_counter()

    start_count = cppyy.gbl.rootErrorCount()
    yield
    end_count = cppyy.gbl.rootErrorCount()
    if end_count != start_count:
        raise AssertionError(f"ROOT emitted {end_count - start_count} error(s) during block!")


try:
    import __pypy__
    ispypy = True
except ImportError:
    ispypy = False
