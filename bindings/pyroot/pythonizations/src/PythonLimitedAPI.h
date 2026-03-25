#ifndef ROOT_PythonLimitedAPI_h
#define ROOT_PythonLimitedAPI_h

// Use what is in the limited API since Python 3.11 if we're building with at
// least Python 3.11. The reason why we can't go back to 3.10 is that that at
// that point, the new buffer interface was not part of the limited API yet.
#if PY_VERSION_HEX >= 0x030B0000

// On Windows we can't use the stable ABI yet: it requires linking against a
// different libpython, so as long as we don't build all translation units in
// the ROOT Pythonization library with the stable ABI we should not use it.
#ifndef _WIN32
#define Py_LIMITED_API 0x030B0000
#endif

#endif

#include <Python.h>

#endif
