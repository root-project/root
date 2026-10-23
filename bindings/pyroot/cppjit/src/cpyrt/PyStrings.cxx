// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "PyStrings.h"

//- data _____________________________________________________________________
#define CPYRT_DEFINE_PYSTRING(var, str)                                        \
  PyObject* cpyrt::PyStrings::var = nullptr;
CPYRT_PYSTRINGS(CPYRT_DEFINE_PYSTRING)
#undef CPYRT_DEFINE_PYSTRING

//-----------------------------------------------------------------------------
bool cpyrt::CreatePyStrings() {
  // Build cache of commonly used python strings (the cache is python intern, so
  // all strings are shared python-wide, not just in cppjit).
#define CPYRT_CREATE_PYSTRING(var, str)                                        \
  if (!(PyStrings::var = PyUnicode_InternFromString(str)))                     \
    return false;
  CPYRT_PYSTRINGS(CPYRT_CREATE_PYSTRING)
#undef CPYRT_CREATE_PYSTRING

  return true;
}

//-----------------------------------------------------------------------------
PyObject* cpyrt::DestroyPyStrings() {
  // Remove all cached python strings.
#define CPYRT_DESTROY_PYSTRING(var, str) Py_CLEAR(PyStrings::var);
  CPYRT_PYSTRINGS(CPYRT_DESTROY_PYSTRING)
#undef CPYRT_DESTROY_PYSTRING

  Py_RETURN_NONE;
}
