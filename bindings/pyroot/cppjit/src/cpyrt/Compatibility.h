#ifndef CPYRT_COMPATIBILITY_H
#define CPYRT_COMPATIBILITY_H

#include "Python.h"

#include "PyStrings.h"
#include "cppjit_interop.h"

#include <string>

namespace cppjit::cpyrt::compat {

inline PyObject* GetClingPrintValue() {
#ifdef CPPJIT_USE_CLING
  static PyObject* printValue = nullptr;
  if (printValue)
    return printValue;

  PyObject* gbl =
      PyDict_GetItemString(PySys_GetObject((char*)"modules"), "cppjit.gbl");
  PyObject* cling = gbl ? PyObject_GetAttrString(gbl, (char*)"cling") : nullptr;
  printValue =
      cling ? PyObject_GetAttrString(cling, (char*)"printValue") : nullptr;
  Py_XDECREF(cling);

  if (printValue) {
    Py_DECREF(printValue); // make borrowed
    if (!PyCallable_Check(printValue))
      printValue = nullptr;
  }

  if (!printValue)
    PyErr_Clear();

  return printValue;
#else
  return nullptr;
#endif
}

// Interpreter::toString is an assert(0) stub (CppInterOp#1100); skip it.
inline std::string ObjToString(interop::TCppScope_t /*klass*/,
                               interop::TCppObject_t /*obj*/) {
  return "";
}

//- cppyy legacy API ----------------------------------------------------------
// Backwards compatibility with the API that former cppyy users have in their
// code. The macro is deliberately not a build option: it only exists so that
// the legacy code paths are easy to find, and to remove once there is a
// transition plan. Keep all such shims in this section.
#define CPPYY_LEGACY_API 1

// Name under which pyclass defines its explicit pythonization hook, looked up
// in its own __dict__ only (no fall-through to base classes). Prefers the
// __cppjit_* spelling over the old __cppyy_* one. Returns a borrowed reference,
// or nullptr with no error set if there is no such hook.
inline PyObject* ExplicitPythonizeHookName(PyObject* pyclass) {
  // go through __dict__ instead of tp_getattr, see HasAttrDirect in
  // Pythonize.cxx for the rationale
  PyObject* name = nullptr;
  PyObject* dct = PyObject_GetAttr(pyclass, PyStrings::gDict);
  if (dct) {
    PyObject* candidates[] = {
        PyStrings::gExPythonize,
#if CPPYY_LEGACY_API
        PyStrings::gCppyyExPythonize,
#endif
    };
    for (PyObject* candidate : candidates) {
      PyObject* attr = PyObject_GetItem(dct, candidate);
      if (attr) {
        Py_DECREF(attr);
        name = candidate;
        break;
      }
      PyErr_Clear();
    }
    Py_DECREF(dct);
  }
  PyErr_Clear();
  return name;
}

// Normal pythonization hook of pyclass, through regular attribute lookup (so
// it falls through to base classes). Prefers the __cppjit_* spelling over the
// old __cppyy_* one. Returns a new reference, or nullptr with no error set if
// there is no such hook.
inline PyObject* GetPythonizeHook(PyObject* pyclass) {
  PyObject* func = PyObject_GetAttr(pyclass, PyStrings::gPythonize);
#if CPPYY_LEGACY_API
  if (!func) {
    PyErr_Clear();
    func = PyObject_GetAttr(pyclass, PyStrings::gCppyyPythonize);
  }
#endif
  if (!func)
    PyErr_Clear();
  return func;
}

} // namespace cppjit::cpyrt::compat

#endif // CPYRT_COMPATIBILITY_H
