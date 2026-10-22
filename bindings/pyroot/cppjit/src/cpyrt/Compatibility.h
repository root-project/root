#ifndef CPYRT_COMPATIBILITY_H
#define CPYRT_COMPATIBILITY_H

#include "Python.h"

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

} // namespace cppjit::cpyrt::compat

#endif // CPYRT_COMPATIBILITY_H
