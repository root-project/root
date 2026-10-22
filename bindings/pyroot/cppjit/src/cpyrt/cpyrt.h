#ifndef CPYRT_CPYRT_H
#define CPYRT_CPYRT_H

#ifdef _WIN32
// Disable warning C4275: non dll-interface class
#pragma warning(disable : 4275)
// Disable warning C4251: needs to have dll-interface to be used by clients
#pragma warning(disable : 4251)
// Disable warning C4800: 'int' : forcing value to bool
#pragma warning(disable : 4800)
// Avoid that pyconfig.h decides using a #pragma what library python library to
// use
// #define MS_NO_COREDLL 1
#endif

// to prevent problems with fpos_t and redefinition warnings
#if defined(linux)

#include <stdio.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _FILE_OFFSET_BITS
#undef _FILE_OFFSET_BITS
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#endif // linux

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <sys/types.h>

namespace cppjit::cpyrt {
typedef Py_ssize_t dim_t;
} // namespace cppjit::cpyrt

// like PyUnicode_AsUTF8AndSize, but also accepts bytes
static inline const char* cpyrt_PyText_AsStringAndSize(PyObject* pystr,
                                                       Py_ssize_t* size) {
  const char* cstr = PyUnicode_AsUTF8AndSize(pystr, size);
  if (!cstr && PyBytes_CheckExact(pystr)) {
    PyErr_Clear();
    PyBytes_AsStringAndSize(pystr, (char**)&cstr, size);
  }
  return cstr;
}

#ifdef R__MACOSX
#if SIZEOF_SIZE_T == SIZEOF_INT
#if defined(MAC_OS_X_VERSION_10_4)
#define PY_SSIZE_T_FORMAT "%ld"
#else
#define PY_SSIZE_T_FORMAT "%d"
#endif
#elif SIZEOF_SIZE_T == SIZEOF_LONG
#define PY_SSIZE_T_FORMAT "%ld"
#endif
#else
#define PY_SSIZE_T_FORMAT "%zd"
#endif

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#ifndef Py_RETURN_TRUE
#define Py_RETURN_TRUE return Py_INCREF(Py_True), Py_True
#endif

#ifndef Py_RETURN_FALSE
#define Py_RETURN_FALSE return Py_INCREF(Py_False), Py_False
#endif

// vector call support
typedef PyObject* const* cpyrt_PyArgs_t;
static inline PyObject* cpyrt_PyArgs_GET_ITEM(cpyrt_PyArgs_t args,
                                              Py_ssize_t i) {
  return args[i];
}
static inline PyObject* cpyrt_PyArgs_SET_ITEM(cpyrt_PyArgs_t args, Py_ssize_t i,
                                              PyObject* item) {
  return ((PyObject**)args)[i] = item;
}
static inline Py_ssize_t cpyrt_PyArgs_GET_SIZE(cpyrt_PyArgs_t, size_t nargsf) {
  return PyVectorcall_NARGS(nargsf);
}
static inline cpyrt_PyArgs_t cpyrt_PyArgs_New(Py_ssize_t N) {
  return (cpyrt_PyArgs_t)PyMem_Malloc(N * sizeof(PyObject*));
}
static inline void cpyrt_PyArgs_DEL(cpyrt_PyArgs_t args) {
  PyMem_Free((void*)args);
}
#define cpyrt_PyObject_Call PyObject_Vectorcall
inline PyObject* cpyrt_tp_call(PyObject* cb, cpyrt_PyArgs_t args, size_t nargsf,
                               PyObject* kwds) {
  Py_ssize_t offset = Py_TYPE(cb)->tp_vectorcall_offset;
  vectorcallfunc func = *(vectorcallfunc*)(((char*)cb) + offset);
  return func(cb, args, nargsf, kwds);
}

#ifndef Py_TPFLAGS_HAVE_VECTORCALL
#define Py_TPFLAGS_HAVE_VECTORCALL _Py_TPFLAGS_HAVE_VECTORCALL
#endif

// weakref forced strong reference
#if PY_VERSION_HEX < 0x30d0000
static inline PyObject* cpyrt_GetWeakRef(PyObject* ref) {
  PyObject* pyobject = PyWeakref_GetObject(ref);
  if (!pyobject || pyobject == Py_None)
    return nullptr;
  Py_INCREF(pyobject);
  return pyobject;
}
#else
static inline PyObject* cpyrt_GetWeakRef(PyObject* ref) {
  PyObject* pyobject = nullptr;
  if (PyWeakref_GetRef(ref, &pyobject) != -1)
    return pyobject;
  return nullptr;
}
#endif

// C++ version of the cppjit API
#include "cppjit_interop.h"

// export macros for our own API
#include "cpyrt/CommonDefs.h"

// --- reusable PyTypeObject initializer tail -------------------------------
// Members appended to PyTypeObject in newer CPython releases. Every cpyrt
// type leaves all of these zero/null-initialized, so they share one tail.
// To support a future Python version, add one block below and one line to
// CPYRT_PYTYPE_TAIL.

#if PY_VERSION_HEX >= 0x030c0000
#define CPYRT_TP_WATCHED , 0 /* tp_watched      (>= 3.12) */
#else
#define CPYRT_TP_WATCHED
#endif

#if PY_VERSION_HEX >= 0x030d0000
#define CPYRT_TP_VERSIONS_USED , 0 /* tp_versions_used(>= 3.13) */
#else
#define CPYRT_TP_VERSIONS_USED
#endif

#if PY_VERSION_HEX >= 0x030f0000
#define CPYRT_TP_ITERITEM , nullptr /* _tp_iteritem    (>= 3.15) */
#else
#define CPYRT_TP_ITERITEM
#endif

#define CPYRT_PYTYPE_TAIL                                                      \
  CPYRT_TP_WATCHED                                                             \
  CPYRT_TP_VERSIONS_USED                                                       \
  CPYRT_TP_ITERITEM
// --------------------------------------------------------------------------

#endif // !CPYRT_CPYRT_H
