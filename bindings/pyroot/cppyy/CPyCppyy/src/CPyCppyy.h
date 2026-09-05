#ifndef CPYCPPYY_CPYCPPYY_H
#define CPYCPPYY_CPYCPPYY_H

#ifdef _WIN32
// Disable warning C4275: non dll-interface class
#pragma warning (disable : 4275)
// Disable warning C4251: needs to have dll-interface to be used by clients
#pragma warning (disable : 4251)
// Disable warning C4800: 'int' : forcing value to bool
#pragma warning (disable : 4800)
// Avoid that pyconfig.h decides using a #pragma what library python library to use
//#define MS_NO_COREDLL 1
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

namespace CPyCppyy {
    typedef Py_ssize_t dim_t;
} // namespace CPyCppyy

#if PY_VERSION_HEX >= 0x030b0000
   typedef Py_ssize_t (*dict_lookup_func)(
       PyDictObject*, PyObject*, Py_hash_t, PyObject**);
#else
   typedef Py_ssize_t (*dict_lookup_func)(
       PyDictObject*, PyObject*, Py_hash_t, PyObject***, Py_ssize_t*);
#endif

// for 3.0 support (backwards compatibility, really)
#define CPyCppyy_PyText_Check              PyUnicode_Check
#define CPyCppyy_PyText_CheckExact         PyUnicode_CheckExact
#define CPyCppyy_PyText_AsString           PyUnicode_AsUTF8
#define CPyCppyy_PyText_AsStringChecked    PyUnicode_AsUTF8
#define CPyCppyy_PyText_GetSize            PyUnicode_GetSize
#define CPyCppyy_PyText_GET_SIZE           PyUnicode_GET_LENGTH
#define CPyCppyy_PyUnicode_GET_SIZE        PyUnicode_GET_LENGTH
#define CPyCppyy_PyText_FromFormat         PyUnicode_FromFormat
#define CPyCppyy_PyText_FromString         PyUnicode_FromString
#define CPyCppyy_PyText_InternFromString   PyUnicode_InternFromString
#define CPyCppyy_PyText_Append             PyUnicode_Append
#define CPyCppyy_PyText_AppendAndDel       PyUnicode_AppendAndDel
#define CPyCppyy_PyText_FromStringAndSize  PyUnicode_FromStringAndSize

#define _CPyCppyy_PyText_AsStringAndSize   PyUnicode_AsUTF8AndSize

static inline const char* CPyCppyy_PyText_AsStringAndSize(PyObject* pystr, Py_ssize_t* size)
{
    const char* cstr = _CPyCppyy_PyText_AsStringAndSize(pystr, size);
    if (!cstr && PyBytes_CheckExact(pystr)) {
        PyErr_Clear();
        PyBytes_AsStringAndSize(pystr, (char**)&cstr, size);
    }
    return cstr;
}

#define CPyCppyy_PyText_Type PyUnicode_Type

#define PyIntObject          PyLongObject
#define PyInt_Check          PyLong_Check
#define PyInt_AsLong         PyLong_AsLong
#define PyInt_AS_LONG        PyLong_AsLong
#define PyInt_AsSsize_t      PyLong_AsSsize_t
#define PyInt_CheckExact     PyLong_CheckExact
#define PyInt_FromLong       PyLong_FromLong
#define PyInt_FromSsize_t    PyLong_FromSsize_t

#define PyInt_Type      PyLong_Type

#define CPyCppyy_PyCapsule_New           PyCapsule_New
#define CPyCppyy_PyCapsule_CheckExact    PyCapsule_CheckExact
#define CPyCppyy_PyCapsule_GetPointer    PyCapsule_GetPointer

#define CPPYY__long__ "__int__"
#define CPPYY__idiv__ "__itruediv__"
#define CPPYY__div__  "__truediv__"
#define CPPYY__next__ "__next__"

#define Py_TPFLAGS_HAVE_RICHCOMPARE 0
#define Py_TPFLAGS_CHECKTYPES 0

#define PyClass_Check   PyType_Check

#define PyBuffer_Type   PyMemoryView_Type

#define CPyCppyy_PySliceCast   PyObject*
#define PyUnicode_GetSize      PyUnicode_GetLength

#define CPyCppyy_PyUnicode_AsWideChar PyUnicode_AsWideChar

#ifdef R__MACOSX
# if SIZEOF_SIZE_T == SIZEOF_INT
#   if defined(MAC_OS_X_VERSION_10_4)
#      define PY_SSIZE_T_FORMAT "%ld"
#   else
#      define PY_SSIZE_T_FORMAT "%d"
#   endif
# elif SIZEOF_SIZE_T == SIZEOF_LONG
#   define PY_SSIZE_T_FORMAT "%ld"
# endif
#else
# define PY_SSIZE_T_FORMAT "%zd"
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
#define CPyCppyy_PyCFunction_Call PyObject_Call

// vector call support
typedef PyObject* const* CPyCppyy_PyArgs_t;
static inline PyObject* CPyCppyy_PyArgs_GET_ITEM(CPyCppyy_PyArgs_t args, Py_ssize_t i) {
    return args[i];
}
static inline PyObject* CPyCppyy_PyArgs_SET_ITEM(CPyCppyy_PyArgs_t args, Py_ssize_t i, PyObject* item) {
    return ((PyObject**)args)[i] = item;
}
static inline Py_ssize_t CPyCppyy_PyArgs_GET_SIZE(CPyCppyy_PyArgs_t, size_t nargsf) {
    return PyVectorcall_NARGS(nargsf);
}
static inline CPyCppyy_PyArgs_t CPyCppyy_PyArgs_New(Py_ssize_t N) {
    return (CPyCppyy_PyArgs_t)PyMem_Malloc(N*sizeof(PyObject*));
}
static inline void CPyCppyy_PyArgs_DEL(CPyCppyy_PyArgs_t args) {
    PyMem_Free((void*)args);
}
#define CPyCppyy_PyObject_Call  PyObject_Vectorcall
inline PyObject* CPyCppyy_tp_call(
        PyObject* cb, CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds) {
    Py_ssize_t offset = Py_TYPE(cb)->tp_vectorcall_offset;
    vectorcallfunc func = *(vectorcallfunc*)(((char*)cb) + offset);
    return func(cb, args, nargsf, kwds);
}

#ifndef Py_TPFLAGS_HAVE_VECTORCALL
#define Py_TPFLAGS_HAVE_VECTORCALL _Py_TPFLAGS_HAVE_VECTORCALL
#endif

// weakref forced strong reference
#if PY_VERSION_HEX < 0x30d0000
static inline PyObject* CPyCppyy_GetWeakRef(PyObject* ref) {
    PyObject* pyobject = PyWeakref_GetObject(ref);
    if (!pyobject || pyobject == Py_None)
        return nullptr;
    Py_INCREF(pyobject);
    return pyobject;
}
#else
static inline PyObject* CPyCppyy_GetWeakRef(PyObject* ref) {
    PyObject* pyobject = nullptr;
    if (PyWeakref_GetRef(ref, &pyobject) != -1)
        return pyobject;
    return nullptr;
}
#endif

// C++ version of the cppyy API
#include "Cppyy.h"

// export macros for our own API
#include "CPyCppyy/CommonDefs.h"

// --- reusable PyTypeObject initializer tail -------------------------------
// Members appended to PyTypeObject in newer CPython releases. Every CPyCppyy
// type leaves all of these zero/null-initialized, so they share one tail.
// To support a future Python version, add one block below and one line to
// CPYCPPYY_PYTYPE_TAIL.

#if PY_VERSION_HEX >= 0x030c0000
#define CPYCPPYY_TP_WATCHED        , 0          /* tp_watched      (>= 3.12) */
#else
#define CPYCPPYY_TP_WATCHED
#endif

#if PY_VERSION_HEX >= 0x030d0000
#define CPYCPPYY_TP_VERSIONS_USED  , 0          /* tp_versions_used(>= 3.13) */
#else
#define CPYCPPYY_TP_VERSIONS_USED
#endif

#if PY_VERSION_HEX >= 0x030f0000
#define CPYCPPYY_TP_ITERITEM       , nullptr    /* _tp_iteritem    (>= 3.15) */
#else
#define CPYCPPYY_TP_ITERITEM
#endif

#define CPYCPPYY_PYTYPE_TAIL \
    CPYCPPYY_TP_WATCHED \
    CPYCPPYY_TP_VERSIONS_USED \
    CPYCPPYY_TP_ITERITEM
// --------------------------------------------------------------------------

#endif // !CPYCPPYY_CPYCPPYY_H
