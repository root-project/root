// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_PYROOT_H
#define PYROOT_PYROOT_H

#ifdef _WIN32
// Disable warning C4275: non dll-interface class
#pragma warning ( disable : 4275 )
// Disable warning C4251: needs to have dll-interface to be used by clients
#pragma warning ( disable : 4251 )
// Disable warning C4800: 'int' : forcing value to bool
#pragma warning ( disable : 4800 )
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

#endif

#include "Python.h"
#include "Rtypes.h"

// backwards compatibility, pre python 2.5
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#define PyInt_AsSsize_t PyInt_AsLong
#define PyInt_FromSsize_t PyInt_FromLong
# define PY_SSIZE_T_FORMAT "%d"
# if !defined(PY_SSIZE_T_MIN)
#  define PY_SSIZE_T_MAX INT_MAX
#  define PY_SSIZE_T_MIN INT_MIN
# endif
#define ssizeobjargproc intobjargproc
#define lenfunc         inquiry
#define ssizeargfunc    intargfunc

#define PyIndex_Check(obj) \
   (PyInt_Check(obj) || PyLong_Check(obj))

inline Py_ssize_t PyNumber_AsSsize_t( PyObject* obj, PyObject* ) {
   return (Py_ssize_t)PyLong_AsLong( obj );
}

#else
# ifdef R__MACOSX
#  if SIZEOF_SIZE_T == SIZEOF_INT
#    if defined(MAC_OS_X_VERSION_10_4)
#       define PY_SSIZE_T_FORMAT "%ld"
#    else
#       define PY_SSIZE_T_FORMAT "%d"
#    endif
#  elif SIZEOF_SIZE_T == SIZEOF_LONG
#    define PY_SSIZE_T_FORMAT "%ld"
#  endif
# else
#  define PY_SSIZE_T_FORMAT "%zd"
# endif
#endif

// the following should quiet Solaris
#ifdef Py_False
#undef Py_False
#define Py_False ( (PyObject*)(void*)&_Py_ZeroStruct )
#endif

#ifdef Py_True
#undef Py_True
#define Py_True ( (PyObject*)(void*)&_Py_TrueStruct )
#endif

#endif // !PYROOT_PYROOT_H
