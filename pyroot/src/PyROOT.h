// @(#)root/pyroot:$Name:  $:$Id: PyROOT.h,v 1.7 2006/08/14 00:21:56 rdm Exp $
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
// Clear the _DEBUG that forces to use different library entry points
#undef _DEBUG
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
# define PY_SSIZE_T_FORMAT "%d"
# if !defined(PY_SSIZE_T_MIN)
#  define PY_SSIZE_T_MAX INT_MAX
#  define PY_SSIZE_T_MIN INT_MIN
# endif
#else
# ifdef R__MACOSX
#  define PY_SSIZE_T_FORMAT "%uzd"
# else
#  define PY_SSIZE_T_FORMAT "%zd"
# endif
#endif

#endif // !PYROOT_PYROOT_H
