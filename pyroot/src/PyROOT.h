// @(#)root/pyroot:$Name:  $:$Id: PyROOT.h,v 1.9 2007/02/12 17:13:59 brun Exp $
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
#ifdef _DEBUG
#define _WASDEBUG
#undef _DEBUG
#endif
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

#ifdef _WIN32
#ifdef _WASDEBUG
#define _DEBUG
#undef _WASDEBUG
#endif
#endif

// backwards compatibility, pre python 2.5
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#define PyInt_AsSsize_t PyInt_AsLong
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

#include <iostream>

#endif // !PYROOT_PYROOT_H
