// @(#)root/pyroot:$Name:  $:$Id: PyROOT.h,v 1.1 2004/04/27 06:28:48 brun Exp $
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

#include "Python.h"

#endif // !PYROOT_PYROOT_H
