// @(#)root/base:$Id$
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Rtypeinfo
#define ROOT_Rtypeinfo

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#if defined(R__SOLARIS)

// <typeinfo> includes <exception> which clashes with <math.h>
//#include <typeinfo.h>
namespace std { class type_info; }
using std::type_info;

#elif defined(R__GLOBALSTL)

#include <typeinfo>

#elif defined(R__WIN32)

// only has ::type_info without _HAS_EXCEPTIONS!
#include <typeinfo>
#if ! _HAS_EXCEPTIONS
namespace std { using ::type_info; }
#endif

#else

#include <typeinfo>
using std::type_info;

#endif

#endif
