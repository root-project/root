// @(#)root/base:$Name:  $:$Id: Rtypeinfo.h,v 1.4 2002/05/03 16:57:32 brun Exp $
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

#if (defined(R__SOLARIS) && !defined(R__KCC)) 

// <typeinfo> includes <exception> which clashes with <math.h>
//#include <typeinfo.h>
namespace std { class type_info; }
using std::type_info;

#elif defined(R__HPUX)

#include <typeinfo>
// type_info is already in the global namespace.
#else

#include <typeinfo>
using std::type_info;

#endif

#endif
