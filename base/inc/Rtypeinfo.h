// @(#)root/base:$Name:  $:$Id: Rtypeinfo.h,v 1.2 2002/05/03 14:41:25 brun Exp $
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

#if (defined(R__SOLARIS) && !defined(R__KCC)) || defined(R__HPUX)

// <typeinfo> includes <exception> which clashes with <math.h>
//#include <typeinfo.h>
namespace std { class type_info; }
using std::type_info;

#else

#include <typeinfo>
using std::type_info;

#endif

#endif
