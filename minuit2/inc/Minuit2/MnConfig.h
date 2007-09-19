// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnConfig
#define ROOT_Minuit2_MnConfig

// for alpha streams 
#if defined(__alpha) && !defined(linux)
#   include <standards.h>
#   ifndef __USE_STD_IOSTREAM
#   define __USE_STD_IOSTREAM
#   endif
#endif


#ifdef _MSC_VER
# pragma warning(disable:4244)  // conversion from __w64 to int
#endif

#if defined(__sun) && !defined(linux) 
#include <stdlib.h>
#endif

#endif
