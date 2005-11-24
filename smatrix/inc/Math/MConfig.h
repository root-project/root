// @(#)root/smatrix:$Name:  $:$Id: MConfig.hv 1.0 2005/11/24 12:00:00 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef ROOT_Math_MConfig_
#define ROOT_Math_MConfig

// for alpha streams 
#if defined(__alpha) && !defined(linux)
#   include <standards.h>
#   ifndef __USE_STD_IOSTREAM
#   define __USE_STD_IOSTREAM
#   endif
#endif


#if defined(__sun) && !defined(linux) 
#include <stdlib.h>
#endif


#endif
