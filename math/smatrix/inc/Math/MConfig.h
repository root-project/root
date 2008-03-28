// @(#)root/smatrix:$Id$
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
// Solaris does not support expression like D1*(D1+1)/2 as template parameters
#define UNSUPPORTED_TEMPLATE_EXPRESSION
#endif


#endif
