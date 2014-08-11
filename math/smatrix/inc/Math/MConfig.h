// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_MConfig
#define ROOT_Math_MConfig

#if defined(__sun) && !defined(linux)
#include <stdlib.h>
// Solaris does not support expression like D1*(D1+1)/2 as template parameters
#define UNSUPPORTED_TEMPLATE_EXPRESSION
#endif


#endif
