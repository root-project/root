/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* array.h
*
* Array class
*
*  Constructor, copy constructor, destructor, operator overloading
* function overloading, reference type
*
**************************************************************************/
#ifndef G__ARRAY_H
#define G__ARRAY_H

#pragma security level0

#include <math.h>
#include <stdlib.h>
#include <iostream.h>

#ifndef G__ARRAYSL
# ifdef G__SHAREDLIB
#pragma include_noerr <array.dll>  
#  ifndef G__ARRAYSL
#include <array.c>  
#  endif
# else
#include <array.c>  
# endif
#endif // G__ARRAYSL

#include <darray.h>
#include <complex.h>

#ifndef G__CARRAYSL
# ifdef G__SHAREDLIB
#pragma include_noerr <carray.dll>
#  ifndef G__CARRAYSL
#include <carray.c>
#  endif
# else
#include <carray.c>
# endif
#endif // G__CARRAYSL

#include <carray.h>
#include <graphbuf.h>
#include <arrayiostream.h>

const double PI=3.141592;
const complex j(0.0,1.0);

#include <matrixstream.h>

#endif
