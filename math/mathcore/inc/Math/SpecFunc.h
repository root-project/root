// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

/**
    Header file declaring the special functions present in both
    MathCore and  optionally MathMore.
    The MathMore ones are included only if ROOT has been built with MathMore.
*/



#ifndef ROOT_Math_SpecFunc
#define ROOT_Math_SpecFunc


#ifndef ROOT_RConfigure
#include "RConfigure.h"
#endif



#ifndef ROOT_Math_SpecFuncMathCore
#include "Math/SpecFuncMathCore.h"
#endif

#ifdef R__HAS_MATHMORE
// in case Mathmore exists include their GSL based special functions

#ifndef ROOT_Math_SpecFuncMathMore
#include "Math/SpecFuncMathMore.h"
#endif

#endif

#endif  // ROOT_Math_SpecFunc
