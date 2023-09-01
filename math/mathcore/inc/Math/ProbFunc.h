// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

/**
    Header file declaring the cdf distributions present in both
    MathCore and optionally MathMore.
    The MathMore ones are included only if ROOT has been built with MathMore.
*/


#ifndef ROOT_Math_ProbFunc
#define ROOT_Math_ProbFunc


#include "RConfigure.h"


// all cdf are in MathCore others in mathmore
#include "Math/ProbFuncMathCore.h"

// include distributions from MathMore when is there
// #ifdef R__HAS_MATHMORE

// // extra cdf in MathMore
// #ifndef ROOT_Math_ProbFuncMathMore
// #include "Math/ProbFuncMathMore.h"
// #endif

// #endif

#endif  // ROOT_Math_ProbFunc
