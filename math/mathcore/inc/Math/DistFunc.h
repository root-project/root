// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

/**
    Header file declaring all distributions, pdf, cdf and quantiles present in
    MathCore and optionally MathMore.
    The MathMore ones are included only if ROOT has been built with MathMore.
*/

#ifndef ROOT_Math_DistFunc
#define ROOT_Math_DistFunc


#include "RConfigure.h"




// pdf functions from MathCore
#include "Math/PdfFuncMathCore.h"

// all cdf are in MathCore now
#include "Math/ProbFuncMathCore.h"

//quantiles functions from mathcore
#include "Math/QuantFuncMathCore.h"

// include distributions from MathMore when is there
#ifdef R__HAS_MATHMORE

// // extra pdf functions from MathMore
#include "Math/PdfFuncMathMore.h"

// no -more extra cdf in MathMore
// #ifndef ROOT_Math_ProbFuncMathMore
// #include "Math/ProbFuncMathMore.h"
// #endif

// inverse (quantiles) are all in mathmore
#include "Math/QuantFuncMathMore.h"

#endif

#endif  // ROOT_Math_DistFunc
