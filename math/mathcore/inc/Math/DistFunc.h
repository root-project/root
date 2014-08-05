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


#ifndef ROOT_RConfigure
#include "RConfigure.h"
#endif




// pdf functions from MathCore
#ifndef ROOT_Math_PdfFuncMathCore
#include "Math/PdfFuncMathCore.h"
#endif

// all cdf are in MathCore now
#ifndef ROOT_Math_ProbFuncMathCore
#include "Math/ProbFuncMathCore.h"
#endif

//quantiles functions from mathcore
#ifndef ROOT_Math_QuantFuncMathCore
#include "Math/QuantFuncMathCore.h"
#endif

// include distributions from MathMore when is there
#ifdef R__HAS_MATHMORE

// // extra pdf functions from MathMore
#ifndef ROOT_Math_PdfFuncMathMore
#include "Math/PdfFuncMathMore.h"
#endif

// no -more extra cdf in MathMore
// #ifndef ROOT_Math_ProbFuncMathMore
// #include "Math/ProbFuncMathMore.h"
// #endif

// inverse (quantiles) are all in mathmore
#ifndef ROOT_Math_QuantFuncMathMore
#include "Math/QuantFuncMathMore.h"
#endif

#endif

#endif  // ROOT_Math_DistFunc
