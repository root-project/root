// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

/**
    Header file declaring the quantile distributions present in both
    MathCore and optionally MathMore.
    The MathMore ones are included only if ROOT has been built with MathMore.
*/


#ifndef ROOT_Math_QuantFunc
#define ROOT_Math_QuantFunc


#include "RConfigure.h"



#include "Math/QuantFuncMathCore.h"

// include distributions from MathMore when is there
#ifdef R__HAS_MATHMORE

// extra quantiles in MathMore
#include "Math/QuantFuncMathMore.h"

#endif

#endif  // ROOT_Math_QuantFunc
