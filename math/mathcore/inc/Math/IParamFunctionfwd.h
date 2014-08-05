// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 14:38:52 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Forward declarations for template class  IParamFunction class

#ifndef ROOT_Math_IParamFunctionfwd
#define ROOT_Math_IParamFunctionfwd

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

namespace ROOT {

   namespace Math {

      class IParametricFunctionOneDim;
      class IParametricGradFunctionOneDim;
      class IParametricFunctionMultiDim;
      class IParametricGradFunctionMultiDim;

      typedef IParametricFunctionOneDim        IParamFunction;
      typedef IParametricFunctionMultiDim      IParamMultiFunction;

      typedef IParametricGradFunctionOneDim        IParamGradFunction;
      typedef IParametricGradFunctionMultiDim      IParamMultiGradFunction;


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IParamFunctionfwd */
