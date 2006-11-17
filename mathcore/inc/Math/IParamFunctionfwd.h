// @(#)root/mathcore:$Name:  $:$Id: inc/Math/IParamFunctionfwd.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
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


      template<class DimensionType, class CapabilityType> class IParamFunction; 

      typedef IParamFunction<OneDim, Base>        IParam1DFunction;   
      typedef IParamFunction<MultiDim, Base>      IParamMultiFunction; 

      typedef IParamFunction<OneDim, Gradient>    IParam1DGradFunction; 
      typedef IParamFunction<MultiDim, Gradient>  IParamMultiGradFunction; 



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IParamFunctionfwd */
