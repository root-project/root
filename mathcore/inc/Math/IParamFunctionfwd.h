// @(#)root/mathcore:$Name:  $:$Id: IParamFunctionfwd.h,v 1.3 2006/12/01 13:42:33 moneta Exp $
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


      template<class DimensionType> class IParametricFunction; 
      template<class DimensionType> class IParametricGradFunction; 

      typedef IParametricFunction<OneDim>        IParamFunction;   
      typedef IParametricFunction<MultiDim>      IParamMultiFunction; 

      typedef IParametricGradFunction<OneDim>        IParamGradFunction; 
      typedef IParametricGradFunction<MultiDim>      IParamMultiGradFunction; 



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IParamFunctionfwd */
