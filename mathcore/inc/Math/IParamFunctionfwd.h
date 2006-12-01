// @(#)root/mathcore:$Name:  $:$Id: IParamFunctionfwd.h,v 1.2 2006/11/23 17:24:38 moneta Exp $
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


      template<class DimensionType> class IParamFunction; 
      template<class DimensionType> class IParamGradFunction; 

      typedef IParamFunction<OneDim>        IParam1DFunction;   
      typedef IParamFunction<MultiDim>      IParamMultiFunction; 

      typedef IParamGradFunction<OneDim>        IParam1DGradFunction; 
      typedef IParamGradFunction<MultiDim>      IParamMultiGradFunction; 



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IParamFunctionfwd */
