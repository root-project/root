// @(#)root/mathcore:$Name:  $:$Id: IParamFunctionfwd.h,v 1.1 2006/11/17 18:18:47 moneta Exp $
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

      typedef IParamFunction<OneDim>        IParam1DFunction;   
      typedef IParamFunction<MultiDim>      IParamMultiFunction; 

      typedef IParamFunction<OneDim>        IParam1DGradFunction; 
      typedef IParamFunction<MultiDim>      IParamMultiGradFunction; 



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IParamFunctionfwd */
