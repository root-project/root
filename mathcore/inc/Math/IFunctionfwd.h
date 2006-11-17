// @(#)root/mathcore:$Name:  $:$Id: inc/Math/IFunctionfwd.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Tue Nov 14 14:38:48 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Defines Forward declaration for template IFunction class and useful typedefs 

#ifndef ROOT_Math_IFunctionfwd
#define ROOT_Math_IFunctionfwd


namespace ROOT { 

   namespace Math { 

      template<class DimensionType> class IBaseFunction; 
      template<class DimensionType> class IGradientFunction; 

      class OneDim; 
      class MultiDim; 
//       class Base; 
//       class Gradient; 

//       typedef IFunction<OneDim, Base>        IGenFunction;   
//       typedef IFunction<MultiDim, Base>      IMultiGenFunction; 

//       typedef IFunction<OneDim, Gradient>    IGradFunction; 
//       typedef IFunction<MultiDim, Gradient>  IMultiGradFunction; 

      typedef IBaseFunction<OneDim>        IGenFunction;   
      typedef IBaseFunction<MultiDim>      IMultiGenFunction; 

      typedef IGradientFunction<OneDim>        IGradFunction; 
      typedef IGradientFunction<MultiDim>      IMultiGradFunction; 
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IFunctionfwd */
