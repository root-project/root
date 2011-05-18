// @(#)root/mathcore:$Id$
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

#ifdef OLD
      template<class DimensionType> class IBaseFunction; 
      template<class DimensionType> class IGradientFunction; 


      /// tag for multi-dimensional functions
      struct MultiDim {}; 
      
      /// tag for one-dimensional functions
      struct OneDim {}; 


      typedef IBaseFunction<OneDim>        IGenFunction;   
      typedef IBaseFunction<MultiDim>      IMultiGenFunction; 

      typedef IGradientFunction<OneDim>        IGradFunction; 
      typedef IGradientFunction<MultiDim>      IMultiGradFunction; 
#endif

      class IBaseFunctionOneDim; 
      class IGradientFunctionOneDim; 
      class IBaseFunctionMultiDim; 
      class IGradientFunctionMultiDim; 


      typedef IBaseFunctionOneDim        IGenFunction;   
      typedef IBaseFunctionMultiDim      IMultiGenFunction; 

      typedef IGradientFunctionOneDim        IGradFunction; 
      typedef IGradientFunctionMultiDim      IMultiGradFunction; 
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IFunctionfwd */
