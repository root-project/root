// @(#)root/mathcore:$Name:  $:$Id: IFunctionfwd.h,v 1.2 2006/11/20 11:05:56 moneta Exp $
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


      /// tag for multi-dimensional functions
      struct MultiDim {}; 
      
      /// tag for one-dimensional functions
      struct OneDim {}; 


      typedef IBaseFunction<OneDim>        IGenFunction;   
      typedef IBaseFunction<MultiDim>      IMultiGenFunction; 

      typedef IGradientFunction<OneDim>        IGradFunction; 
      typedef IGradientFunction<MultiDim>      IMultiGradFunction; 
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IFunctionfwd */
