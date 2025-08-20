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

      class IBaseFunctionOneDim;
      class IGradientFunctionOneDim;
      template<class T>
      class IBaseFunctionMultiDimTempl;
      template <class T>
      class IGradientFunctionMultiDimTempl;
      template <class T>
      class IGradientMultiDimTempl;

      template<class T>
      using IMultiGenFunctionTempl = IBaseFunctionMultiDimTempl<T>;
      using IBaseFunctionMultiDim = IBaseFunctionMultiDimTempl<double>;
      using IMultiGenFunction = IMultiGenFunctionTempl<double>;

      using IGradientFunctionMultiDim = IGradientFunctionMultiDimTempl<double>;
      using IGradientMultiDim = IGradientMultiDimTempl<double>;

      typedef IBaseFunctionOneDim        IGenFunction;
      typedef IBaseFunctionMultiDim      IMultiGenFunction;

      typedef IGradientFunctionOneDim        IGradFunction;
      typedef IGradientFunctionMultiDim      IMultiGradFunction;


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_IFunctionfwd */
