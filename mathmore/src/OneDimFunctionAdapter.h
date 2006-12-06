// @(#)root/mathcore:$Name:  $:$Id: src/OneDimFunctionAdapter.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Wed Dec  6 11:45:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class OneDimMultiFunctionAdapter

#ifndef ROOT_Math_OneDimFunctionAdapter
#define ROOT_Math_OneDimFunctionAdapter

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#include <cassert> 

namespace ROOT { 

namespace Math { 


/** 
   OneDimMultiFunctionAdapter class to wrap a multidimensional function in 
   one dimensional one. 
   Given a f(x1,x2,x3,....xn) transforms in a f( x_i) given the coordinate intex i and the vector x[]
   of the coordinates. 
   It has to be used with care, since for efficiency reason it does not copy the coordinate object 
   but re-uses the given pointer for  the x[] vector. 

   @ingroup  CppFunctions
   
*/ 
template <class MultiFuncType = const ROOT::Math::IMultiGenFunction &> 
class OneDimMultiFunctionAdapter {

public: 

  
   /** 
      Constructor from the function object , x value and coordinate we want to adapt
   */ 
   OneDimMultiFunctionAdapter (MultiFuncType f, const double * x, unsigned int icoord =0 ) : 
      fFunc(f), 
      fX(x ), 
      fCoord(icoord)
   {
      assert(fX != 0); 
   }  

   /** 
      Destructor (no operations)
   */ 
   ~OneDimMultiFunctionAdapter ()  {}  

public: 

   /**
      evaluate function at the  values x[] given in the constructor and  
      as function of  the coordinate fCoord. 
   */
   double operator()(double x) const {
      // HACK: use const_cast to modify the function values x[] and restore afterwards the original ones
      double * w = const_cast<double *>(fX); 
      double xprev = fX[fCoord]; // keep original value to restore in fX
      w[fCoord] = x; 
      double y =  fFunc( w );
      w[fCoord] = xprev; 
      return y; 
   }


private: 

   MultiFuncType fFunc; 
   //mutable std::vector<double> fX; 
   const double * fX; 
   unsigned int fCoord;
   

}; 


/** 
   OneDimParamFunctionAdapter class to wrap a parameteric function in 
   one dimensional one. 
   Given a f(x[],p1,...pn) transforms in a f( p_i) given the param index i and the vectors x[] and p[]
   of the coordinates and parameters
   It has to be used with care, since for efficiency reason it does not copy the parameter object 
   but re-uses the given pointer for  the p[] vector. 
   The ParamFuncType reference by default is not const because the operator()(x,p) is not a const method

   @ingroup  CppFunctions
   
*/ 
template <class ParamFuncType = ROOT::Math::IParamMultiFunction &> 
class OneDimParamFunctionAdapter {

public: 

  
   /** 
      Constructor from the function object , x value and coordinate we want to adapt
   */ 
   OneDimParamFunctionAdapter (ParamFuncType f, const double * x, const double * p, unsigned int ipar =0 ) : 
      fFunc(f), 
      fX(x ), 
      fParams(p), 
      fIpar(ipar)
   {
      assert(fX != 0); 
      assert(fParams != 0); 
   }  

   /** 
      Destructor (no operations)
   */ 
   ~OneDimParamFunctionAdapter ()  {}  

public: 

   /**
      evaluate function at the  values x[] given in the constructor and  
      as function of  the coordinate fCoord. 
   */
   double operator()(double x) const {
      // HACK: use const_cast to modify the function values x[] and restore afterwards the original ones
      double * p = const_cast<double *>(fParams); 
      double pprev = fParams[fIpar]; // keep original value to restore in fX
      p[fIpar] = x; 
      double y =  fFunc( fX, p );
      p[fIpar] = pprev; 
      return y; 
   }


private: 

   // need to be mutable since ofter operator()(x,p) is not a const method
   mutable ParamFuncType fFunc; 
   const double * fX; 
   const double * fParams; 
   unsigned int fIpar;
   

}; 




} // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_OneDimFunctionAdapter */
