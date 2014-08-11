// @(#)root/mathmore:$Id: OneDimFunctionAdapter.h 20063 2007-09-24 13:16:14Z moneta $
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
#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

#include <cassert>

namespace ROOT {

namespace Math {



// struct using for evaluating the function
template<class MultiFuncType>
struct EvaluatorOneDim {
   // evaluate function (in general case no param)
   static double F (MultiFuncType f, const double * x, const double *  = 0 ) {
      return f( x );
   }
};
// specialized for param functions
template<>
struct EvaluatorOneDim< const ROOT::Math::IParamMultiFunction &> {
   static double F ( const ROOT::Math::IParamMultiFunction &  f, const double * x, const double * p = 0 ) {
      return f( x, p );
   }
};


/**
   OneDimMultiFunctionAdapter class to wrap a multidimensional function in
   one dimensional one.
   Given a f(x1,x2,x3,....xn) transforms in a f( x_i) given the coordinate intex i and the vector x[]
   of the coordinates.
   It provides the possibility to copy and own the data array of the coordinates or to maintain internally a pointer to an external array
   for being more efficient. In this last case the user must garantee the life of the given passed pointer

   @ingroup  GenFunc

*/
template <class MultiFuncType = const ROOT::Math::IMultiGenFunction &>
class OneDimMultiFunctionAdapter : public ROOT::Math::IGenFunction  {

public:


   /**
      Constructor from the function object , pointer to an external array of x values
      and coordinate we want to adapt
   */
   OneDimMultiFunctionAdapter (MultiFuncType f, const double * x, unsigned int icoord =0, const double * p = 0 ) :
      fFunc(f),
      fX( const_cast<double *>(x) ), // wee need to modify x but then we restore it as before
      fParams(p),
      fCoord(icoord),
      fDim(0),
      fOwn(false)
   {
      assert(fX != 0);
   }
   /**
      Constructor from the function object , dimension of the function and
      and coordinate we want to adapt.
      The coordinate cached vector is created inside and eventually the values must be passed
      later with the SetX which will copy them
   */
   OneDimMultiFunctionAdapter (MultiFuncType f, unsigned int dim = 1, unsigned int icoord =0, const double * p = 0 ) :
      fFunc(f),
      fX(0 ),
      fParams(p),
      fCoord(icoord),
      fDim(dim),
      fOwn(true)
   {
      fX = new double[dim];
   }

   /**
      Destructor (no operations)
   */
   virtual ~OneDimMultiFunctionAdapter ()  { if (fOwn && fX) delete [] fX; }

   /**
      clone
   */
   virtual OneDimMultiFunctionAdapter * Clone( ) const {
      if (fOwn) {
         OneDimMultiFunctionAdapter * f =  new OneDimMultiFunctionAdapter( fFunc, fDim, fCoord, fParams);
         std::copy(fX, fX+fDim, f->fX);
         return f;
      }
      else
         return new OneDimMultiFunctionAdapter( fFunc, fX, fCoord, fParams);
   }

public:

   /**
       Set X values in case vector is own, iterator size must match previous
       set dimension
   */
   template<class Iterator>
   void SetX(Iterator begin, Iterator end) {
      if (fOwn) std::copy(begin, end, fX);
   }


   /**
      set pointer without copying the values
    */
   void SetX(double * x) {
      if (!fOwn) fX = x;
   }

   /**
       set values
   */
   void SetX(const double * x) {
      if (fOwn) std::copy(x, x+fDim, fX);
      else
         SetX( const_cast<double *>(x) ); // wee need to modify x but then we restore it as before
   }


   void SetCoord(int icoord) { fCoord=icoord;}

   // copy constructor
   OneDimMultiFunctionAdapter( const OneDimMultiFunctionAdapter & rhs) :
      fFunc(rhs.fFunc),
      fParams(rhs.fParams),
      fCoord(rhs.fCoord),
      fDim(rhs.fDim),
      fOwn(rhs.fOwn)
   {
      if (fOwn) {
         fX = new double[fDim];
         std::copy( rhs.fX, rhs.fX+fDim, fX);
      }
      else fX = rhs.fX;
   }


private:

   // dummy assignment (should never be called and clone must be used)
   OneDimMultiFunctionAdapter & operator= ( const OneDimMultiFunctionAdapter & rhs) {
      if (this == &rhs)  return *this;
      assert(false);
   }

   /**
      evaluate function at the  values x[] given in the constructor and
      as function of  the coordinate fCoord.
   */
   double DoEval(double x) const {
      if (fOwn) {
         fX[fCoord] = x;
         return EvaluatorOneDim<MultiFuncType>::F( fFunc, fX, fParams );
      }
      else {

         // case vector fX represents useful values needed later
         // need to modify fX and restore afterwards the original values
         double xprev = fX[fCoord]; // keep original value to restore in fX
         fX[fCoord] = x;
         double y = EvaluatorOneDim<MultiFuncType>::F( fFunc, fX, fParams );
         // restore original values
         fX[fCoord] = xprev;
         return y;
      }
   }


private:

   MultiFuncType fFunc;
   mutable double * fX;
   const double   * fParams;
   unsigned int fCoord;
   unsigned int fDim;
   bool fOwn;

};


/**
   OneDimParamFunctionAdapter class to wrap a multi-dim parameteric function in
   one dimensional one.
   Given a f(x[],p1,...pn) transforms in a f( p_i) given the param index i and the vectors x[] and p[]
   of the coordinates and parameters
   It has to be used carefully, since for efficiency reason it does not copy the parameter object
   but re-uses the given pointer for  the p[] vector.
   The ParamFuncType reference by default is not const because the operator()(x,p) is not a const method

   @ingroup  GenFunc

*/
template <class ParamFuncType = ROOT::Math::IParamMultiFunction &>
class OneDimParamFunctionAdapter :  public ROOT::Math::IGenFunction {

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

   /**
      clone
   */
   virtual OneDimParamFunctionAdapter * Clone( ) const {
      return new OneDimParamFunctionAdapter(fFunc, fX, fParams, fIpar);
   }

   // can use default copy constructor

private:

   /**
      evaluate function at the  values x[] given in the constructor and
      as function of  the coordinate fCoord.
   */
   double DoEval(double x) const {
      // HACK: use const_cast to modify the function values x[] and restore afterwards the original ones
      double * p = const_cast<double *>(fParams);
      double pprev = fParams[fIpar]; // keep original value to restore in fX
      p[fIpar] = x;
      double y =  fFunc( fX, p );
      p[fIpar] = pprev;
      return y;
   }


private:

   ParamFuncType fFunc;
   const double * fX;
   const double * fParams;
   unsigned int fIpar;

};




} // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_OneDimFunctionAdapter */
