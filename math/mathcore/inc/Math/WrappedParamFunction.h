// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Nov 23 10:38:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class WrappedParamFunction

#ifndef ROOT_Math_WrappedParamFunction
#define ROOT_Math_WrappedParamFunction

#include "Math/IParamFunction.h"

//#include <iostream>
//#include <iterator>

#include <vector>


namespace ROOT {

   namespace Math {


typedef double( * FreeParamMultiFunctionPtr ) (const double *, const double * );

/**
   WrappedParamFunction class to wrap any multi-dimensional function object
   implementing the operator()(const double * x, const double * p)
   in an interface-like IParamFunction with a vector storing and caching internally the
   parameter values

   @ingroup  ParamFunc

*/
template< typename FuncPtr =  FreeParamMultiFunctionPtr   >
class WrappedParamFunction : public IParamMultiFunction {

public:

   /**
      Constructor a wrapped function from a pointer to a callable object, the function dimension and number of parameters
      which are set to zero by default
   */
   WrappedParamFunction (FuncPtr  func, unsigned int dim = 1, unsigned int npar = 0, double * par = 0) :
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(npar) )
   {
      if (par != 0) std::copy(par,par+npar,fParams.begin() );
   }

//    /**
//       Constructor a wrapped function from a non-const pointer to a callable object, the function dimension and number of parameters
//       which are set to zero by default
//       This constructor is needed in the case FuncPtr is a std::unique_ptr which has a copy ctor taking non const objects
//    */
//    WrappedParamFunction (FuncPtr & func, unsigned int dim = 1, unsigned int npar = 0, double * par = 0) :
//       fFunc(func),
//       fDim(dim),
//       fParams(std::vector<double>(npar) )
//    {
//       if (par != 0) std::copy(par,par+npar,fParams.begin() );
//    }

   /**
      Constructor a wrapped function from a pointer to a callable object, the function dimension and an iterator specifying begin and end
      of parameters
   */
   template<class Iterator>
   WrappedParamFunction (FuncPtr func, unsigned int dim, Iterator begin, Iterator end) :
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(begin,end) )
   {}

//    /**
//       Constructor a wrapped function from a non - const pointer to a callable object, the function dimension and an iterator specifying begin and end of parameters.
//       This constructor is needed in the case FuncPtr is a std::unique_ptr which has a copy ctor taking non const objects
//    */
//    template<class Iterator>
//    WrappedParamFunction (FuncPtr func, unsigned int dim, Iterator begin, Iterator end) :
//       fFunc(func),
//       fDim(dim),
//       fParams(std::vector<double>(begin,end) )
//    {}

   /// clone the function
   IMultiGenFunction * Clone() const override {
      return new WrappedParamFunction(fFunc, fDim, fParams.begin(), fParams.end());
   }

   const double * Parameters() const override {
      return fParams.empty() ? nullptr : &fParams.front();
   }

   void SetParameters(const double * p) override  {
      std::copy(p, p+NPar(), fParams.begin() );
   }

   unsigned int NPar() const override { return fParams.size(); }

   unsigned int NDim() const override { return fDim; }


private:

   /// evaluate the function given values and parameters (requested interface)
   double DoEvalPar(const double * x, const double * p) const override {
      return (*fFunc)( x, p );
   }


   FuncPtr fFunc;
   unsigned int fDim;
   std::vector<double> fParams;



};


typedef double( * FreeMultiFunctionPtr ) (const double *);

/**
   WrappedParamGenFunction class to wrap any multi-dimensional function
   implementing the operator()(const double * )
   in an interface-like IParamFunction, by fixing some of the variables and define them as
   parameters.
   i.e. transform any multi-dim function in a parametric function

   @ingroup  ParamFunc

*/
template< typename FuncPtr =  FreeMultiFunctionPtr   >
class WrappedParamFunctionGen : public IParamMultiFunction {

public:

   /**
      Constructor a wrapped function from a pointer to a generic callable object implementation operator()(const double *), the new function dimension, the number of parameters (number of fixed variables) and an array specifying the index of the fixed variables which became
      parameters in the new API
   */

   WrappedParamFunctionGen (const FuncPtr & func, unsigned int dim, unsigned int npar, const double * par, const unsigned int * idx) :
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(par,par+npar) ),
      fParIndices(std::vector<unsigned int>(idx, idx + npar) ),
      fX(std::vector<double>(npar+dim) )  // cached vector
   {
      DoInit();
   }

   /**
      Constructor as before but taking now a non - const pointer to a callable object.
      This constructor is needed in the case FuncPtr is a std::unique_ptr which has a copy ctor taking non const objects
   */
   WrappedParamFunctionGen (FuncPtr & func, unsigned int dim, unsigned int npar, const double * par, const unsigned int * idx) :
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(par,par+npar) ),
      fParIndices(std::vector<unsigned int>(idx, idx + npar) ),
      fX(std::vector<double>(npar+dim) ) // cached vector
   {
      DoInit();
   }

   /// clone the function
   IMultiGenFunction * Clone() const override {
      return new WrappedParamFunctionGen(fFunc, fDim, fParams.size(), fParams.empty() ? nullptr : &fParams.front(), fParIndices.empty() ? nullptr : &fParIndices.front());
   }

private:
   // copy ctor
   WrappedParamFunctionGen(const  WrappedParamFunctionGen &);   // not implemented
   WrappedParamFunctionGen & operator=(const  WrappedParamFunctionGen &); // not implemented

public:

   const double * Parameters() const override {
      return fParams.empty() ? nullptr : &fParams.front();
   }

   void SetParameters(const double * p) override  {
      unsigned int npar = NPar();
      std::copy(p, p+ npar, fParams.begin() );
      SetParValues(npar, p);
   }

   unsigned int NPar() const override { return fParams.size(); }

   unsigned int NDim() const override { return fDim; }

//    // re-implement this since is more efficient
//    double operator() (const double * x, const double * p) {
//       unsigned int n = fX.size();
//       unsigned int npar = fParams.size();
//       unsigned j = 0;
//       return (*fFunc)( fX);
//    }

private:

   /// evaluate the function (re-implement for being more efficient)
   double DoEval(const double * x) const override {

//       std::cout << this << fDim << " x : ";
//       std::ostream_iterator<double> oix(std::cout," ,  ");
//       std::copy(x, x+fDim, oix);
//       std::cout << std::endl;
//       std::cout << "npar " << npar << std::endl;
//       std::cout <<  fVarIndices.size() << std::endl;
//       assert ( fVarIndices.size() == fDim);  // otherwise something is wrong

      for (unsigned int i = 0; i < fDim; ++i) {
         unsigned int j = fVarIndices[i];
         assert ( j  < NPar() + fDim);
         fX[ j ] = x[i];
      }
//       std::cout << "X : (";
//       std::ostream_iterator<double> oi(std::cout," ,  ");
//       std::copy(fX.begin(), fX.end(), oi);
//       std::cout << std::endl;

      return (*fFunc)( fX.empty() ? nullptr : &fX.front() );
   }


   /**
       implement the required IParamFunction interface
   */
   double DoEvalPar(const double * x, const double * p ) const override {
      SetParValues(NPar(), p);
      return DoEval(x);
   }


   void DoInit() {
      // calculate variable indices and set in X the parameter values
      fVarIndices.reserve(fDim);
      unsigned int npar = NPar();
      for (unsigned int i = 0; i < npar + fDim; ++i) {
         bool isVar = true;
         for (unsigned int j = 0; j < npar; ++j) {
            if (fParIndices[j] == i) {
               isVar = false;
               break;
            }
         }
         if (isVar) fVarIndices.push_back(i);
      }
      assert ( fVarIndices.size() == fDim);  // otherwise something is wrong

//       std::cout << "n variables " << fVarIndices.size() << std::endl;
//       std::ostream_iterator<int> oi(std::cout,"  ");
//       std::copy(fVarIndices.begin(), fVarIndices.end(), oi);
//       std::cout << std::endl;
//       assert( fVarIndices.size() == fDim);
//       std::cout << this << std::endl;

      // set parameter values in fX
      SetParValues(npar, fParams.empty() ? nullptr : &fParams.front());
      for (unsigned int i = 0; i < npar; ++i) {
         unsigned int j = fParIndices[i];
         assert ( j  < npar + fDim);
         fX[j] = fParams[i];
      }

   }

   // set the parameter values in the cached fX vector
   // make const because it might be called from const methods
   void SetParValues(unsigned int npar, const double * p) const {
      for (unsigned int i = 0; i < npar; ++i) {
         unsigned int j = fParIndices[i];
         assert ( j  < npar + fDim);
         fX[j] = p[i];
      }
   }


   mutable FuncPtr fFunc;
   unsigned int fDim;
   std::vector<double> fParams;
   std::vector<unsigned int> fVarIndices;
   std::vector<unsigned int> fParIndices;
   mutable std::vector<double> fX;



};


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_WrappedParamFunction */
