// @(#)root/mathcore:$Name:  $:$Id: WrappedParamFunction.h,v 1.1 2006/12/06 15:08:52 moneta Exp $
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

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

//#include <iostream>
//#include <iterator>

namespace ROOT { 

   namespace Math { 


typedef double( * FreeParamMultiFunctionPtr ) (const double *, const double * ); 

/** 
   WrappedParamFunction class to wrap any multi-dimensional parameteric function 
   implementing an operator()(const double * , const double *) 
   in an interface-like IParamFunction
*/ 
template< typename FuncPtr =  FreeParamMultiFunctionPtr   >
class WrappedParamFunction : public IParamMultiFunction {

public: 

   /** 
      Constructor a wrapped function from a pointer to a callable object and an iterator specifying begin and end 
      of parameters
   */ 
   template<class Iterator> 
   WrappedParamFunction (const FuncPtr & func, unsigned int dim, Iterator begin, Iterator end) : 
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(begin,end) )
   {}

   /** 
      Constructor a wrapped function from a non - const pointer to a callable object and an iterator specifying begin and end of parameters. This constructor is needed in the case FuncPtr is a std::auto_ptr which has a copy ctor taking non const objects
   */ 
   template<class Iterator> 
   WrappedParamFunction (FuncPtr & func, unsigned int dim, Iterator begin, Iterator end) : 
      fFunc(func),
      fDim(dim),
      fParams(std::vector<double>(begin,end) )
   {}

   /// clone the function
   IMultiGenFunction * Clone() const { 
      return new WrappedParamFunction(fFunc, fDim, fParams.begin(), fParams.end()); 
   }

   const double * Parameters() const { 
      return  &(fParams.front()); 
   }

   void SetParameters(const double * p)  { 
      std::copy(p, p+NPar(), fParams.begin() );
   }

   unsigned int NPar() const { return fParams.size(); }

   unsigned int NDim() const { return fDim; }

   // re-implement this since is more efficient
   double operator() (const double * x, const double * p) { 
      return (*fFunc)( x, p );
   }

private: 
   
   /// evaluate the function
   double DoEval(const double * x) const { 
//      std::cout << x << "  " << *x << "   " << fParams.size() << "  " << &fParams[0] << "  " << fParams[0] << std::endl; 
      return (*fFunc)( x, &(fParams.front()) );
   }


   mutable FuncPtr fFunc; 
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
*/ 
template< typename FuncPtr =  FreeMultiFunctionPtr   >
class WrappedParamFunctionGen : public IParamMultiFunction {

public: 

   /** 
      Constructor a wrapped function from a pointer to a generic callable object implemention operator()(const double), the new function dimension, the number of parameters (number of fixed variables) and an array specifying the index of the fixed variables. 
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
      This constructor is needed in the case FuncPtr is a std::auto_ptr which has a copy ctor taking non const objects
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
   IMultiGenFunction * Clone() const { 
      return new WrappedParamFunctionGen(fFunc, fDim, fParams.size() , &fParams.front(), &fParIndices.front()); 
   }

private: 
   // copy ctor 
   WrappedParamFunctionGen(const  WrappedParamFunctionGen & rhs) {}
   WrappedParamFunctionGen & operator=(const  WrappedParamFunctionGen & rhs) { return *this;}

public:

   const double * Parameters() const { 
      return  &(fParams.front()); 
   }

   void SetParameters(const double * p)  { 
      std::copy(p, p+NPar(), fParams.begin() );
      unsigned int npar = NPar();
      for (unsigned int i = 0; i < npar; ++i) { 
         unsigned int j = fParIndices[i]; 
         assert ( j  < npar + fDim);
         fX[j] = fParams[i];
      } 
   }

   unsigned int NPar() const { return fParams.size(); }

   unsigned int NDim() const { return fDim; }

//    // re-implement this since is more efficient
//    double operator() (const double * x, const double * p) { 
//       unsigned int n = fX.size(); 
//       unsigned int npar = fParams.size(); 
//       unsigned j = 0; 
//       return (*fFunc)( fX);
//    }

private: 
   
   /// evaluate the function
   double DoEval(const double * x) const { 

      unsigned int npar = NPar();
      
//       std::cout << this << fDim << " x : "; 
//       std::ostream_iterator<double> oix(std::cout," ,  ");
//       std::copy(x, x+fDim, oix);
//       std::cout << std::endl;
//       std::cout << "npar " << npar << std::endl;
//       std::cout <<  fVarIndices.size() << std::endl;
//       assert ( fVarIndices.size() == fDim);  // otherwise something is wrong

      for (unsigned int i = 0; i < fDim; ++i) { 
         unsigned int j = fVarIndices[i]; 
         assert ( j  < npar + fDim);
         fX[ j ] = x[i];
      }
//       std::cout << "X : (";
//       std::ostream_iterator<double> oi(std::cout," ,  ");
//       std::copy(fX.begin(), fX.end(), oi);
//       std::cout << std::endl;

      return (*fFunc)( &fX.front() );
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
      for (unsigned int i = 0; i < npar; ++i) { 
         unsigned int j = fParIndices[i]; 
         assert ( j  < npar + fDim);
         fX[j] = fParams[i];
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
