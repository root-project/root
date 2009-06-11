// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:05:19 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Fitter

#ifndef ROOT_Fit_Fitter
#define ROOT_Fit_Fitter

/**
@defgroup Fit Fitting and Parameter Estimation

Classes used for fitting (regression analysis) and estimation of parameter values given a data sample. 
*/

#ifndef ROOT_Fit_DataVectorfwd
#include "Fit/DataVectorfwd.h"
#endif

#ifndef ROOT_Fit_FitConfig
#include "Fit/FitConfig.h"
#endif

#ifndef ROOT_Fit_FitResult
#include "Fit/FitResult.h"
#endif

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#include <memory>


namespace ROOT { 


   namespace Math { 
      class Minimizer;
   } 

   /**
      Namespace for the fitting classes
      @ingroup Fit
    */

   namespace Fit { 

/**
   @defgroup FitMain User Fitting classes

   Main Classes used for fitting a given data set
   @ingroup Fit
*/

//___________________________________________________________________________________
/** 
   Fitter class, entry point for performing all type of fits. 
   Fits are performed using the generic ROOT::Fit::Fitter::Fit method. 
   The inputs are the data points and a model function (using a ROOT::Math::IParamFunction)
   The result of the fit is returned and kept internally in the  ROOT::Fit::FitResult class. 
   The configuration of the fit (parameters, options, etc...) are specified in the 
   ROOT::Math::FitConfig class. 

   @ingroup FitMain
*/ 
class Fitter {

public: 

   typedef ROOT::Math::IParamMultiFunction       IModelFunction; 
   typedef ROOT::Math::IParamMultiGradFunction   IGradModelFunction;
   typedef ROOT::Math::IParamFunction            IModel1DFunction; 
   typedef ROOT::Math::IParamGradFunction        IGradModel1DFunction; 

   typedef ROOT::Math::IMultiGenFunction BaseFunc; 
   typedef ROOT::Math::IMultiGradFunction BaseGradFunc; 


   /** 
      Default constructor
   */ 
   Fitter (); 

   /** 
      Destructor
   */ 
   ~Fitter (); 

private: 

   /** 
      Copy constructor (disabled, class is not copyable)
   */ 
   Fitter(const Fitter &);

   /** 
      Assignment operator (disabled, class is not copyable) 
   */ 
   Fitter & operator = (const Fitter & rhs);  


public: 

   /** 
       fit a data set using any  generic model  function
       Pre-requisite on the function: 
   */ 
   template < class Data , class Function> 
   bool Fit( const Data & data, const Function & func) { 
      SetFunction(func);
      return Fit(data);
   }

   /** 
       fit a binned data set (default method: use chi2)
       To be implemented option to do likelihood bin fit
   */ 
   bool Fit(const BinData & data) { 
      return DoLeastSquareFit(data); 
   } 
   /** 
       fit an binned data set using loglikelihood method
   */ 
   bool Fit(const UnBinData & data) { 
      return DoLikelihoodFit(data); 
   } 

   /**
      Likelihood fit 
    */
   template <class Data> 
   bool LikelihoodFit(const Data & data) { 
      return DoLikelihoodFit(data);
   }

   /** 
       fit a data set using any  generic model  function
       Pre-requisite on the function: 
   */ 
   template < class Data , class Function> 
   bool LikelihoodFit( const Data & data, const Function & func) { 
      SetFunction(func);
      return DoLikelihoodFit(data);
   }

   /**
      fit using the given FCN function represented by a multi-dimensional function interface 
      (ROOT::Math::IMultiGenFunction). 
      Give optionally initial the parameter values and data size to have the fit Ndf correctly 
      set in the FitResult. 
      If the parameters values are not given (parameter pointers=0) the 
      current parameter settings are used. The parameter settings can be created before 
      by using the FitConfig::SetParamsSetting. If they have not been created they are created 
      automatically when the params pointer is not zero
    */
   bool FitFCN(const ROOT::Math::IMultiGenFunction & fcn, const double * params = 0, unsigned int dataSize = 0 ); 

   /**
      Fit using the given FCN function representing a multi-dimensional gradient function 
      interface (ROOT::Math::IMultiGradFunction). In this case the minimizer will use the 
      gradient information provided by the function. 
      For the other arguments same consideration as in the previous method
    */
   bool FitFCN(const ROOT::Math::IMultiGradFunction & fcn, const double * params = 0, unsigned int dataSize = 0); 

   /**
      Fit using the a generic FCN function as a C++ callable object implementing 
      double () (const double *) 
      The function dimension (i.e. the number of parameter) is needed in this case
      For the other arguments same consideration as in the previous methods
    */
   template <class Function>
   bool FitFCN(unsigned int npar, Function  fcn, const double * params = 0, unsigned int dataSize = 0);
      
   /**
      fit using user provided FCN with Minuit-like interface
      Parameter Settings must have be created before
    */
   typedef  void (* MinuitFCN_t )(int &npar, double *gin, double &f, double *u, int flag);
   bool FitFCN( MinuitFCN_t fcn);

   /**
      do a linear fit on a set of bin-data
    */
   bool LinearFit(const BinData & data) { return DoLinearFit(data); }

   /** 
       Set the fitted function (model function) from a parametric function interface
   */ 
   void  SetFunction(const IModelFunction & func); 
   /**
      Set the fitted function from a parametric 1D function interface
    */
   void  SetFunction(const IModel1DFunction & func); 

   /** 
       Set the fitted function (model function) from a parametric gradient function interface
   */ 
   void  SetFunction(const IGradModelFunction & func); 
   /**
      Set the fitted function from 1D gradient parametric function interface
    */
   void  SetFunction(const IGradModel1DFunction & func); 


   /**
      get fit result
   */
   const FitResult & Result() const { 
      assert( fResult.get() );
      return *fResult; 
   } 

   /**
      perform an error analysis on the result using the Hessian
      Errors are obtaied from the inverse of the Hessian matrix
      To be called only after fitting and when a minimizer supporting the Hessian calculations is used 
      otherwise an error (false) is returned.
      A new  FitResult with the Hessian result will be produced
    */
   bool CalculateHessErrors();  

   /**
      perform an error analysis on the result using MINOS
      To be called only after fitting and when a minimizer supporting MINOS is used 
      otherwise an error (false) is returned.
      The result will be appended in the fit result class 
      Optionally a vector of parameter indeces can be passed for selecting 
      the parameters to analyse using FitConfig::SetMinosErrors 
    */
   bool CalculateMinosErrors();  

   /**
      access to the fit configuration (const method)
   */
   const FitConfig & Config() const { return fConfig; } 

   /**
      access to the configuration (non const method)
   */
   FitConfig & Config() { return fConfig; } 

   /**
      query if fit is binned. In cse of false teh fit can be unbinned 
      or is not defined (like in case of fitting through a ::FitFCN)
    */
   bool IsBinFit() const { return fBinFit; } 

   /**
      return pointer to last used minimizer 
      (is NULL in case fit is not yet done)
      This pointer will be valid as far as the data, the objective function
      and the fitter class  have not been deleted.  
      To be used only after fitting.
      
    */
   ROOT::Math::Minimizer * GetMinimizer() { return fMinimizer.get(); } 

   /**
      return pointer to last used objective function 
      (is NULL in case fit is not yet done)
      This pointer will be valid as far as the data and the fitter class
      have not been deleted. To be used after the fitting
    */
   ROOT::Math::IMultiGenFunction * GetFCN() { return fObjFunction.get(); } 


protected: 

   /// least square fit 
   bool DoLeastSquareFit(const BinData & data); 
   /// binned likelihood fit
   bool DoLikelihoodFit(const BinData & data); 
   /// un-binned likelihood fit
   bool DoLikelihoodFit(const UnBinData & data); 
   /// linear least square fit 
   bool DoLinearFit(const BinData & data);

   /// do minimization
   template<class ObjFunc> 
   bool DoMinimization(const ObjFunc & f, unsigned int dataSize, const ROOT::Math::IMultiGenFunction * chifunc = 0); 

private: 

   bool fUseGradient;       // flag to indicate if using gradient or not

   bool fBinFit;            // flag to indicate if fit is binned (in case of false the fit is unbinned or undefined)

   IModelFunction * fFunc;  // copy of the fitted  function containing on output the fit result (managed by FitResult)

   FitConfig fConfig;       // fitter configuration (options and parameter settings)

   std::auto_ptr<ROOT::Fit::FitResult>  fResult;  //! pointer to the object containing the result of the fit

   std::auto_ptr<ROOT::Math::Minimizer>  fMinimizer;  //! pointer to used minimizer

   std::auto_ptr<ROOT::Math::IMultiGenFunction>  fObjFunction;  //! pointer to used objective function

}; 

   } // end namespace Fit

} // end namespace ROOT

// implementation of inline methods


#ifndef __CINT__


#ifndef ROOT_Math_WrappedFunction
#include "Math/WrappedFunction.h"
#endif

template<class Function>
bool ROOT::Fit::Fitter::FitFCN(unsigned int npar, Function f, const double * par, unsigned int datasize) {
   ROOT::Math::WrappedMultiFunction<Function> wf(f,npar); 
   return FitFCN(wf,par,datasize);
}

#endif  // endif __CINT__

#endif /* ROOT_Fit_Fitter */
