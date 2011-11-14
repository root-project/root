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

      // should maybe put this in a FitMethodFunctionfwd file
      template<class FunctionType> class BasicFitMethodFunction;

      // define the normal and gradient function
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGenFunction>  FitMethodFunction;      
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGradFunction> FitMethodGradFunction;

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
   After fitting the config of the fit will be modified to have the new values the resulting 
   parameter of the fit with step sizes equal to the errors. FitConfig can be preserved with 
   initial parameters by calling FitConfig.SetUpdateAfterFit(false);

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
       If data set is binned a least square fit is performed
       If data set is unbinned a maximum likelihood fit is done
       Pre-requisite on the function: 
       it must have 
   */ 
   template < class Data , class Function> 
   bool Fit( const Data & data, const Function & func) { 
      SetFunction(func);
      return Fit(data);
   }

   /** 
       Fit a binned data set using a least square fit (default method)
   */ 
   bool Fit(const BinData & data) { 
      return DoLeastSquareFit(data); 
   } 
   /** 
       fit an unbinned data set using loglikelihood method
   */ 
   bool Fit(const UnBinData & data, bool extended = false) { 
      return DoLikelihoodFit(data, extended); 
   } 

   /**
      Likelihood fit (unbinned or unbinned) depending on the type of data
      If Binned default is extended
      If Unbinned defult is NOT extended (for backward compatibility)
    */
   template <class Data> 
   bool LikelihoodFit(const Data & data ) { 
      return DoLikelihoodFit(data);
   }


   /**
      Likelihood fit using extended or not extended method
    */
   template <class Data> 
   bool LikelihoodFit(const Data & data, bool extended ) { 
      return DoLikelihoodFit(data, extended);
   }

   /** 
       fit a data set using any  generic model  function
       Pre-requisite on the function: 
   */ 
   template < class Data , class Function> 
   bool LikelihoodFit( const Data & data, const Function & func, bool extended) { 
      SetFunction(func);
      return DoLikelihoodFit(data, extended);
   }

   /**
      Fit using the a generic FCN function as a C++ callable object implementing 
      double () (const double *) 
      Note that the function dimension (i.e. the number of parameter) is needed in this case
      For the options see documentation for following methods FitFCN(IMultiGenFunction & fcn,..)
    */
   template <class Function>
   bool FitFCN(unsigned int npar, Function  & fcn, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false);

   /**
      Set a generic FCN function as a C++ callable object implementing 
      double () (const double *) 
      Note that the function dimension (i.e. the number of parameter) is needed in this case
      For the options see documentation for following methods FitFCN(IMultiGenFunction & fcn,..)
    */
   template <class Function>
   bool SetFCN(unsigned int npar, Function  & fcn, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false);

   /**
      Fit using the given FCN function represented by a multi-dimensional function interface 
      (ROOT::Math::IMultiGenFunction). 
      Give optionally the initial arameter values, data size to have the fit Ndf correctly 
      set in the FitResult and flag specifying if it is a chi2 fit. 
      Note that if the parameters values are not given (params=0) the 
      current parameter settings are used. The parameter settings can be created before 
      by using the FitConfig::SetParamsSetting. If they have not been created they are created 
      automatically when the params pointer is not zero. 
      Note that passing a params != 0 will set the parameter settings to the new value AND also the 
      step sizes to some pre-defined value (stepsize = 0.3 * abs(parameter_value) )
    */
   bool FitFCN(const ROOT::Math::IMultiGenFunction & fcn, const double * params = 0, unsigned int dataSize = 0, bool
      chi2fit = false); 

   /** 
       Fit using a FitMethodFunction interface. Same as method above, but now extra information
       can be taken from the function class
   */
   bool FitFCN(const ROOT::Math::FitMethodFunction & fcn, const double * params = 0); 

   /**
      Set the FCN function represented by a multi-dimensional function interface 
      (ROOT::Math::IMultiGenFunction) and optionally the initial parameters
      See also note above for the initial parameters for FitFCN
    */
   bool SetFCN(const ROOT::Math::IMultiGenFunction & fcn, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false); 

   /** 
       Set the objective function (FCN)  using a FitMethodFunction interface. 
       Same as method above, but now extra information can be taken from the function class
   */
   bool SetFCN(const ROOT::Math::FitMethodFunction & fcn, const double * params = 0); 

   /**
      Fit using the given FCN function representing a multi-dimensional gradient function 
      interface (ROOT::Math::IMultiGradFunction). In this case the minimizer will use the 
      gradient information provided by the function. 
      For the options same consideration as in the previous method
    */
   bool FitFCN(const ROOT::Math::IMultiGradFunction & fcn, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false); 

   /** 
       Fit using a FitMethodGradFunction interface. Same as method above, but now extra information
       can be taken from the function class
   */
   bool FitFCN(const ROOT::Math::FitMethodGradFunction & fcn, const double * params = 0); 

   /**
      Set the FCN function represented by a multi-dimensional gradient function interface 
      (ROOT::Math::IMultiGenFunction) and optionally the initial parameters
      See also note above for the initial parameters for FitFCN
    */
   bool SetFCN(const ROOT::Math::IMultiGradFunction & fcn, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false); 

   /** 
       Set the objective function (FCN)  using a FitMethodGradFunction interface. 
       Same as method above, but now extra information can be taken from the function class
   */
   bool SetFCN(const ROOT::Math::FitMethodGradFunction & fcn, const double * params = 0); 

      
   /**
      fit using user provided FCN with Minuit-like interface
      If npar = 0 it is assumed that the parameters are specified in the parameter settings created before
      For the options same consideration as in the previous method
    */
   typedef  void (* MinuitFCN_t )(int &npar, double *gin, double &f, double *u, int flag);
   bool FitFCN( MinuitFCN_t fcn, int npar = 0, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false);

   /**
      set objective function using user provided FCN with Minuit-like interface
      If npar = 0 it is assumed that the parameters are specified in the parameter settings created before
      For the options same consideration as in the previous method
    */
   bool SetFCN( MinuitFCN_t fcn, int npar = 0, const double * params = 0, unsigned int dataSize = 0, bool chi2fit = false);

   /**
      Perform a fit with the previously set FCN function. Require SetFCN before
    */
   bool FitFCN(); 

   /**
      Perform a simple FCN evaluation. FitResult will be modified and contain  the value of the FCN 
    */
   bool EvalFCN(); 


   /**
      do a linear fit on a set of bin-data
    */
   bool LinearFit(const BinData & data) { return DoLinearFit(data); }

   /** 
       Set the fitted function (model function) from a parametric function interface
   */ 
   void  SetFunction(const IModelFunction & func, bool useGradient = false); 
   /**
      Set the fitted function from a parametric 1D function interface
    */
   void  SetFunction(const IModel1DFunction & func, bool useGradient = false); 

   /** 
       Set the fitted function (model function) from a parametric gradient function interface
   */ 
   void  SetFunction(const IGradModelFunction & func, bool useGradient = true); 
   /**
      Set the fitted function from 1D gradient parametric function interface
    */
   void  SetFunction(const IGradModel1DFunction & func, bool useGradient = true); 


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
      The pointer should not be stored and will be invalided after performing a new fitting.  
      In this case a new instance of ROOT::Math::Minimizer will be re-created and can be 
      obtained calling again GetMinimizer()
    */
   ROOT::Math::Minimizer * GetMinimizer() const { return fMinimizer.get(); } 

   /**
      return pointer to last used objective function 
      (is NULL in case fit is not yet done)
      This pointer will be valid as far as the data and the fitter class
      have not been deleted. To be used after the fitting.
      The pointer should not be stored and will be invalided after performing a new fitting.
      In this case a new instance of the function pointer will be re-created and can be 
      obtained calling again GetFCN()  
    */
   ROOT::Math::IMultiGenFunction * GetFCN() const { return fObjFunction.get(); } 


   /**
      apply correction in the error matrix for the weights for likelihood fits
      This method can be called only after a fit. The 
      passed function (loglw2) is a log-likelihood function impelemented using the 
      sum of weight squared 
      When using FitConfig.SetWeightCorrection() this correction is applied 
      automatically when doing a likelihood fit (binned or unbinned)
   */
   bool ApplyWeightCorrection(const ROOT::Math::IMultiGenFunction & loglw2);


protected: 


   /// least square fit 
   bool DoLeastSquareFit(const BinData & data); 
   /// binned likelihood fit
   bool DoLikelihoodFit(const BinData & data, bool extended = true); 
   /// un-binned likelihood fit
   bool DoLikelihoodFit(const UnBinData & data, bool extended = false);  
   /// linear least square fit 
   bool DoLinearFit(const BinData & data);

   // initialize the minimizer 
   bool DoInitMinimizer(); 
   /// do minimization
   bool DoMinimization(const BaseFunc & f, const ROOT::Math::IMultiGenFunction * chifunc = 0); 
   // do minimization after having set obj function
   bool DoMinimization(const ROOT::Math::IMultiGenFunction * chifunc = 0); 
   // update config after fit 
   void DoUpdateFitConfig(); 
   // get function calls from the FCN 
   int GetNCallsFromFCN(); 

   // set 1D function
   void DoSetFunction(const IModel1DFunction & func, bool useGrad); 
   // set generic N-d function 
   void DoSetFunction(const IModelFunction & func, bool useGrad); 

private: 

   bool fUseGradient;       // flag to indicate if using gradient or not

   bool fBinFit;            // flag to indicate if fit is binned 
                            // in case of false the fit is unbinned or undefined)
                            // flag it is used to compute chi2 for binned likelihood fit

   int fFitType;   // type of fit   (0 undefined, 1 least square, 2 likelihood)

   int fDataSize;  // size of data sets (need for Fumili or LM fitters)

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
bool ROOT::Fit::Fitter::FitFCN(unsigned int npar, Function & f, const double * par, unsigned int datasize,bool chi2fit) {
   ROOT::Math::WrappedMultiFunction<Function &> wf(f,npar); 
   return FitFCN(wf,par,datasize,chi2fit);
}
template<class Function>
bool ROOT::Fit::Fitter::SetFCN(unsigned int npar, Function & f, const double * par, unsigned int datasize,bool chi2fit) {
   ROOT::Math::WrappedMultiFunction<Function &> wf(f,npar); 
   return SetFCN(wf,par,datasize,chi2fit);
}




#endif  // endif __CINT__

#endif /* ROOT_Fit_Fitter */
