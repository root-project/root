// @(#)root/mathcore:$Id$
// Author: L. Moneta Mon Sep  4 17:00:10 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class Fitter


#include "Fit/Fitter.h"
#include "Fit/Chi2FCN.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "Fit/LogLikelihoodFCN.h"
#include "Math/Minimizer.h"
#include "Math/MinimizerOptions.h"
#include "Fit/MinimizerControlParams.h"
#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/FcnAdapter.h"
#include "Math/Error.h"

#include <memory> 

#include "Math/IParamFunction.h" 

#include "Math/MultiDimParamFunctionAdapter.h"

namespace ROOT { 

   namespace Fit { 



Fitter::Fitter() : 
   fUseGradient(false),
   fBinFit(false),
   fFunc(0)
{
   // Default constructor implementation.
   fResult = std::auto_ptr<ROOT::Fit::FitResult>(new ROOT::Fit::FitResult() );
}

Fitter::~Fitter() 
{
   // Destructor implementation.
   // delete function if not empty
   if (fFunc) delete fFunc; 
}

Fitter::Fitter(const Fitter & rhs) 
{
   // Implementation of copy constructor.
   // copy FitResult, FitConfig and clone fit function
   (*this) = rhs; 
}

Fitter & Fitter::operator = (const Fitter &rhs) 
{
   // Implementation of assignment operator.
   // dummy implementation, since it is private
   if (this == &rhs) return *this;  // time saving self-test
//    fUseGradient = rhs.fUseGradient; 
//    fBinFit = rhs.fBinFit; 
//    fResult = rhs.fResult;
//    fConfig = rhs.fConfig; 
//    // function is copied and managed by FitResult (maybe should use an auto_ptr)
//    fFunc = fResult.ModelFunction(); 
//    if (rhs.fFunc != 0 && fResult.ModelFunction() == 0) { // case no fit has been done yet - then clone 
//       if (fFunc) delete fFunc; 
//       fFunc = dynamic_cast<IModelFunction *>( (rhs.fFunc)->Clone() ); 
//       assert(fFunc != 0); 
//    }
   return *this; 
}

void Fitter::SetFunction(const IModelFunction & func) 
{
   fUseGradient = false;
 
   //  set the fit model function (clone the given one and keep a copy ) 
   //std::cout << "set a non-grad function" << std::endl; 

   fFunc = dynamic_cast<IModelFunction *> ( func.Clone() ); 
   
   // creates the parameter  settings 
   fConfig.CreateParamsSettings(*fFunc); 
}


void Fitter::SetFunction(const IModel1DFunction & func) 
{ 
   fUseGradient = false;
   //std::cout << "set a 1d function" << std::endl; 

   // function is cloned when creating the adapter
   fFunc = new ROOT::Math::MultiDimParamFunctionAdapter(func);

   // creates the parameter  settings 
   fConfig.CreateParamsSettings(*fFunc); 
}

void Fitter::SetFunction(const IGradModelFunction & func) 
{ 
   fUseGradient = true;
   //std::cout << "set a grad function" << std::endl; 
   //  set the fit model function (clone the given one and keep a copy ) 
   fFunc = dynamic_cast<IModelFunction *> ( func.Clone() ); 

   // creates the parameter  settings 
   fConfig.CreateParamsSettings(*fFunc); 
}


void Fitter::SetFunction(const IGradModel1DFunction & func) 
{ 
   //std::cout << "set a 1d grad function" << std::endl; 
   fUseGradient = true;
   // function is cloned when creating the adapter
   fFunc = new ROOT::Math::MultiDimParamGradFunctionAdapter(func);

   // creates the parameter  settings 
   fConfig.CreateParamsSettings(*fFunc); 
}


bool Fitter::FitFCN(const BaseFunc & fcn, const double * params, unsigned int dataSize) { 
   // fit a user provided FCN function
   // create fit parameter settings
   unsigned int npar  = fcn.NDim(); 
   if (params != 0 ) 
      fConfig.SetParamsSettings(npar, params);
   else {
      if ( fConfig.ParamsSettings().size() != npar) { 
         MATH_ERROR_MSG("Fitter::FitFCN","wrong fit parameter settings");
         return false;
      }
   }
   fBinFit = false; 

   // create Minimizer  
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );
   if (fMinimizer.get() == 0) return false; 

   if (fFunc && fResult->FittedFunction() == 0) delete fFunc; 
   fFunc = 0;

   return DoMinimization<BaseFunc> (fcn, dataSize); 
}

bool Fitter::FitFCN(const BaseGradFunc & fcn, const double * params, unsigned int dataSize) { 
   // fit a user provided FCN gradient function
   unsigned int npar  = fcn.NDim(); 
   if (params != 0  ) 
      fConfig.SetParamsSettings(npar, params);
   else {
      if ( fConfig.ParamsSettings().size() != npar) { 
         MATH_ERROR_MSG("Fitter::FitFCN","wrong fit parameter settings");
         return false;
      }
   }
   fBinFit = false; 

   // create Minimizer  (need to be done afterwards)
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );
   if (fMinimizer.get() == 0) return false; 

   if (fFunc && fResult->FittedFunction() == 0) delete fFunc; 
   fFunc = 0;

   // create fit configuration if null 
   return DoMinimization<BaseGradFunc> (fcn, dataSize); 
}

bool Fitter::FitFCN(MinuitFCN_t fcn ) { 
   // fit using Minuit style FCN type (global function pointer)  
   // create corresponfing objective function from that function
   int npar = fConfig.ParamsSettings().size(); 
   if (npar == 0) { 
      MATH_ERROR_MSG("Fitter::FitFCN","wrong fit parameter settings - npar = 0 ");
      return false;
   }
   fBinFit = false; 

   ROOT::Fit::FcnAdapter  newFcn(fcn,npar); 
   return FitFCN(newFcn); 
}

bool Fitter::DoLeastSquareFit(const BinData & data) { 
   // perform a chi2 fit on a set of binned data 


   // create Minimizer  
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );

   if (fMinimizer.get() == 0) return false; 
   
   if (fFunc == 0) return false; 

#ifdef DEBUG
   std::cout << "Fitter ParamSettings " << Config().ParamsSettings()[3].IsBound() << " lower limit " <<  Config().ParamsSettings()[3].LowerLimit() << " upper limit " <<  Config().ParamsSettings()[3].UpperLimit() << std::endl;
#endif

   fBinFit = true; 

   // check if fFunc provides gradient
   if (!fUseGradient) { 
      // do minimzation without using the gradient
      Chi2FCN<BaseFunc> chi2(data,*fFunc); 
      return DoMinimization<Chi2FCN<BaseFunc>::BaseObjFunction > (chi2, data.Size()); 
   } 
   else { 
      // use gradient 
      IGradModelFunction * gradFun = dynamic_cast<IGradModelFunction *>(fFunc); 
      if (gradFun != 0) { 
         Chi2FCN<BaseGradFunc> chi2(data,*gradFun); 
         return DoMinimization<Chi2FCN<BaseGradFunc>::BaseObjFunction > (chi2, data.Size()); 
      }
      MATH_ERROR_MSG("Fitter::DoLeastSquareFit","wrong type of function - it does not provide gradient");
   }
   return false; 
}

bool Fitter::DoLikelihoodFit(const BinData & data) { 

   // perform a likelihood fit on a set of binned data 

   // create Minimizer  
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );
   if (fMinimizer.get() == 0) return false; 
   if (fFunc == 0) return false; 

   // logl fit (error should be 0.5) set if different than default values (of 1)
   if (fConfig.MinimizerOptions().ErrorDef() == ROOT::Math::MinimizerOptions::DefaultErrorDef() ) { 
      fConfig.MinimizerOptions().SetErrorDef(0.5);
         fMinimizer->SetErrorDef(0.5);
   }

   fBinFit = true; 

   // create a chi2 function to be used for the equivalent chi-square
   Chi2FCN<BaseFunc> chi2(data,*fFunc); 

   if (!fUseGradient) { 
      // do minimzation without using the gradient
      PoissonLikelihoodFCN<BaseFunc> logl(data,*fFunc); 
      return DoMinimization<PoissonLikelihoodFCN<BaseFunc>::BaseObjFunction > (logl, data.Size(), &chi2); 
   } 
   else { 
      // check if fFunc provides gradient
      IGradModelFunction * gradFun = dynamic_cast<IGradModelFunction *>(fFunc); 
      if (gradFun != 0) { 
         // use gradient 
         PoissonLikelihoodFCN<BaseGradFunc> logl(data,*gradFun); 
         return DoMinimization<PoissonLikelihoodFCN<BaseGradFunc>::BaseObjFunction > (logl, data.Size(), &chi2); 
      }
      MATH_ERROR_MSG("Fitter::DoLikelihoodFit","wrong type of function - it does not provide gradient");

   }
   return false; 
}

bool Fitter::DoLikelihoodFit(const UnBinData & data) { 
   // perform a likelihood fit on a set of unbinned data 

   // create Minimizer  
   fMinimizer = std::auto_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );

   if (fMinimizer.get() == 0) return false; 
   
   if (fFunc == 0) return false; 

   fBinFit = false; 

#ifdef DEBUG
   int ipar = 0;
   std::cout << "Fitter ParamSettings " << Config().ParamsSettings()[ipar].IsBound() << " lower limit " <<  Config().ParamsSettings()[ipar].LowerLimit() << " upper limit " <<  Config().ParamsSettings()[ipar].UpperLimit() << std::endl;
#endif

   // logl fit (error should be 0.5) set if different than default values (of 1)
   if (fConfig.MinimizerOptions().ErrorDef() == ROOT::Math::MinimizerOptions::DefaultErrorDef() ) {
      fConfig.MinimizerOptions().SetErrorDef(0.5);
      fMinimizer->SetErrorDef(0.5);
   }

   if (!fUseGradient) { 
      // do minimzation without using the gradient
      LogLikelihoodFCN<BaseFunc> logl(data,*fFunc); 
      return DoMinimization<LogLikelihoodFCN<BaseFunc>::BaseObjFunction > (logl, data.Size()); 
   } 
   else { 
      // use gradient : check if fFunc provides gradient
      IGradModelFunction * gradFun = dynamic_cast<IGradModelFunction *>(fFunc); 
      if (gradFun != 0) { 
         LogLikelihoodFCN<BaseGradFunc> logl(data,*gradFun); 
         return DoMinimization<LogLikelihoodFCN<BaseGradFunc>::BaseObjFunction > (logl, data.Size()); 
      }
      MATH_ERROR_MSG("Fitter::DoLikelihoodFit","wrong type of function - it does not provide gradient");
   }      
   return false; 
}

bool Fitter::DoLinearFit(const BinData & data ) { 

   // perform a linear fit on a set of binned data 
   std::string  prevminimizer = fConfig.MinimizerType();  
   fConfig.SetMinimizer("Linear"); 

   fBinFit = true; 

   bool ret =  DoLeastSquareFit(data); 
   fConfig.SetMinimizer(prevminimizer.c_str());
   return ret; 
}

// traits for distinhuishing fit methods functions from generic objective functions
template<class Func> 
struct ObjFuncTrait { 
   static unsigned int NCalls(const Func &  ) { return 0; }
   static int Type(const Func & ) { return -1; }
   static bool IsGrad() { return false; }
};
template<>
struct ObjFuncTrait<ROOT::Math::FitMethodFunction> { 
   static unsigned int NCalls(const ROOT::Math::FitMethodFunction & f ) { return f.NCalls(); }
   static int Type(const ROOT::Math::FitMethodFunction & f) { return f.Type(); }
   static bool IsGrad() { return false; }
};
template<>
struct ObjFuncTrait<ROOT::Math::FitMethodGradFunction> { 
   static unsigned int NCalls(const ROOT::Math::FitMethodGradFunction & f ) { return f.NCalls(); }
   static int Type(const ROOT::Math::FitMethodGradFunction & f) { return f.Type(); }
   static bool IsGrad() { return true; }
};


template<class ObjFunc> 
bool Fitter::DoMinimization(const ObjFunc & objFunc, unsigned int dataSize, const ROOT::Math::IMultiGenFunction * chi2func) { 

   // assert that params settings have been set correctly
   assert( fConfig.ParamsSettings().size() == objFunc.NDim() );

   // keep also a copy of FCN function and set this in minimizer so they will be managed together
   // (remember that cloned copy will still depends on data and model function pointers) 
   fObjFunction = std::auto_ptr<ROOT::Math::IMultiGenFunction> ( objFunc.Clone() ); 
   // in case of gradient function needs to downcast the pointer
   const ObjFunc * fcn = dynamic_cast<const ObjFunc *> (fObjFunction.get() );
   assert(fcn); 
   fMinimizer->SetFunction( *fcn);

   fMinimizer->SetVariables(fConfig.ParamsSettings().begin(), fConfig.ParamsSettings().end() ); 


   // if requested parabolic error do correct error  analysis by the minimizer (call HESSE) 
   if (fConfig.ParabErrors()) fMinimizer->SetValidError(true);


   bool ret = fMinimizer->Minimize(); 

#ifdef DEBUG
   std::cout << "ROOT::Fit::Fitter::DoMinimization : ncalls = " << ObjFuncTrait<ObjFunc>::NCalls(objFunc) << " type of objfunc " << ObjFuncTrait<ObjFunc>::Type(objFunc) << "  typeid: " << typeid(objFunc).name() << " prov gradient " << ObjFuncTrait<ObjFunc>::IsGrad() << std::endl;
#endif
   
   unsigned int ncalls =  ObjFuncTrait<ObjFunc>::NCalls(objFunc);
   int fitType =  ObjFuncTrait<ObjFunc>::Type(objFunc);

   fResult = std::auto_ptr<FitResult> ( new FitResult(*fMinimizer,fConfig, fFunc, ret, dataSize, fBinFit, chi2func, fConfig.MinosErrors(), ncalls ) );

   if (fConfig.NormalizeErrors() && fitType == ROOT::Math::FitMethodFunction::kLeastSquare ) fResult->NormalizeErrors(); 

   return ret; 
}



   } // end namespace Fit

} // end namespace ROOT

