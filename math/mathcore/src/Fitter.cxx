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
#include "Math/FitMethodFunction.h"
#include "Fit/BasicFCN.h"
#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/FcnAdapter.h"
#include "Fit/FitConfig.h"
#include "Fit/FitResult.h"
#include "Math/Error.h"

#include <memory>

#include "Math/IParamFunction.h"

#include "Math/MultiDimParamFunctionAdapter.h"

// #include "TMatrixDSym.h"
// for debugging
//#include "TMatrixD.h"
// #include <iomanip>

namespace ROOT {

   namespace Fit {

// use a static variable to get default minimizer options for error def
// to see if user has changed it later on. If it has not been changed we set
// for the likelihood method an error def of 0.5
// t.b.d : multiply likelihood by 2 so have same error def definition as chi2
      double gDefaultErrorDef = ROOT::Math::MinimizerOptions::DefaultErrorDef();


Fitter::Fitter(const std::shared_ptr<FitResult> & result) :
   fResult(result)
{
   if (result->fFitFunc)  SetFunction(*fResult->fFitFunc); // this will create also the configuration
   if (result->fObjFunc)  fObjFunction = fResult->fObjFunc;
   if (result->fFitData)  fData = fResult->fFitData;
}

void Fitter::SetFunction(const IModelFunction & func, bool useGradient)
{

   fUseGradient = useGradient;
   if (fUseGradient) {
      const IGradModelFunction * gradFunc = dynamic_cast<const IGradModelFunction*>(&func);
      if (gradFunc) {
         SetFunction(*gradFunc, true);
         return;
      }
      else {
         MATH_WARN_MSG("Fitter::SetFunction","Requested function does not provide gradient - use it as non-gradient function ");
      }
   }
   fUseGradient = false;

   //  set the fit model function (clone the given one and keep a copy )
   //std::cout << "set a non-grad function" << std::endl;

   fFunc = std::shared_ptr<IModelFunction>(dynamic_cast<IModelFunction *>(func.Clone() ) );
   assert(fFunc);

   // creates the parameter  settings
   fConfig.CreateParamsSettings(*fFunc);
   fFunc_v.reset();
}

void Fitter::SetFunction(const IModel1DFunction & func, bool useGradient)
{
   fUseGradient = useGradient;
   if (fUseGradient) {
      const IGradModel1DFunction * gradFunc = dynamic_cast<const IGradModel1DFunction*>(&func);
      if (gradFunc) {
         SetFunction(*gradFunc, true);
         return;
      }
      else {
         MATH_WARN_MSG("Fitter::SetFunction","Requested function does not provide gradient - use it as non-gradient function ");
      }
   }
   fUseGradient = false;
   //std::cout << "set a 1d function" << std::endl;

   // function is cloned when creating the adapter
   fFunc = std::shared_ptr<IModelFunction>(new ROOT::Math::MultiDimParamFunctionAdapter(func));

   // creates the parameter  settings
   fConfig.CreateParamsSettings(*fFunc);
   fFunc_v.reset();
}

void Fitter::SetFunction(const IGradModelFunction & func, bool useGradient)
{
   fUseGradient = useGradient;
   //std::cout << "set a grad function" << std::endl;
   //  set the fit model function (clone the given one and keep a copy )
   fFunc = std::shared_ptr<IModelFunction>( dynamic_cast<IGradModelFunction *> ( func.Clone() ) );
   assert(fFunc);

   // creates the parameter  settings
   fConfig.CreateParamsSettings(*fFunc);
   fFunc_v.reset();
}


void Fitter::SetFunction(const IGradModel1DFunction & func, bool useGradient)
{
   //std::cout << "set a 1d grad function" << std::endl;
   fUseGradient = useGradient;
   // function is cloned when creating the adapter
   fFunc =  std::shared_ptr<IModelFunction>(new ROOT::Math::MultiDimParamGradFunctionAdapter(func));

   // creates the parameter  settings
   fConfig.CreateParamsSettings(*fFunc);
   fFunc_v.reset();
}


bool Fitter::DoSetFCN(bool extFcn, const ROOT::Math::IMultiGenFunction & fcn, const double * params, unsigned int dataSize, bool chi2fit) {
   // Set the objective function for the fit. First parameter specifies if function object is managed external or internal.
   // In case of an internal function object we need to clone because it is a temporary one
   // if params is not NULL create the parameter settings
   fUseGradient = false;
   unsigned int npar  = fcn.NDim();
   if (npar == 0) {
      MATH_ERROR_MSG("Fitter::SetFCN","FCN function has zero parameters ");
      return false;
   }
   if (params != 0 )
      fConfig.SetParamsSettings(npar, params);
   else {
      if ( fConfig.ParamsSettings().size() != npar) {
         MATH_ERROR_MSG("Fitter::SetFCN","wrong fit parameter settings");
         return false;
      }
   }

   fBinFit = chi2fit;
   fDataSize = dataSize;

   // store external provided FCN without cloning it
   // it will be cloned in fObjFunc after the fit
   if (extFcn) {
      fExtObjFunction = &fcn;
      fObjFunction.reset();
   }
   else {
      // case FCN is built from Minuit interface so function object is created internally in Fitter class
      // and needs to be cloned and managed
      fExtObjFunction = nullptr;
      fObjFunction.reset(fcn.Clone());
   }

   // in case a model function and data exists from a previous fit - reset shared-ptr
   if (fResult && fResult->FittedFunction() == 0 && fFunc)  fFunc.reset();
   if (fData) fData.reset();

   return true;
}
bool Fitter::SetFCN(const ROOT::Math::IMultiGenFunction & fcn, const double * params, unsigned int dataSize, bool chi2fit) {
   // set the objective function for the fit
   return DoSetFCN(true, fcn, params, dataSize, chi2fit);
}
bool Fitter::SetFCN(const ROOT::Math::IMultiGenFunction &fcn, const IModelFunction & func, const double *params, unsigned int dataSize, bool chi2fit) {
   // set the objective function for the fit and a model function
   if (!SetFCN(fcn, params, dataSize, chi2fit) ) return false;
   // need to set fFunc afterwards because SetFCN could reset fFUnc
   fFunc = std::shared_ptr<IModelFunction>(dynamic_cast<IModelFunction *>(func.Clone()));
   return (fFunc != nullptr);
}

bool Fitter::SetFCN(const ROOT::Math::IMultiGradFunction &fcn, const double *params, unsigned int dataSize,
                       bool chi2fit)
{
   // set the objective function for the fit
   // if params is not NULL create the parameter settings
   if (!SetFCN(static_cast<const ROOT::Math::IMultiGenFunction &>(fcn), params, dataSize, chi2fit))
      return false;
   fUseGradient = true;
   return true;
}

bool Fitter::SetFCN(const ROOT::Math::IMultiGradFunction &fcn, const IModelFunction &func, const double *params,
                    unsigned int dataSize, bool chi2fit)
{
   // set the objective function for the fit and a model function
   if (!SetFCN(fcn, params, dataSize, chi2fit) ) return false;
   fFunc = std::shared_ptr<IModelFunction>(dynamic_cast<IModelFunction *>(func.Clone()));
   return (fFunc != nullptr);
}

bool Fitter::SetFCN(const ROOT::Math::FitMethodFunction &fcn, const double *params)
{
   // set the objective function for the fit
   // if params is not NULL create the parameter settings
   bool chi2fit = (fcn.Type() == ROOT::Math::FitMethodFunction::kLeastSquare);
   if (!SetFCN(fcn, params, fcn.NPoints(), chi2fit))
      return false;
   fUseGradient = false;
   fFitType = fcn.Type();
   return true;
}

bool Fitter::SetFCN(const ROOT::Math::FitMethodGradFunction &fcn, const double *params)
{
   // set the objective function for the fit
   // if params is not NULL create the parameter settings
   bool chi2fit = (fcn.Type() == ROOT::Math::FitMethodGradFunction::kLeastSquare);
   if (!SetFCN(fcn, params, fcn.NPoints(), chi2fit))
      return false;
   fUseGradient = true;
   fFitType = fcn.Type();
   return true;
}

bool Fitter::FitFCN(const BaseFunc &fcn, const double *params, unsigned int dataSize, bool chi2fit)
{
   // fit a user provided FCN function
   // create fit parameter settings
   if (!SetFCN(fcn, params, dataSize, chi2fit))
      return false;
   return FitFCN();
}

bool Fitter::FitFCN(const BaseGradFunc &fcn, const double *params, unsigned int dataSize, bool chi2fit)
{
   // fit a user provided FCN gradient function

   if (!SetFCN(fcn, params, dataSize, chi2fit))
      return false;
   return FitFCN();
}

bool Fitter::FitFCN(const ROOT::Math::FitMethodFunction &fcn, const double *params)
{
   // fit using the passed objective function for the fit
   if (!SetFCN(fcn, params))
      return false;
   return FitFCN();
}

bool Fitter::FitFCN(const ROOT::Math::FitMethodGradFunction &fcn, const double *params)
{
   // fit using the passed objective function for the fit
   if (!SetFCN(fcn, params))
      return false;
   return FitFCN();
}

bool Fitter::SetFCN(MinuitFCN_t fcn, int npar, const double *params, unsigned int dataSize, bool chi2fit)
{
   // set TMinuit style FCN type (global function pointer)
   // create corresponfing objective function from that function

   if (npar == 0) {
      npar = fConfig.ParamsSettings().size();
      if (npar == 0) {
         MATH_ERROR_MSG("Fitter::FitFCN", "Fit Parameter settings have not been created ");
         return false;
      }
   }

   ROOT::Fit::FcnAdapter newFcn(fcn, npar);
   return DoSetFCN(false,newFcn, params, dataSize, chi2fit);
}

bool Fitter::FitFCN(MinuitFCN_t fcn, int npar, const double *params, unsigned int dataSize, bool chi2fit)
{
   // fit using Minuit style FCN type (global function pointer)
   // create corresponfing objective function from that function
   if (!SetFCN(fcn, npar, params, dataSize, chi2fit))
      return false;
   fUseGradient = false;
   return FitFCN();
}

bool Fitter::FitFCN()
{
   // fit using the previously set  FCN function


   if (!fExtObjFunction && !fObjFunction) {
      MATH_ERROR_MSG("Fitter::FitFCN", "Objective function has not been set");
      return false;
   }
   // look if FCN is of a known type and we can get retrieve the  model function and data objects
   if (!fFunc || !fData)
      ExamineFCN();
   // init the minimizer
   if (!DoInitMinimizer())
      return false;
   // perform the minimization
   return DoMinimization();
}

bool Fitter::EvalFCN()
{
   // evaluate the FCN using the stored values in fConfig

   if (fFunc && fResult->FittedFunction() == 0)
      fFunc.reset();

   if (!ObjFunction()) {
      MATH_ERROR_MSG("Fitter::FitFCN", "Objective function has not been set");
      return false;
   }
   // create a Fit result from the fit configuration
   fResult = std::make_shared<ROOT::Fit::FitResult>(fConfig);
   // evaluate one time the FCN
   double fcnval = (*ObjFunction())(fResult->GetParams());
   // update fit result
   fResult->fVal = fcnval;
   fResult->fNCalls++;
   return true;
}

bool Fitter::DoLeastSquareFit(const ROOT::EExecutionPolicy &executionPolicy)
{

   // perform a chi2 fit on a set of binned data
   std::shared_ptr<BinData> data = std::dynamic_pointer_cast<BinData>(fData);
   assert(data);

   // check function
   if (!fFunc && !fFunc_v) {
      MATH_ERROR_MSG("Fitter::DoLeastSquareFit", "model function is not set");
      return false;
   } else {

#ifdef DEBUG
      std::cout << "Fitter ParamSettings " << Config().ParamsSettings()[3].IsBound() << " lower limit "
                << Config().ParamsSettings()[3].LowerLimit() << " upper limit "
                << Config().ParamsSettings()[3].UpperLimit() << std::endl;
#endif

      fBinFit = true;
      fDataSize = data->Size();
      // check if fFunc provides gradient
      if (!fUseGradient) {
         // do minimization without using the gradient
         if (fFunc_v) {
            return DoMinimization(std::make_unique<Chi2FCN<BaseFunc, IModelFunction_v>>(data, fFunc_v, executionPolicy));
         } else {
            return DoMinimization(std::make_unique<Chi2FCN<BaseFunc>>(data, fFunc, executionPolicy));
         }
      } else {
         // use gradient
         if (fConfig.MinimizerOptions().PrintLevel() > 0)
            MATH_INFO_MSG("Fitter::DoLeastSquareFit", "use gradient from model function");

         if (fFunc_v) {
            std::shared_ptr<IGradModelFunction_v> gradFun = std::dynamic_pointer_cast<IGradModelFunction_v>(fFunc_v);
            if (gradFun) {
               return DoMinimization(std::make_unique<Chi2FCN<BaseGradFunc, IModelFunction_v>>(data, gradFun, executionPolicy));
            }
         } else {
            std::shared_ptr<IGradModelFunction> gradFun = std::dynamic_pointer_cast<IGradModelFunction>(fFunc);
            if (gradFun) {
               return DoMinimization(std::make_unique<Chi2FCN<BaseGradFunc>>(data, gradFun, executionPolicy));
            }
         }
         MATH_ERROR_MSG("Fitter::DoLeastSquareFit", "wrong type of function - it does not provide gradient");
      }
   }
   return false;
}

bool Fitter::DoBinnedLikelihoodFit(bool extended, const ROOT::EExecutionPolicy &executionPolicy)
{
   // perform a likelihood fit on a set of binned data
   // The fit is extended (Poisson logl_ by default

   std::shared_ptr<BinData> data = std::dynamic_pointer_cast<BinData>(fData);
   assert(data);

   bool useWeight = fConfig.UseWeightCorrection();

   // check function
   if (!fFunc && !fFunc_v) {
      MATH_ERROR_MSG("Fitter::DoBinnedLikelihoodFit", "model function is not set");
      return false;
   }

   // logl fit (error should be 0.5) set if different than default values (of 1)
   if (fConfig.MinimizerOptions().ErrorDef() == gDefaultErrorDef) {
      fConfig.MinimizerOptions().SetErrorDef(0.5);
   }

   if (useWeight && fConfig.MinosErrors()) {
      MATH_INFO_MSG("Fitter::DoBinnedLikelihoodFit", "MINOS errors cannot be computed in weighted likelihood fits");
      fConfig.SetMinosErrors(false);
   }

   fBinFit = true;
   fDataSize = data->Size();

   if (!fUseGradient) {
      // do minimization without using the gradient
      if (fFunc_v) {
         // create a chi2 function to be used for the equivalent chi-square
         Chi2FCN<BaseFunc, IModelFunction_v> chi2(data, fFunc_v);
         auto logl = std::make_unique<PoissonLikelihoodFCN<BaseFunc, IModelFunction_v>>(data, fFunc_v, useWeight, extended, executionPolicy);
         return (useWeight) ? DoWeightMinimization(std::move(logl),&chi2) : DoMinimization(std::move(logl),&chi2);
      } else {
         // create a chi2 function to be used for the equivalent chi-square
         Chi2FCN<BaseFunc> chi2(data, fFunc);
         auto logl = std::make_unique<PoissonLikelihoodFCN<BaseFunc>>(data, fFunc, useWeight, extended, executionPolicy);
         return (useWeight) ? DoWeightMinimization(std::move(logl),&chi2) : DoMinimization(std::move(logl),&chi2);
      }
   } else {
      if (fConfig.MinimizerOptions().PrintLevel() > 0)
            MATH_INFO_MSG("Fitter::DoLikelihoodFit", "use gradient from model function");
      // not-extended is not implemented in this case
      if (!extended) {
         MATH_WARN_MSG("Fitter::DoBinnedLikelihoodFit",
                     "Not-extended binned fit with gradient not yet supported - do an extended fit");
         extended = true;
      }
      if (fFunc_v) {
         // create a chi2 function to be used for the equivalent chi-square
         Chi2FCN<BaseFunc, IModelFunction_v> chi2(data, fFunc_v);
         std::shared_ptr<IGradModelFunction_v> gradFun = std::dynamic_pointer_cast<IGradModelFunction_v>(fFunc_v);
         if (!gradFun) {
            MATH_ERROR_MSG("Fitter::DoBinnedLikelihoodFit", "wrong type of function - it does not provide gradient");
            return false;
         }
         auto logl = std::make_unique<PoissonLikelihoodFCN<BaseGradFunc, IModelFunction_v>>(data, gradFun, useWeight, extended, executionPolicy);
         // do minimization
         return (useWeight) ? DoWeightMinimization(std::move(logl),&chi2) : DoMinimization(std::move(logl),&chi2);
      } else {
         // create a chi2 function to be used for the equivalent chi-square
         Chi2FCN<BaseFunc> chi2(data, fFunc);
         // check if fFunc provides gradient
         std::shared_ptr<IGradModelFunction> gradFun = std::dynamic_pointer_cast<IGradModelFunction>(fFunc);
         if (!gradFun) {
            MATH_ERROR_MSG("Fitter::DoBinnedLikelihoodFit", "wrong type of function - it does not provide gradient");
            return false;
         }
         // use gradient for minimization
         auto logl = std::make_unique<PoissonLikelihoodFCN<BaseGradFunc>>(data, gradFun, useWeight, extended, executionPolicy);
         // do minimization
         return (useWeight) ? DoWeightMinimization(std::move(logl),&chi2) : DoMinimization(std::move(logl),&chi2);
      }
   }
   return false;
}

bool Fitter::DoUnbinnedLikelihoodFit(bool extended, const ROOT::EExecutionPolicy &executionPolicy) {
   // perform a likelihood fit on a set of unbinned data

   std::shared_ptr<UnBinData> data = std::dynamic_pointer_cast<UnBinData>(fData);
   assert(data);

   bool useWeight = fConfig.UseWeightCorrection();

   if (!fFunc && !fFunc_v) {
      MATH_ERROR_MSG("Fitter::DoUnbinnedLikelihoodFit","model function is not set");
      return false;
   }

   if (useWeight && fConfig.MinosErrors() ) {
      MATH_INFO_MSG("Fitter::DoUnbinnedLikelihoodFit","MINOS errors cannot be computed in weighted likelihood fits");
      fConfig.SetMinosErrors(false);
   }


   fBinFit = false;
   fDataSize = data->Size();

#ifdef DEBUG
   int ipar = 0;
   std::cout << "Fitter ParamSettings " << Config().ParamsSettings()[ipar].IsBound() << " lower limit " <<  Config().ParamsSettings()[ipar].LowerLimit() << " upper limit " <<  Config().ParamsSettings()[ipar].UpperLimit() << std::endl;
#endif

   // logl fit (error should be 0.5) set if different than default values (of 1)
   if (fConfig.MinimizerOptions().ErrorDef() == gDefaultErrorDef ) {
      fConfig.MinimizerOptions().SetErrorDef(0.5);
   }

   if (!fUseGradient) {
      // do minimization without using the gradient
      if (fFunc_v ){
         auto logl = std::make_unique<LogLikelihoodFCN<BaseFunc, IModelFunction_v>>(data, fFunc_v, useWeight, extended, executionPolicy);
         // do minimization
         return (useWeight) ? DoWeightMinimization(std::move(logl)) : DoMinimization(std::move(logl));
     } else {
         auto logl = std::make_unique<LogLikelihoodFCN<BaseFunc>>(data, fFunc, useWeight, extended, executionPolicy);
         return (useWeight) ? DoWeightMinimization(std::move(logl)) : DoMinimization(std::move(logl));
     }
   } else {
      // use gradient : check if fFunc provides gradient
      if (fConfig.MinimizerOptions().PrintLevel() > 0)
            MATH_INFO_MSG("Fitter::DoUnbinnedLikelihoodFit", "use gradient from model function");
      if (extended) {
         MATH_WARN_MSG("Fitter::DoUnbinnedLikelihoodFit",
                        "Extended unbinned fit with gradient not yet supported - do a not-extended fit");
         extended = false;
      }
      if (fFunc_v) {
         std::shared_ptr<IGradModelFunction_v> gradFun = std::dynamic_pointer_cast<IGradModelFunction_v>(fFunc_v);
         if (!gradFun) {
            MATH_ERROR_MSG("Fitter::DoUnbinnedLikelihoodFit", "wrong type of function - it does not provide gradient");
            return false;
         }
         auto logl = std::make_unique<LogLikelihoodFCN<BaseGradFunc, IModelFunction_v>>(data, gradFun, useWeight, extended, executionPolicy);
         return (useWeight) ? DoWeightMinimization(std::move(logl)) : DoMinimization(std::move(logl));
      } else {
         std::shared_ptr<IGradModelFunction> gradFun = std::dynamic_pointer_cast<IGradModelFunction>(fFunc);
         if (!gradFun) {
            MATH_ERROR_MSG("Fitter::DoUnbinnedLikelihoodFit", "wrong type of function - it does not provide gradient");
            return false;
         }
         auto logl = std::make_unique<LogLikelihoodFCN<BaseGradFunc>>(data, gradFun, useWeight, extended, executionPolicy);
         return (useWeight) ? DoWeightMinimization(std::move(logl)) : DoMinimization(std::move(logl));
      }
   }
   return false;
}


bool Fitter::DoLinearFit( ) {

   std::shared_ptr<BinData> data = std::dynamic_pointer_cast<BinData>(fData);
   assert(data);

   // perform a linear fit on a set of binned data
   std::string  prevminimizer = fConfig.MinimizerType();
   fConfig.SetMinimizer("Linear");

   fBinFit = true;

   bool ret =  DoLeastSquareFit();
   fConfig.SetMinimizer(prevminimizer.c_str());
   return ret;
}


bool Fitter::CalculateHessErrors() {
   // compute the Hesse errors according to configuration
   // set in the parameters and append value in fit result
   if (!ObjFunction()) {
      MATH_ERROR_MSG("Fitter::CalculateHessErrors","Objective function has not been set");
      return false;
   }

   // need a special treatment in case of weighted likelihood fit
   // (not yet implemented)
   if (fFitType == 2 && fConfig.UseWeightCorrection() ) {
      MATH_ERROR_MSG("Fitter::CalculateHessErrors","Re-computation of Hesse errors not implemented for weighted likelihood fits");
      MATH_INFO_MSG("Fitter::CalculateHessErrors","Do the Fit using configure option FitConfig::SetParabErrors()");
      return false;
   }

     // a fit Result pointer must exist when a minimizer exists
   if (fMinimizer && !fResult ) {
      MATH_ERROR_MSG("Fitter::CalculateHessErrors", "FitResult has not been created");
      return false;
   }

   // update  minimizer (recreate if not done or if name has changed
   if (!DoUpdateMinimizerOptions()) {
       MATH_ERROR_MSG("Fitter::CalculateHessErrors","Error re-initializing the minimizer");
       return false;
   }

   if (!fMinimizer ) {
      // this should not happen
      MATH_ERROR_MSG("Fitter::CalculateHessErrors", "Need to do a fit before calculating the errors");
      assert(false);
      return false;
   }

   //run Hesse
   bool ret = fMinimizer->Hesse();
   if (!ret) MATH_WARN_MSG("Fitter::CalculateHessErrors","Error when calculating Hessian");

   // update minimizer results with what comes out from Hesse
   // in case is empty - create from a FitConfig
   if (fResult->IsEmpty() )
      fResult.reset(new ROOT::Fit::FitResult(fConfig) );

   // update obj function in case it was an external one
   if (fExtObjFunction) fObjFunction.reset(fExtObjFunction->Clone());
   fResult->fObjFunc = fObjFunction;

   // re-give a minimizer instance in case it has been changed
   ret |= fResult->Update(fMinimizer, fConfig, ret);

   // when possible get ncalls from FCN and set in fit result
   if (fFitType != ROOT::Math::FitMethodFunction::kUndefined ) {
      fResult->fNCalls = GetNCallsFromFCN();
   }

   // set also new errors in FitConfig
   if (fConfig.UpdateAfterFit() && ret) DoUpdateFitConfig();

   return ret;
}


bool Fitter::CalculateMinosErrors() {
   // compute the Minos errors according to configuration
   // set in the parameters and append value in fit result
   // normally Minos errors are computed just after the minimization
   // (in DoMinimization) aftewr minimizing if the
   //  FitConfig::MinosErrors() flag is set

   if (!fMinimizer) {
       MATH_ERROR_MSG("Fitter::CalculateMinosErrors","Minimizer does not exist - cannot calculate Minos errors");
       return false;
   }

   if (!fResult || fResult->IsEmpty() ) {
       MATH_ERROR_MSG("Fitter::CalculateMinosErrors","Invalid Fit Result - cannot calculate Minos errors");
       return false;
   }

   if (fFitType == 2 && fConfig.UseWeightCorrection() ) {
      MATH_ERROR_MSG("Fitter::CalculateMinosErrors","Computation of MINOS errors not implemented for weighted likelihood fits");
      return false;
   }

   // update  minimizer (but cannot re-create in this case). Must use an existing one
   if (!DoUpdateMinimizerOptions(false)) {
       MATH_ERROR_MSG("Fitter::CalculateHessErrors","Error re-initializing the minimizer");
       return false;
   }

   // set flag to compute Minos error to false in FitConfig to avoid that
   // following minimizaiton calls perform unwanted Minos error calculations
   /// fConfig.SetMinosErrors(false);


   const std::vector<unsigned int> & ipars = fConfig.MinosParams();
   unsigned int n = (ipars.size() > 0) ? ipars.size() : fResult->Parameters().size();
   bool ok = false;

   int iparNewMin = 0;
   int iparMax = n;
   int iter = 0;
   // rerun minos for the parameters run before a new Minimum has been found
   do {
      if (iparNewMin > 0)
         MATH_INFO_MSG("Fitter::CalculateMinosErrors","Run again Minos for some parameters because a new Minimum has been found");
      iparNewMin = 0;
      for (int i = 0; i < iparMax; ++i) {
         double elow, eup;
         unsigned int index = (ipars.size() > 0) ? ipars[i] : i;
         bool ret = fMinimizer->GetMinosError(index, elow, eup);
         // flags case when a new minimum has been found
         if ((fMinimizer->MinosStatus() & 8) != 0) {
            iparNewMin = i;
         }
         if (ret)
            fResult->SetMinosError(index, elow, eup);
         ok |= ret;
      }

      iparMax = iparNewMin;
      iter++;  // to avoid infinite looping
   }
   while( iparNewMin > 0 && iter < 10);
   if (!ok) {
       MATH_ERROR_MSG("Fitter::CalculateMinosErrors","Minos error calculation failed for all the selected parameters");
   }

   // update obj function in case it was an external one
   if (fExtObjFunction) fObjFunction.reset(fExtObjFunction->Clone());
   fResult->fObjFunc = fObjFunction;

   // re-give a minimizer instance in case it has been changed
   // but maintain previous valid status. Do not set result to false if minos failed
   ok &= fResult->Update(fMinimizer, fConfig, fResult->IsValid());

   return ok;
}



// traits for distinguishing fit methods functions from generic objective functions
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

bool Fitter::DoInitMinimizer() {
   //initialize minimizer by creating it
   // and set there the objective function
   // obj function must have been set before
   auto objFunction = ObjFunction();
   if (!objFunction) {
      MATH_ERROR_MSG("Fitter::DoInitMinimizer","Objective function has not been set");
      return false;
   }

   // check configuration and objective  function
   if ( fConfig.ParamsSettings().size() != objFunction->NDim() ) {
      MATH_ERROR_MSG("Fitter::DoInitMinimizer","wrong function dimension or wrong size for FitConfig");
      return false;
   }

   // create first Minimizer
   // using an auto_Ptr will delete the previous existing one
   fMinimizer = std::shared_ptr<ROOT::Math::Minimizer> ( fConfig.CreateMinimizer() );
   if (fMinimizer.get() == 0) {
      MATH_ERROR_MSG("Fitter::DoInitMinimizer","Minimizer cannot be created");
      return false;
   }

   // in case of gradient function one needs to downcast the pointer
   if (fUseGradient) {
      const ROOT::Math::IMultiGradFunction * gradfcn = dynamic_cast<const ROOT::Math::IMultiGradFunction *> (objFunction );
      if (!gradfcn) {
         MATH_ERROR_MSG("Fitter::DoInitMinimizer","wrong type of function - it does not provide gradient");
         return false;
      }
      fMinimizer->SetFunction( *gradfcn);
      // set also Hessian if available
      if (Config().MinimizerType() == "Minuit2") {
         const ROOT::Math::FitMethodGradFunction *fitGradFcn =
            dynamic_cast<const ROOT::Math::FitMethodGradFunction *>(gradfcn);
         if (fitGradFcn && fitGradFcn->HasHessian()) {
            auto hessFcn = [=](const std::vector<double> &x, double *hess) {
               unsigned int ndim = x.size();
               unsigned int nh = ndim * (ndim + 1) / 2;
               std::vector<double> h(nh);
               bool ret = fitGradFcn->Hessian(x.data(), h.data());
               if (!ret) return false;
               for (unsigned int i = 0; i < ndim; i++) {
                  for (unsigned int j = 0; j <= i; j++) {
                     unsigned int index = j + i * (i + 1) / 2; // formula for j < i
                     hess[ndim * i + j] = h[index];
                     if (j != i)
                        hess[ndim * j + i] = h[index];
                  }
               }
               return true;
            };

            fMinimizer->SetHessianFunction(hessFcn);
         }
      }
   }
   else
      fMinimizer->SetFunction( *objFunction);


   fMinimizer->SetVariables(fConfig.ParamsSettings().begin(), fConfig.ParamsSettings().end() );

   // if requested parabolic error do correct error  analysis by the minimizer (call HESSE)
   if (fConfig.ParabErrors()) fMinimizer->SetValidError(true);

   return true;

}

bool Fitter::DoUpdateMinimizerOptions(bool canDifferentMinim ) {
   // update minimizer options when re-doing a Fit or computing Hesse or Minos errors


   // create a new minimizer if it is different type
   // minimizer type string stored in FitResult is "minimizer name" + " / " + minimizer algo
   std::string newMinimType = fConfig.MinimizerName();
   if (fMinimizer && fResult && newMinimType != fResult->MinimizerType()) {
      // if a different minimizer is allowed (e.g. when calling Hesse)
      if (canDifferentMinim) {
         std::string msg = "Using now " + newMinimType;
         MATH_INFO_MSG("Fitter::DoUpdateMinimizerOptions: ", msg.c_str());
         if (!DoInitMinimizer() )
            return false;
      }
      else {
         std::string msg = "Cannot change minimizer. Continue using " + fResult->MinimizerType();
         MATH_WARN_MSG("Fitter::DoUpdateMinimizerOptions",msg.c_str());
      }
   }

   // create minimizer if it was not done before
   if (!fMinimizer) {
      if (!DoInitMinimizer())
         return false;
   }

   // set new minimizer options (but not functions and parameters)
   fMinimizer->SetOptions(fConfig.MinimizerOptions());
   return true;
}

bool Fitter::DoMinimization(const ROOT::Math::IMultiGenFunction * chi2func) {
   // perform the minimization (assume we have already initialized the minimizer)

   assert(fMinimizer );

   bool isValid = fMinimizer->Minimize();

   if (!fResult) fResult = std::make_shared<FitResult>();

   fResult->FillResult(fMinimizer,fConfig, fFunc, isValid, fDataSize, fBinFit, chi2func );

   // if requested run Minos after minimization
   if (isValid && fConfig.MinosErrors()) {
      // minos error calculation will update also FitResult
      CalculateMinosErrors();
   }

      // when possible get number of calls from FCN and set in fit result
      if (fResult->fNCalls == 0 && fFitType != ROOT::Math::FitMethodFunction::kUndefined) {
         fResult->fNCalls = GetNCallsFromFCN();
   }

   // fill information in fit result
   // if using an external obj function clone it for storing in FitResult
   if (fExtObjFunction) fObjFunction.reset(fExtObjFunction->Clone());
   fResult->fObjFunc = fObjFunction;
   fResult->fFitData = fData;

#ifdef DEBUG
      std::cout << "ROOT::Fit::Fitter::DoMinimization : ncalls = " << fResult->fNCalls << " type of objfunc " << fFitFitResType << "  typeid: " << typeid(*fObjFunction).name() << " use gradient " << fUseGradient << std::endl;
#endif

   if (fConfig.NormalizeErrors() && fFitType == ROOT::Math::FitMethodFunction::kLeastSquare )
      fResult->NormalizeErrors();

   // set also new parameter values and errors in FitConfig
   if (fConfig.UpdateAfterFit() &&  isValid) DoUpdateFitConfig();

   return isValid;
}
template<class ObjFunc_t>
bool Fitter::DoMinimization(std::unique_ptr<ObjFunc_t>  objFunc, const ROOT::Math::IMultiGenFunction * chi2func) {
   // perform the minimization initializing the minimizer starting from a given obj function
   fFitType = objFunc->Type();
   fExtObjFunction = nullptr;
   fObjFunction = std::move(objFunc);
   if (!DoInitMinimizer()) return false;
   return DoMinimization(chi2func);
}
template<class ObjFunc_t>
bool Fitter::DoWeightMinimization(std::unique_ptr<ObjFunc_t> objFunc, const ROOT::Math::IMultiGenFunction * chi2func) {
   // perform the minimization initializing the minimizer starting from a given obj function
   // and apply afterwards the correction for weights. This applyies only for logL fitting
   this->fFitType = objFunc->Type();
   fExtObjFunction = nullptr;
   fObjFunction = std::move(objFunc);
   if (!DoInitMinimizer()) return false;
   if (!DoMinimization(chi2func)) return false;
   objFunc->UseSumOfWeightSquare();
   return ApplyWeightCorrection(*objFunc);
}


void Fitter::DoUpdateFitConfig() {
   // update the fit configuration after a fit using the obtained result
   if (fResult->IsEmpty() || !fResult->IsValid() ) return;
   for (unsigned int i = 0; i < fConfig.NPar(); ++i) {
      ParameterSettings & par = fConfig.ParSettings(i);
      par.SetValue( fResult->Value(i) );
      if (fResult->Error(i) > 0) par.SetStepSize( fResult->Error(i) );
   }
}

int Fitter::GetNCallsFromFCN() {
   // retrieve ncalls from the fit method functions
   // this function is called when minimizer does not provide a way of returning the nnumber of function calls
   int ncalls = 0;
   if (!fUseGradient) {
      const ROOT::Math::FitMethodFunction * fcn = dynamic_cast<const ROOT::Math::FitMethodFunction *>(fObjFunction.get());
      if (fcn) ncalls = fcn->NCalls();
   }
   else {
      const ROOT::Math::FitMethodGradFunction * fcn = dynamic_cast<const ROOT::Math::FitMethodGradFunction*>(fObjFunction.get());
      if (fcn) ncalls = fcn->NCalls();
   }
   return ncalls;
}


bool Fitter::ApplyWeightCorrection(const ROOT::Math::IMultiGenFunction & loglw2, bool minimizeW2L) {
   // apply correction for weight square
   // Compute Hessian of the loglikelihood function using the sum of the weight squared
   // This method assumes:
   // - a fit has been done before and a covariance matrix exists
   // - the objective function is a likelihood function and Likelihood::UseSumOfWeightSquare()
   //    has been called before

   if (fMinimizer.get() == 0) {
      MATH_ERROR_MSG("Fitter::ApplyWeightCorrection","Must perform first a fit before applying the correction");
      return false;
   }

   unsigned int n = loglw2.NDim();
   // correct errors for weight squared
   std::vector<double> cov(n*n);
   bool ret = fMinimizer->GetCovMatrix(&cov[0] );
   if (!ret) {
      MATH_ERROR_MSG("Fitter::ApplyWeightCorrection","Previous fit has no valid Covariance matrix");
      return false;
   }
   // need to use new obj function computed with weight-square
   std::shared_ptr<ROOT::Math::IMultiGenFunction>  objFunc(loglw2.Clone());
   fObjFunction.swap( objFunc );

   // need to re-initialize the minimizer for the changes applied in the
   // objective functions
   if (!DoInitMinimizer()) return false;

   //std::cout << "Running Hesse ..." << std::endl;

   // run eventually before a minimization
   // ignore its error
   if (minimizeW2L) fMinimizer->Minimize();
   // run Hesse on the log-likelihood build using sum of weight squared
   ret = fMinimizer->Hesse();
   if (!ret) {
      MATH_ERROR_MSG("Fitter::ApplyWeightCorrection","Error running Hesse on weight2 likelihood - cannot compute errors");
      return false;
   }

   if (fMinimizer->CovMatrixStatus() != 3) {
      MATH_WARN_MSG("Fitter::ApplyWeightCorrection","Covariance matrix for weighted likelihood is not accurate, the errors may be not reliable");
      if (fMinimizer->CovMatrixStatus() == 2)
         MATH_WARN_MSG("Fitter::ApplyWeightCorrection","Covariance matrix for weighted likelihood was forced to be defined positive");
      if (fMinimizer->CovMatrixStatus() <= 0)
         // probably should have failed before
         MATH_ERROR_MSG("Fitter::ApplyWeightCorrection","Covariance matrix for weighted likelihood is not valid !");
   }

   // get Hessian matrix from weight-square likelihood
   std::vector<double> hes(n*n);
   ret = fMinimizer->GetHessianMatrix(&hes[0] );
   if (!ret) {
      MATH_ERROR_MSG("Fitter::ApplyWeightCorrection","Error retrieving Hesse on weight2 likelihood - cannot compute errors");
      return false;
   }


   // perform product of matrix cov * hes * cov
   // since we do not want to add matrix dependence do product by hand
   // first do  hes * cov
   std::vector<double> tmp(n*n);
   for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < n; ++j) {
         for (unsigned int k = 0; k < n; ++k)
            tmp[i*n+j] += hes[i*n + k] * cov[k*n + j];
      }
   }
   // do multiplication now cov * tmp save result
   std::vector<double> newCov(n*n);
   for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < n; ++j) {
         for (unsigned int k = 0; k < n; ++k)
            newCov[i*n+j] += cov[i*n + k] * tmp[k*n + j];
      }
   }
   // update fit result with new corrected covariance matrix
   unsigned int k = 0;
   for (unsigned int i = 0; i < n; ++i) {
      fResult->fErrors[i] = std::sqrt( newCov[i*(n+1)] );
      for (unsigned int j = 0; j <= i; ++j)
         fResult->fCovMatrix[k++] = newCov[i *n + j];
   }

   // restore previous used objective function
   fObjFunction.swap( objFunc );

   return true;
}



void Fitter::ExamineFCN()  {
   // return a pointer to the binned data used in the fit
   // works only for chi2 or binned likelihood fits
   // thus when the objective function stored is a Chi2Func or a PoissonLikelihood
   // This also set the model function correctly if it has not been set

   if ( GetDataFromFCN<BasicFCN<ROOT::Math::IMultiGenFunction, ROOT::Math::IParamMultiFunction, BinData> >() ) return;
   if ( GetDataFromFCN<BasicFCN<ROOT::Math::IMultiGenFunction, ROOT::Math::IParamMultiFunction, UnBinData> >() ) return;

   if ( GetDataFromFCN<BasicFCN<ROOT::Math::IMultiGradFunction, ROOT::Math::IParamMultiFunction, BinData> >() ) return;
   if ( GetDataFromFCN<BasicFCN<ROOT::Math::IMultiGradFunction, ROOT::Math::IParamMultiFunction, UnBinData> >() ) return;

   //MATH_INFO_MSG("Fitter::ExamineFCN","Objective function is not of a known type - FitData and ModelFunction objects are not available");
   return;
}

   } // end namespace Fit

} // end namespace ROOT
