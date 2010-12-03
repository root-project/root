// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 17 14:29:24 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class LogLikelihoodFCN

#ifndef ROOT_Fit_LogLikelihoodFCN
#define ROOT_Fit_LogLikelihoodFCN

#ifndef ROOT_Math_FitMethodFunction
#include "Math/FitMethodFunction.h"
#endif

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

#ifndef ROOT_Fit_UnBinData
#include "Fit/UnBinData.h"
#endif

#ifndef ROOT_Fit_FitUtil
#include "Fit/FitUtil.h"
#endif

#ifdef ROOT_FIT_PARALLEL
#ifndef ROOT_Fit_FitUtilParallel
#include "Fit/FitUtilParallel.h"
#endif
#endif

namespace ROOT { 

   namespace Fit { 


//___________________________________________________________________________________
/** 
   LogLikelihoodFCN class 
   for likelihood fits 

   it is template to distinguish gradient and non-gradient case

   @ingroup  FitMethodFunc   
*/ 
template<class FunType> 
class LogLikelihoodFCN : public ::ROOT::Math::BasicFitMethodFunction<FunType>  {

public: 



   typedef  ::ROOT::Math::BasicFitMethodFunction<FunType> BaseObjFunction; 
   typedef typename  BaseObjFunction::BaseFunction BaseFunction; 

   typedef  ::ROOT::Math::IParamMultiFunction IModelFunction;


   /** 
      Constructor from unbin data set and model function (pdf)
   */ 
   LogLikelihoodFCN (const UnBinData & data, const IModelFunction & func) : 
      BaseObjFunction(func.NPar(), data.Size() ),
      fData(data), 
      fFunc(func), 
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func.NPar() ) )
   {}
  

   /** 
      Destructor (no operations)
   */ 
   virtual ~LogLikelihoodFCN () {}

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Dummy Copy constructor (private)
   */ 
   LogLikelihoodFCN(const LogLikelihoodFCN &);

   /** 
      Dummy Assignment operator (private)
   */ 
   LogLikelihoodFCN & operator = (const LogLikelihoodFCN &);

public: 

   /// clone the function (need to return Base for Windows)
   virtual BaseFunction * Clone() const { return  new LogLikelihoodFCN(fData,fFunc); }


   //using BaseObjFunction::operator();

   // effective points used in the fit
   virtual unsigned int NFitPoints() const { return fNEffPoints; }

   /// i-th likelihood contribution and its gradient
   virtual double DataElement(const double * x, unsigned int i, double * g) const { 
      if (i==0) this->UpdateNCalls();
      return FitUtil::EvaluatePdf(fFunc, fData, x, i, g); 
   }


   // need to be virtual to be instantited
   virtual void Gradient(const double *x, double *g) const { 
      // evaluate the chi2 gradient
      FitUtil::EvaluateLogLGradient(fFunc, fData, x, g, fNEffPoints);
   }

   /// get type of fit method function
   virtual  typename BaseObjFunction::Type_t Type() const { return BaseObjFunction::kLogLikelihood; }

   /// access to const reference to the data 
   virtual const UnBinData & Data() const { return fData; }

   /// access to const reference to the model function
   virtual const IModelFunction & ModelFunction() const { return fFunc; }


protected: 


private:

   /**
      Evaluation of the  function (required by interface)
    */
   virtual double DoEval (const double * x) const { 
      this->UpdateNCalls();

#ifdef ROOT_FIT_PARALLEL
      return FitUtilParallel::EvaluateLogL(fFunc, fData, x, fNEffPoints); 
#else 
      return FitUtil::EvaluateLogL(fFunc, fData, x, fNEffPoints); 
#endif
   } 

   // for derivatives 
   virtual double  DoDerivative(const double * x, unsigned int icoord ) const { 
      Gradient(x, &fGrad[0]); 
      return fGrad[icoord]; 
   }

 
      //data member

   const UnBinData & fData; 
   const IModelFunction & fFunc; 

   mutable unsigned int fNEffPoints;  // number of effective points used in the fit 

   mutable std::vector<double> fGrad; // for derivatives


}; 

      // define useful typedef's
      typedef LogLikelihoodFCN<ROOT::Math::IMultiGenFunction>  LogLikelihoodFunction; 
      typedef LogLikelihoodFCN<ROOT::Math::IMultiGradFunction> LogLikelihoodGradFunction; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_LogLikelihoodFCN */
