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

#ifndef ROOT_Fit_BasicFCN
#include "Fit/BasicFCN.h"
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

#include <memory>

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
class LogLikelihoodFCN : public BasicFCN<FunType,UnBinData>  {

public:

   typedef  BasicFCN<FunType,UnBinData> BaseFCN; 

   typedef  ::ROOT::Math::BasicFitMethodFunction<FunType> BaseObjFunction;
   typedef typename  BaseObjFunction::BaseFunction BaseFunction;

   typedef  ::ROOT::Math::IParamMultiFunction IModelFunction;


   /**
      Constructor from unbin data set and model function (pdf)
   */
   LogLikelihoodFCN (const std::shared_ptr<UnBinData> & data, const std::shared_ptr<IModelFunction> & func, int weight = 0, bool extended = false) :
      BaseFCN( data, func),
      fIsExtended(extended),
      fWeight(weight),
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func->NPar() ) )
   {}

      /**
      Constructor from unbin data set and model function (pdf) for object managed by users
   */
   LogLikelihoodFCN (const UnBinData & data, const IModelFunction & func, int weight = 0, bool extended = false) :
      BaseFCN(std::shared_ptr<UnBinData>(const_cast<UnBinData*>(&data), DummyDeleter<UnBinData>()), std::shared_ptr<IModelFunction>(dynamic_cast<IModelFunction*>(func.Clone() ) ) ),
      fIsExtended(extended),
      fWeight(weight),
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func.NPar() ) )
   {}

   /**
      Destructor (no operations)
   */
   virtual ~LogLikelihoodFCN () {}

   /**
      Copy constructor 
   */
   LogLikelihoodFCN(const LogLikelihoodFCN & f) :
      BaseFCN(f.DataPtr(), f.ModelFunctionPtr() ),
      fIsExtended(f.fIsExtended ),
      fWeight( f.fWeight ),
      fNEffPoints( f.fNEffPoints ),
      fGrad( f.fGrad)
   {  }


   /**
      Assignment operator 
   */
   LogLikelihoodFCN & operator = (const LogLikelihoodFCN & rhs) {
      SetData(rhs.DataPtr() );
      SetModelFunction(rhs.ModelFunctionPtr() );
      fNEffPoints = rhs.fNEffPoints;
      fGrad = rhs.fGrad; 
      fIsExtended = rhs.fIsExtended;
      fWeight = rhs.fWeight; 
   }


   /// clone the function (need to return Base for Windows)
   virtual BaseFunction * Clone() const { return  new LogLikelihoodFCN(*this); }


   //using BaseObjFunction::operator();

   // effective points used in the fit
   virtual unsigned int NFitPoints() const { return fNEffPoints; }

   /// i-th likelihood contribution and its gradient
   virtual double DataElement(const double * x, unsigned int i, double * g) const {
      if (i==0) this->UpdateNCalls();
      return FitUtil::EvaluatePdf(BaseFCN::ModelFunction(), BaseFCN::Data(), x, i, g);
   }


   // need to be virtual to be instantited
   virtual void Gradient(const double *x, double *g) const {
      // evaluate the chi2 gradient
      FitUtil::EvaluateLogLGradient(BaseFCN::ModelFunction(), BaseFCN::Data(), x, g, fNEffPoints);
   }

   /// get type of fit method function
   virtual  typename BaseObjFunction::Type_t Type() const { return BaseObjFunction::kLogLikelihood; }


   // Use sum of the weight squared in evaluating the likelihood
   // (this is needed for calculating the errors)
   void UseSumOfWeightSquare(bool on = true) {
      if (fWeight == 0) return; // do nothing if it was not weighted
      if (on) fWeight = 2;
      else fWeight = 1;
   }



protected:


private:

   /**
      Evaluation of the  function (required by interface)
    */
   virtual double DoEval (const double * x) const {
      this->UpdateNCalls();

#ifdef ROOT_FIT_PARALLEL
      return FitUtilParallel::EvaluateLogL(BaseFCN::ModelFunction(), BaseFCN::Data(), x, fNEffPoints);
#else
      return FitUtil::EvaluateLogL(BaseFCN::ModelFunction(), BaseFCN::Data(), x, fWeight, fIsExtended, fNEffPoints);
#endif
   }

   // for derivatives
   virtual double  DoDerivative(const double * x, unsigned int icoord ) const {
      Gradient(x, &fGrad[0]);
      return fGrad[icoord];
   }


      //data member
   bool fIsExtended;  // flag for indicating if likelihood is extended
   int  fWeight;  // flag to indicate if needs to evaluate using weight or weight squared (default weight = 0)


   mutable unsigned int fNEffPoints;  // number of effective points used in the fit

   mutable std::vector<double> fGrad; // for derivatives


};

      // define useful typedef's
      typedef LogLikelihoodFCN<ROOT::Math::IMultiGenFunction>  LogLikelihoodFunction;
      typedef LogLikelihoodFCN<ROOT::Math::IMultiGradFunction> LogLikelihoodGradFunction;

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_LogLikelihoodFCN */
