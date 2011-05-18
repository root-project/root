// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 17 14:29:24 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class PoissonLikelihoodFCN

#ifndef ROOT_Fit_PoissonLikelihoodFCN
#define ROOT_Fit_PoissonLikelihoodFCN

#ifndef ROOT_Math_FitMethodunction
#include "Math/FitMethodFunction.h"
#endif

#ifndef ROOT_Math_IParamFunction
#include "Math/IParamFunction.h"
#endif

#ifndef ROOT_Fit_BinData
#include "Fit/BinData.h"
#endif

#ifndef ROOT_Fit_FitUtil
#include "Fit/FitUtil.h"
#endif

//#define PARALLEL
// #ifdef PARALLEL
// #ifndef ROOT_Fit_FitUtilParallel
// #include "Fit/FitUtilParallel.h"
// #endif
// #endif

namespace ROOT {

   namespace Fit {


//___________________________________________________________________________________
/**
   class evaluating the log likelihood
   for binned Poisson likelihood fits
   it is template to distinguish gradient and non-gradient case

   @ingroup  FitMethodFunc
*/
template<class FunType>
class PoissonLikelihoodFCN : public ::ROOT::Math::BasicFitMethodFunction<FunType>  {

public:


   typedef  ::ROOT::Math::BasicFitMethodFunction<FunType> BaseObjFunction;
   typedef typename  BaseObjFunction::BaseFunction BaseFunction;

   typedef  ::ROOT::Math::IParamMultiFunction IModelFunction;


   /**
      Constructor from unbin data set and model function (pdf)
   */
   PoissonLikelihoodFCN (const BinData & data, const IModelFunction & func) :
      BaseObjFunction(func.NPar(), data.Size() ),
      fData(data),
      fFunc(func),
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func.NPar() ) )
   { }


   /**
      Destructor (no operations)
   */
   ~PoissonLikelihoodFCN () {}

private:
   // usually copying is non trivial, so we declare but don't implement them

   /**
      Copy constructor
   */
   PoissonLikelihoodFCN(const PoissonLikelihoodFCN &);

   /**
      Assignment operator
   */
   PoissonLikelihoodFCN & operator = (const PoissonLikelihoodFCN &);

public:

   /// clone the function (need to return Base for Windows)
   virtual BaseFunction * Clone() const { return new  PoissonLikelihoodFCN(fData,fFunc); }

   // effective points used in the fit
   virtual unsigned int NFitPoints() const { return fNEffPoints; }

   /// i-th likelihood element and its gradient
   virtual double DataElement(const double * x, unsigned int i, double * g) const {
      if (i==0) this->UpdateNCalls();
      return FitUtil::EvaluatePoissonBinPdf(fFunc, fData, x, i, g);
   }

   /// evaluate gradient
   virtual void Gradient(const double *x, double *g) const {
      // evaluate the chi2 gradient
      FitUtil::EvaluatePoissonLogLGradient(fFunc, fData, x, g );
   }

   /// get type of fit method function
   virtual  typename BaseObjFunction::Type_t Type() const { return BaseObjFunction::kLogLikelihood; }

   /// access to const reference to the data
   virtual const BinData & Data() const { return fData; }

   /// access to const reference to the model function
   virtual const IModelFunction & ModelFunction() const { return fFunc; }


protected:


private:

   /**
      Evaluation of the  function (required by interface)
    */
   virtual double DoEval (const double * x) const {
      this->UpdateNCalls();
      return FitUtil::EvaluatePoissonLogL(fFunc, fData, x, fNEffPoints);
   }

   // for derivatives
   virtual double  DoDerivative(const double * x, unsigned int icoord ) const {
      Gradient(x, &fGrad[0]);
      return fGrad[icoord];
   }


      //data member

   const BinData & fData;
   const IModelFunction & fFunc;

   mutable unsigned int fNEffPoints;  // number of effective points used in the fit

   mutable std::vector<double> fGrad; // for derivatives

};

      // define useful typedef's
      typedef PoissonLikelihoodFCN<ROOT::Math::IMultiGenFunction> PoissonLLFunction;
      typedef PoissonLikelihoodFCN<ROOT::Math::IMultiGradFunction> PoissonLLGradFunction;


   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_PoissonLikelihoodFCN */
