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

#include "ROOT/EExecutionPolicy.hxx"
#include "Fit/BasicFCN.h"
#include "Fit/BinData.h"
#include "Fit/FitUtil.h"
#include "Math/IParamFunction.h"

#include <memory>
#include <vector>

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
template<class DerivFunType, class ModelFunType = ROOT::Math::IParamMultiFunction>
class PoissonLikelihoodFCN : public BasicFCN<DerivFunType,ModelFunType,BinData>  {

public:
   typedef typename ModelFunType::BackendType T;
   typedef  BasicFCN<DerivFunType,ModelFunType,BinData> BaseFCN;

   typedef  ::ROOT::Math::BasicFitMethodFunction<DerivFunType> BaseObjFunction;
   typedef typename  BaseObjFunction::BaseFunction BaseFunction;

   typedef ::ROOT::Math::IParamMultiFunctionTempl<T> IModelFunction;
   typedef typename BaseObjFunction::Type_t Type_t;

   /**
      Constructor from unbin data set and model function (pdf)
   */
   PoissonLikelihoodFCN (const std::shared_ptr<BinData> & data, const std::shared_ptr<IModelFunction> & func, int weight = 0, bool extended = true, const ::ROOT::EExecutionPolicy &executionPolicy = ::ROOT::EExecutionPolicy::kSequential ) :
      BaseFCN( data, func),
      fIsExtended(extended),
      fWeight(weight),
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func->NPar() ) ),
      fExecutionPolicy(executionPolicy)
   { }

   /**
      Constructor from unbin data set and model function (pdf) managed by the users
   */
   PoissonLikelihoodFCN (const BinData & data, const IModelFunction & func, int weight = 0, bool extended = true, const ::ROOT::EExecutionPolicy &executionPolicy = ::ROOT::EExecutionPolicy::kSequential ) :
      BaseFCN(std::make_shared<BinData>(data), std::shared_ptr<IModelFunction>(dynamic_cast<IModelFunction*>(func.Clone() ) ) ),
      fIsExtended(extended),
      fWeight(weight),
      fNEffPoints(0),
      fGrad ( std::vector<double> ( func.NPar() ) ),
      fExecutionPolicy(executionPolicy)
   { }


   /**
      Destructor (no operations)
   */
   virtual ~PoissonLikelihoodFCN () {}

   /**
      Copy constructor
   */
   PoissonLikelihoodFCN(const PoissonLikelihoodFCN & f) :
      BaseFCN(f.DataPtr(), f.ModelFunctionPtr() ),
      fIsExtended(f.fIsExtended ),
      fWeight( f.fWeight ),
      fNEffPoints( f.fNEffPoints ),
      fGrad( f.fGrad),
      fExecutionPolicy(f.fExecutionPolicy)
   {  }

   /**
      Assignment operator
   */
   PoissonLikelihoodFCN & operator = (const PoissonLikelihoodFCN & rhs) {
      SetData(rhs.DataPtr() );
      SetModelFunction(rhs.ModelFunctionPtr() );
      fNEffPoints = rhs.fNEffPoints;
      fGrad = rhs.fGrad;
      fIsExtended = rhs.fIsExtended;
      fWeight = rhs.fWeight;
      fExecutionPolicy = rhs.fExecutionPolicy;
   }


   /// clone the function (need to return Base for Windows)
   BaseFunction * Clone() const override { return new  PoissonLikelihoodFCN(*this); }

   // effective points used in the fit
   virtual unsigned int NFitPoints() const { return fNEffPoints; }

   /// i-th likelihood element and its gradient
   double DataElement(const double * x, unsigned int i, double * g, double * h, bool fullHessian) const override {
      if (i==0) this->UpdateNCalls();
      return FitUtil::Evaluate<T>::EvalPoissonBinPdf(BaseFCN::ModelFunction(), BaseFCN::Data(), x, i, g, h, BaseFCN::IsAGradFCN(), fullHessian);
   }

   /// evaluate gradient
   void Gradient(const double *x, double *g) const override
   {
      // evaluate the Poisson gradient
      FitUtil::Evaluate<typename BaseFCN::T>::EvalPoissonLogLGradient(BaseFCN::ModelFunction(), BaseFCN::Data(), x, g,
                                                                      fNEffPoints, fExecutionPolicy);
   }

   /**
    * Computes the full Hessian. Return false if Hessian is not supported
    */
   // virtual bool Hessian(const double * x, double * hess) const {
   //    //return full Hessian
   //    unsigned int np = NPoints();
   //    unsigned int ndim = NDim();
   //    unsigned int nh = ndim*(ndim+1)/2;
   //    for (unsigned int k = 0; k < nh;  ++k) {
   //       hess[k] = 0;
   //    }
   //    std::vector<double> g(np);  // gradient of the residual (y-f)/sigma
   //    std::vector<double> h(nh);  // hessian of residual
   //    for (unsigned int i = 0; i < np; i++) {
   //       double f = DataElement(x,i,g,h,true);
   //       if (f == std::numeric_limits<double>::quiet_NaN() ) return false;
   //       for (unsigned int j = 0; j < np; j++) {
   //          for (unsigned int k = 0; k <=j; k++) {
   //             unsigned int index = j + k * (k + 1) / 2;
   //             h[index] += 2. * ( g[j]*g[k] + f * h[index]*h[index] );
   //          }
   //       }
   //    }
   //    return true;
   // }

   /// get type of fit method function
   typename BaseObjFunction::Type_t Type() const override { return BaseObjFunction::kPoissonLikelihood; }

   bool IsWeighted() const { return (fWeight != 0); }

   // Use the weights in evaluating the likelihood
   void UseSumOfWeights() {
      if (fWeight == 0) return; // do nothing if it was not weighted
      fWeight = 1;
   }

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
   double DoEval(const double * x) const override {
      this->UpdateNCalls();
      return FitUtil::Evaluate<T>::EvalPoissonLogL(BaseFCN::ModelFunction(), BaseFCN::Data(), x, fWeight, fIsExtended,
                                                   fNEffPoints, fExecutionPolicy);
   }

   // for derivatives
   double DoDerivative(const double * x, unsigned int icoord) const override {
      Gradient(x, &fGrad[0]);
      return fGrad[icoord];
   }


      //data member

   bool fIsExtended; ///< flag to indicate if is extended (when false is a Multinomial likelihood), default is true
   int fWeight;  ///< flag to indicate if needs to evaluate using weight or weight squared (default weight = 0)

   mutable unsigned int fNEffPoints;  ///< number of effective points used in the fit

   mutable std::vector<double> fGrad; ///< for derivatives

   ::ROOT::EExecutionPolicy fExecutionPolicy; ///< Execution policy
};

      // define useful typedef's
      typedef PoissonLikelihoodFCN<ROOT::Math::IMultiGenFunction, ROOT::Math::IParamMultiFunction> PoissonLLFunction;
      typedef PoissonLikelihoodFCN<ROOT::Math::IMultiGradFunction, ROOT::Math::IParamMultiFunction> PoissonLLGradFunction;


   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_PoissonLikelihoodFCN */
