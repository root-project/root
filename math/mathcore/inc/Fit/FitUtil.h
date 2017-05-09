// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitUtil

#ifndef ROOT_Fit_FitUtil
#define ROOT_Fit_FitUtil

#include "Math/IParamFunctionfwd.h"

#include "ROOT/TThreadExecutor.hxx"
#include "ROOT/TProcessExecutor.hxx"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"

#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"

#include "TError.h"

namespace ROOT {

   namespace Fit {

/**
   namespace defining utility free functions using in Fit for evaluating the various fit method
   functions (chi2, likelihood, etc..)  given the data and the model function

   @ingroup FitMain
*/
namespace FitUtil {

  typedef  ROOT::Math::IParamMultiFunction IModelFunction;
  typedef  ROOT::Math::IParamMultiGradFunction IGradModelFunction;

  template<class T>
  using IModelFunctionTempl = ROOT::Math::IParamMultiFunctionTempl<T>;

   /** Chi2 Functions */

   /**
       evaluate the Chi2 given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */
   double EvaluateChi2(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints, const unsigned int &executionPolicy, unsigned nChunks=1 );

   /**
       evaluate the effective Chi2 given a model function and the data at the point x.
       The effective chi2 uses the errors on the coordinates : W = 1/(sigma_y**2 + ( sigma_x_i * df/dx_i )**2 )
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */
   double EvaluateChi2Effective(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);

   /**
       evaluate the Chi2 gradient given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the Chi2 evaluation
   */
   void EvaluateChi2Gradient(const IModelFunction & func, const BinData & data, const double * x, double * grad, unsigned int & nPoints);

   /**
       evaluate the LogL given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the LogL evaluation
   */
   double EvaluateLogL(const IModelFunction & func, const UnBinData & data, const double * x, int iWeight, bool extended, unsigned int & nPoints);

   /**
       evaluate the LogL gradient given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the LogL evaluation
   */
   void EvaluateLogLGradient(const IModelFunction & func, const UnBinData & data, const double * x, double * grad, unsigned int & nPoints);

   /**
       evaluate the Poisson LogL given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the LogL evaluation
       By default is extended, pass extedend to false if want to be not extended (MultiNomial)
   */
   double EvaluatePoissonLogL(const IModelFunction & func, const BinData & data, const double * x, int iWeight, bool extended, unsigned int & nPoints);

   /**
       evaluate the Poisson LogL given a model function and the data at the point x.
       return also nPoints as the effective number of used points in the LogL evaluation
   */
   void EvaluatePoissonLogLGradient(const IModelFunction & func, const BinData & data, const double * x, double * grad);

//    /**
//        Parallel evaluate the Chi2 given a model function and the data at the point x.
//        return also nPoints as the effective number of used points in the Chi2 evaluation
//    */
//    double ParallelEvalChi2(const IModelFunction & func, const BinData & data, const double * x, unsigned int & nPoints);

   // methods required by dedicate minimizer like Fumili

   /**
       evaluate the residual contribution to the Chi2 given a model function and the BinPoint data
       and if the pointer g is not null evaluate also the gradient of the residual.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation
       is used
   */
   double EvaluateChi2Residual(const IModelFunction & func, const BinData & data, const double * x, unsigned int ipoint, double *g = 0);

   /**
       evaluate the pdf contribution to the LogL given a model function and the BinPoint data.
       If the pointer g is not null evaluate also the gradient of the pdf.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation
       is used
   */
   double EvaluatePdf(const IModelFunction & func, const UnBinData & data, const double * x, unsigned int ipoint, double * g = 0);

   /**
       evaluate the pdf contribution to the Poisson LogL given a model function and the BinPoint data.
       If the pointer g is not null evaluate also the gradient of the Poisson pdf.
       If the function provides parameter derivatives they are used otherwise a simple derivative calculation
       is used
   */
   double EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * x, unsigned int ipoint, double * g = 0);

  template<class T>
  struct EvalChi2{
    static double DoEval(const IModelFunctionTempl<T> & func, const BinData & data, const double * p, unsigned int & nPoints, const unsigned int &executionPolicy, unsigned nChunks = 0){
      // evaluate the chi2 given a  vectorized function reference  , the data and returns the value and also in nPoints
      // the actual number of used points
      // normal chi2 using only error on values (from fitting histogram)
      // optionally the integral of function in the bin is used

      unsigned int n = data.Size();

      nPoints = 0; // count the effective non-zero points
      // set parameters of the function to cache integral value
      #ifdef USE_PARAMCACHE
        (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);
      #endif
      // do not cache parameter values (it is not thread safe)
      //func.SetParameters(p);


      // get fit option and check case if using integral of bins
      const DataOptions & fitOpt = data.Opt();
      if (fitOpt.fExpErrors || fitOpt.fIntegral || fitOpt.fExpErrors)
        Error("FitUtil::EvaluateChi2","The vectorized implementation doesn't support Integrals, BinVolume or ExpErrors\n. Aborting operation.");

      (const_cast<IModelFunctionTempl<T> &>(func)).SetParameters(p);

      double maxResValue = std::numeric_limits<double>::max() /n;
      std::vector<double> ones{1,1,1,1};
      auto vecSize = vecCore::VectorSize<T>();

      auto mapFunction = [&](unsigned int i){
          // in case of no error in y invError=1 is returned
          T x, y, invErrorVec;
          vecCore::Load<T>(x, data.GetCoordComponent(i*vecSize,0));
          vecCore::Load<T>(y, data.ValuePtr(i*vecSize));
          const auto invError = data.ErrorPtr(i*vecSize);
          auto invErrorptr = (invError != nullptr) ? invError : &ones.front();
          vecCore::Load<T>(invErrorVec, invErrorptr);

          T fval{};

    #ifdef USE_PARAMCACHE
          fval = func ( &x );
    #else
          fval = func ( &x, p );
    #endif
          nPoints++;

          T tmp = ( y - fval ) * invErrorVec;
          T chi2 = tmp * tmp;


          // avoid inifinity or nan in chi2 values due to wrong function values
          auto m = vecCore::Mask_v<T>(chi2 > maxResValue);

          vecCore::MaskedAssign<T>(chi2, m, maxResValue);

          return chi2;
      };

      auto redFunction = [](const std::vector<T> & objs){
                          return std::accumulate(objs.begin(), objs.end(), T{});
      };

      T res{};
      if(executionPolicy == 0){
        for (unsigned int i=0; i<(data.Size()/vecSize); i++) {
          res += mapFunction(i);
        }
      } else if(executionPolicy == 1) {
        ROOT::TThreadExecutor pool;
        res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size()/vecSize), redFunction, nChunks);
      // } else if(executionPolicy == 2){
      //   ROOT::TProcessExecutor pool;
      //   res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size()/vecSize), redFunction);
      } else{
        Error("FitUtil::EvaluateChi2","Execution policy unknown. Avalaible choices:\n 0: Serial (default)\n 1: MultiThread\n 2: MultiProcess");
      }
      nPoints=n;

    #ifdef DEBUG
      std::cout << "chi2 = " << chi2 << " n = " << nPoints  /*<< " rejected = " << nRejected */ << std::endl;
    #endif

      return res.sum();
    }
  };

  template<>
  struct EvalChi2<double>{
    static double DoEval(const IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints, const unsigned int &executionPolicy,unsigned nChunks = 1) {
      // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
      // the actual number of used points
      // normal chi2 using only error on values (from fitting histogram)
      // optionally the integral of function in the bin is used
      return FitUtil::EvaluateChi2(func, data, p, nPoints, executionPolicy, nChunks);
    }
  };


} // end namespace FitUtil

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitUtil */
