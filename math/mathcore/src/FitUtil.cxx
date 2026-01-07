// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitUtil

#include "Fit/FitUtil.h"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"

#include "Math/IFunctionfwd.h"
#include "Math/IParamFunction.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/WrappedFunction.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Math/RichardsonDerivator.h"

#include "Math/Error.h"
#include "Math/Util.h"  // for safe log(x)

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "ROOT/TSequentialExecutor.hxx"

#include <limits>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

#include "TROOT.h"

//#define DEBUG
#ifdef DEBUG
#define NSAMPLE 10
#include <iostream>
#endif

//  need to implement integral option

namespace ROOT {

   namespace Fit {

      namespace FitUtil {

         // derivative with respect of the parameter to be integrated
         template<class GradFunc = IGradModelFunction>
         struct ParamDerivFunc {
            ParamDerivFunc(const GradFunc & f) : fFunc(f), fIpar(0) {}
            void SetDerivComponent(unsigned int ipar) { fIpar = ipar; }
            double operator() (const double *x, const double *p) const {
               return fFunc.ParameterDerivative( x, p, fIpar );
            }
            unsigned int NDim() const { return fFunc.NDim(); }
            const GradFunc & fFunc;
            unsigned int fIpar;
         };

// simple gradient calculator using the 2 points rule

         class SimpleGradientCalculator {

         public:
            // construct from function and gradient dimension gdim
            // gdim = npar for parameter gradient
            // gdim = ndim for coordinate gradients
            // construct (the param values will be passed later)
            // one can choose between 2 points rule (1 extra evaluation) istrat=1
            // or two point rule (2 extra evaluation)
            // (found 2 points rule does not work correctly - minuit2FitBench fails)
            SimpleGradientCalculator(int gdim, const IModelFunction & func,double eps = 2.E-8, int istrat = 1) :
               fEps(eps),
               fPrecision(1.E-8 ), // sqrt(epsilon)
               fStrategy(istrat),
               fN(gdim ),
               fFunc(func),
               fVec(std::vector<double>(gdim) ) // this can be probably optimized
            {}

            // internal method to calculate single partial derivative
            // assume cached vector fVec is already set
            double DoParameterDerivative(const double *x, const double *p, double f0, int k) const {
               double p0 = p[k];
               double h = std::max( fEps* std::abs(p0), 8.0*fPrecision*(std::abs(p0) + fPrecision) );
               fVec[k] += h;
               double deriv = 0;
               // t.b.d : treat case of infinities
               //if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() )
               double f1 = fFunc(x, &fVec.front() );
               if (fStrategy > 1) {
                  fVec[k] = p0 - h;
                  double f2 = fFunc(x, &fVec.front() );
                  deriv = 0.5 * ( f2 - f1 )/h;
               }
               else
                  deriv = ( f1 - f0 )/h;

               fVec[k] = p[k]; // restore original p value
               return deriv;
            }
            // number of dimension in x (needed when calculating the integrals)
            unsigned int NDim() const {
               return fFunc.NDim();
            }
            // number of parameters (needed for grad ccalculation)
            unsigned int NPar() const {
               return fFunc.NPar();
            }

            double ParameterDerivative(const double *x, const double *p, int ipar) const {
               // fVec are the cached parameter values
               std::copy(p, p+fN, fVec.begin());
               double f0 = fFunc(x, p);
               return DoParameterDerivative(x,p,f0,ipar);
            }

            // calculate all gradient at point (x,p) knnowing already value f0 (we gain a function eval.)
            void ParameterGradient(const double * x, const double * p, double f0, double * g) {
               // fVec are the cached parameter values
               std::copy(p, p+fN, fVec.begin());
               for (unsigned int k = 0; k < fN; ++k) {
                  g[k] = DoParameterDerivative(x,p,f0,k);
               }
            }

            // calculate gradient w.r coordinate values
            void Gradient(const double * x, const double * p, double f0, double * g) {
               // fVec are the cached coordinate values
               std::copy(x, x+fN, fVec.begin());
               for (unsigned int k = 0; k < fN; ++k) {
                  double x0 = x[k];
                  double h = std::max( fEps* std::abs(x0), 8.0*fPrecision*(std::abs(x0) + fPrecision) );
                  fVec[k] += h;
                  // t.b.d : treat case of infinities
                  //if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() )
                  double f1 = fFunc( &fVec.front(), p );
                  if (fStrategy > 1) {
                     fVec[k] = x0 - h;
                     double f2 = fFunc( &fVec.front(), p  );
                     g[k] = 0.5 * ( f2 - f1 )/h;
                  }
                  else
                     g[k] = ( f1 - f0 )/h;

                  fVec[k] = x[k]; // restore original x value
               }
            }

         private:

            double fEps;
            double fPrecision;
            int fStrategy; // strategy in calculation ( =1 use 2 point rule( 1 extra func) , = 2 use r point rule)
            unsigned int fN; // gradient dimension
            const IModelFunction & fFunc;
            mutable std::vector<double> fVec; // cached coordinates (or parameter values in case of gradientpar)
         };


         // function to avoid infinities or nan
         double CorrectValue(double rval) {
            // avoid infinities or nan in  rval
            if (rval > - std::numeric_limits<double>::max() && rval < std::numeric_limits<double>::max() )
               return rval;
            else if (rval < 0)
               // case -inf
               return -std::numeric_limits<double>::max();
            else
               // case + inf or nan
               return  + std::numeric_limits<double>::max();
         }

         // Check if the value is a finite number. The argument rval is updated if it is infinite or NaN,
         // setting it to the maximum finite value (preserving the sign).
         bool CheckInfNaNValue(double &rval)
         {
            if (rval > - std::numeric_limits<double>::max() && rval < std::numeric_limits<double>::max() )
               return true;
            else if (rval < 0) {
               // case -inf
               rval =  -std::numeric_limits<double>::max();
               return false;
            }
            else {
               // case + inf or nan
               rval =  + std::numeric_limits<double>::max();
               return false;
            }
         }


         // calculation of the integral of the gradient functions
         // for a function providing derivative w.r.t parameters
         // x1 and x2 defines the integration interval , p the parameters
         template <class GFunc>
         void CalculateGradientIntegral(const GFunc & gfunc,
                                        const double *x1, const double * x2, const double * p, double *g) {

            // needs to calculate the integral for each partial derivative
            ParamDerivFunc<GFunc> pfunc( gfunc);
            IntegralEvaluator<ParamDerivFunc<GFunc> > igDerEval( pfunc, p, true);
            // loop on the parameters
            unsigned int npar = gfunc.NPar();
            for (unsigned int k = 0; k < npar; ++k ) {
               pfunc.SetDerivComponent(k);
               g[k] = igDerEval( x1, x2 );
            }
         }



      } // end namespace  FitUtil



//___________________________________________________________________________________________________________________________
// for chi2 functions
//___________________________________________________________________________________________________________________________

double FitUtil::EvaluateChi2(const IModelFunction &func, const BinData &data, const double *p, unsigned int &nPoints,
                              ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
         // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
         // the actual number of used points
         // normal chi2 using only error on values (from fitting histogram)
         // optionally the integral of function in the bin is used

   unsigned int n = data.Size();

   // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunction &>(func)).SetParameters(p);
#endif
   // do not cache parameter values (it is not thread safe)
   //func.SetParameters(p);


   // get fit option and check case if using integral of bins
   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useExpErrors = (fitOpt.fExpErrors);
   bool isWeighted = fitOpt.fExpErrors && !fitOpt.fErrors1 && data.IsWeighted();  //used in case of Person weighted chi2 fits
#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function " << &func << "  " << p << std::endl;
   std::cout << "use empty bins  " << fitOpt.fUseEmpty << std::endl;
   std::cout << "use integral    " << useBinIntegral << std::endl;
   std::cout << "use binvolume   " << useBinVolume << std::endl;
   std::cout << "use Exp Errors  " << useExpErrors << std::endl;
   std::cout << "use all error=1 " << fitOpt.fErrors1 << std::endl;
   if (isWeighted)   std::cout << "Weighted data set - sumw =  " << data.SumOfContent() << "  sumw2 = " << data.SumOfError2() << std::endl;
#endif

   ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT;
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      // do not use GSL integrator which is not thread safe
      igType = ROOT::Math::IntegrationOneDim::kGAUSS;
   }
#ifdef USE_PARAMCACHE
   IntegralEvaluator<> igEval( func, nullptr, useBinIntegral, igType);
#else
   IntegralEvaluator<> igEval( func, p, useBinIntegral, igType);
#endif
   double maxResValue = std::numeric_limits<double>::max() /n;
   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume) wrefVolume /= data.RefVolume();
   }

   (const_cast<IModelFunction &>(func)).SetParameters(p);

   auto mapFunction = [&](const unsigned i){

      double chi2{};
      double fval{};

      const auto x1 = data.GetCoordComponent(i, 0);
      const auto y = data.Value(i);
      auto invError = data.InvError(i);

      //invError = (invError!= 0.0) ? 1.0/invError :1;

      const double * x = nullptr;
      std::vector<double> xc;
      double binVolume = 1.0;
      if (useBinVolume) {
         unsigned int ndim = data.NDim();
         xc.resize(data.NDim());
         for (unsigned int j = 0; j < ndim; ++j) {
            double xx = *data.GetCoordComponent(i, j);
            double x2 = data.GetBinUpEdgeComponent(i, j);
            binVolume *= std::abs(x2 - xx);
            xc[j] = (useBinIntegral) ? xx : 0.5*(x2 + xx);
         }
         x = xc.data();
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if(data.NDim() > 1) {
         // multi-dim case (no bin volume)
         // in case of bin integral xc is x1
         xc.resize(data.NDim());
         xc[0] = *x1;
         for (unsigned int j = 1; j < data.NDim(); ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
            // for dim 1
            x = x1;
      }


      if (!useBinIntegral) {
#ifdef USE_PARAMCACHE
         fval = func ( x );
#else
         fval = func ( x, p );
#endif
      }
      else {
         // calculate integral normalized (divided) by bin volume
         // need to set function and parameters here in case loop is parallelized
         std::vector<double> x2(data.NDim());
         data.GetBinUpEdgeCoordinates(i, x2.data());
         fval = igEval(x, x2.data());
      }
      // normalize result if requested according to bin volume
      // we need to multiply by the bin volume (e.g. for variable bins histograms)
      if (useBinVolume) fval *= binVolume;

      // expected errors
      if (useExpErrors) {
         double invWeight  = 1.0;
         // case of weighted Pearson chi2 fit
         if (isWeighted) {
            // in case of requested a weighted Pearson fit (option "PW") a weight factor needs to be applied
            // the bin inverse weight is estimated from bin error and bin content
            if (y != 0)
               invWeight = y * invError * invError;
            else
               // when y is 0 we use a global weight estimated form all histogram (correct if scaling the histogram)
               // note that if the data is weighted data.SumOfError2 will not be equal to zero
               invWeight = data.SumOfContent()/ data.SumOfError2();
         }
         // compute expected error  as f(x) or f(x) / weight (if weighted fit)
         double invError2 = (fval > 0) ? invWeight / fval : 0.0;
         invError = std::sqrt(invError2);
         //std::cout << "using Pearson chi2 " << x[0] << "  " << 1./invError2 << "  " << fval << std::endl;
      }

#ifdef DEBUG
      std::cout << x[0] << "  " << y << "  " << 1./invError << " params : ";
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << " bin volume " << binVolume << " ref " << wrefVolume << std::endl;
#endif

      if (invError > 0) {

         double tmp = ( y -fval )* invError;
         double resval = tmp * tmp;


         // avoid infinity or nan in chi2 values due to wrong function values
         if ( resval < maxResValue )
            chi2 += resval;
         else {
            //nRejected++;
            chi2 += maxResValue;
         }
      }
      return chi2;
  };

#ifdef R__USE_IMT
  auto redFunction = [](const std::vector<double> & objs){
                          return std::accumulate(objs.begin(), objs.end(), double{});
  };
#else
  (void)nChunks;

  // If IMT is disabled, force the execution policy to the serial case
  if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
     Warning("FitUtil::EvaluateChi2", "Multithread execution policy requires IMT, which is disabled. Changing "
                                      "to ROOT::EExecutionPolicy::kSequential.");
     executionPolicy = ROOT::EExecutionPolicy::kSequential;
  }
#endif

  double res{};
  if(executionPolicy == ROOT::EExecutionPolicy::kSequential){
    for (unsigned int i=0; i<n; ++i) {
      res += mapFunction(i);
    }
#ifdef R__USE_IMT
  } else if(executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
    ROOT::TThreadExecutor pool;
    auto chunks = nChunks !=0? nChunks: setAutomaticChunking(data.Size());
    res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction, chunks);
#endif
//   } else if(executionPolicy == ROOT::Fit::kMultitProcess){
    // ROOT::TProcessExecutor pool;
    // res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
  } else{
    Error("FitUtil::EvaluateChi2","Execution policy unknown. Available choices:\n ROOT::EExecutionPolicy::kSequential (default)\n ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
  }

  // reset the number of fitting data points
  nPoints = n;  // no points are rejected
  //if (nRejected != 0)  nPoints = n - nRejected;

   return res;
}


//___________________________________________________________________________________________________________________________

double FitUtil::EvaluateChi2Effective(const IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints) {
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
   // the actual number of used points
   // method using the error in the coordinates
   // integral of bin does not make sense in this case

   unsigned int n = data.Size();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate effective chi2 using function " << &func << "  " << p << std::endl;
#endif

   assert(data.HaveCoordErrors()  || data.HaveAsymErrors());

   double chi2 = 0;
   //int nRejected = 0;


   //func.SetParameters(p);

   unsigned int ndim = func.NDim();

   // use Richardson derivator
   ROOT::Math::RichardsonDerivator derivator;

   double maxResValue = std::numeric_limits<double>::max() /n;



   for (unsigned int i = 0; i < n; ++ i) {


      double y = 0;
      const double * x = data.GetPoint(i,y);

      double fval = func( x, p );

      double delta_y_func = y - fval;


      double ey = 0;
      const double * ex = nullptr;
      if (!data.HaveAsymErrors() )
         ex = data.GetPointError(i, ey);
      else {
         double eylow, eyhigh = 0;
         ex = data.GetPointError(i, eylow, eyhigh);
         if ( delta_y_func < 0)
            ey = eyhigh; // function is higher than points
         else
            ey = eylow;
      }
      double e2 = ey * ey;
      // before calculating the gradient check that all error in x are not zero
      unsigned int j = 0;
      while ( j < ndim && ex[j] == 0.)  { j++; }
      // if j is less ndim some elements are not zero
      if (j < ndim) {
         // need an adapter from a multi-dim function to a one-dimensional
         ROOT::Math::OneDimMultiFunctionAdapter<const IModelFunction &> f1D(func,x,0,p);
         // select optimal step size  (use 10--2 by default as was done in TF1:
         double kEps = 0.01;
         double kPrecision = 1.E-8;
         for (unsigned int icoord = 0; icoord < ndim; ++icoord) {
            // calculate derivative for each coordinate
            if (ex[icoord] > 0) {
               //gradCalc.Gradient(x, p, fval, &grad[0]);
               f1D.SetCoord(icoord);
               // optimal step size (take ex[] as scale for the points and 1% of it
               double x0= x[icoord];
               double h = std::max( kEps* std::abs(ex[icoord]), 8.0*kPrecision*(std::abs(x0) + kPrecision) );
               double deriv = derivator.Derivative1(f1D, x[icoord], h);
               double edx = ex[icoord] * deriv;
               e2 += edx * edx;
#ifdef DEBUG
               std::cout << "error for coord " << icoord << " = " << ex[icoord] << " deriv " << deriv << std::endl;
#endif
            }
         }
      }
      double w2 = (e2 > 0) ? 1.0/e2 : 0;
      double resval = w2 * ( y - fval ) *  ( y - fval);

#ifdef DEBUG
      std::cout << x[0] << "  " << y << " ex " << ex[0] << " ey  " << ey << " params : ";
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << "\tresval = " << resval << std::endl;
#endif

      // avoid (infinity and nan ) in the chi2 sum
      // eventually add possibility of excluding some points (like singularity)
      if ( resval < maxResValue )
         chi2 += resval;
      else
         chi2 += maxResValue;
      //nRejected++;

   }

   // reset the number of fitting data points
   nPoints = n;  // no points are rejected
   //if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nPoints  << std::endl;
#endif

   return chi2;

}


////////////////////////////////////////////////////////////////////////////////
/// evaluate the chi2 contribution (residual term) only for data with no coord-errors
/// This function is used in the specialized least square algorithms like FUMILI or L.M.
/// if we have error on the coordinates the residual weight depends on the function value and
/// the approximation used by Fumili and Levenberg-Marquardt cannot be used.
/// Also the expected error and bin integral options should not be used in this case

double FitUtil::EvaluateChi2Residual(const IModelFunction & func, const BinData & data, const double * p, unsigned int i, double * g, double * h, bool hasGrad, bool useFullHessian) {
   if (data.GetErrorType() == BinData::kCoordError && data.Opt().fCoordErrors ) {
      MATH_ERROR_MSG("FitUtil::EvaluateChi2Residual","Error on the coordinates are not used in calculating Chi2 residual");
      return 0; // it will assert otherwise later in GetPoint
   }

   double y, invError = 0;

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useExpErrors = (fitOpt.fExpErrors);
   bool useNormBinVolume = (useBinVolume && fitOpt.fNormBinVolume);

   const double * x1 = data.GetPoint(i,y, invError);

   unsigned int ndim = data.NDim();
   double binVolume = 1.0;
   const double * x2 = nullptr;
   if (useBinVolume || useBinIntegral) x2 = data.BinUpEdge(i);

   std::vector<double> xc;

   if (useBinVolume) {
      if (!useBinIntegral) {
         xc.resize(ndim);
         for (unsigned int j = 0; j < ndim; ++j) {
            binVolume *= std::abs( x2[j]-x1[j] );
            xc[j] = 0.5*(x2[j]+ x1[j]);
         }
      }
      // normalize the bin volume using a reference value
      if (useNormBinVolume) binVolume /= data.RefVolume();
   }

   const double * x = (useBinVolume) ? xc.data() : x1;

   // calculate integral (normalized by bin volume)
   // need to set function and parameters here in case loop is parallelized
   IntegralEvaluator<> igEval( func, p, useBinIntegral);
   double fval0 = (useBinIntegral) ? igEval( x1, x2) : func ( x, p );

   // normalize result if requested according to bin volume
   double fval = fval0;
   if (useBinVolume) fval = fval0*binVolume;

   // expected errors
   if (useExpErrors) {
      // check if a weight factor needs to be applied
      // for bins with y = 0 use as weight the global of the full histogram
      double invWeight = 1.0;
      if (data.IsWeighted() && !fitOpt.fErrors1 ) { // case of weighted Pearson chi2 fit
         invWeight = y * invError * invError;
         if (y == 0) invWeight = (data.SumOfError2() > 0) ? data.SumOfContent()/ data.SumOfError2() : 1.0;
      }
      // compute expected error  as f(x) / weight
      double invError2 = (fval > 0) ? invWeight / fval : 0.0;
      invError = std::sqrt(invError2);
   }


   double resval =   ( y -fval ) * invError;

   // avoid infinities or nan in  resval
   resval = CorrectValue(resval);

   // estimate gradient
   if (g) {

      unsigned int npar = func.NPar();

      // use gradient of model function only if FCN support gradient
      const IGradModelFunction * gfunc = (hasGrad) ?
         dynamic_cast<const IGradModelFunction *>( &func) : nullptr;

      if (!h ) useFullHessian = false;
      // this is not supported yet!
      if (useFullHessian &&  (!gfunc || useBinIntegral || (gfunc && !gfunc->HasParameterHessian())))
         return std::numeric_limits<double>::quiet_NaN();

      if (gfunc) {
         //case function provides gradient
         if (!useBinIntegral ) {
            gfunc->ParameterGradient(x , p, g);
            if (useFullHessian) {
               gfunc->ParameterHessian(x, p, h);
            }
         }
         else {
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral( *gfunc, x1, x2, p, g);
         }
      }
      else {
         SimpleGradientCalculator  gc( npar, func);
         if (!useBinIntegral ) {
            // need to use un-normalized fval
            gc.ParameterGradient(x, p, fval0, g);
         } else {
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral( gc, x1, x2, p, g);
         }
      }
      // multiply by - 1 * weight
      for (unsigned int k = 0; k < npar; ++k) {
         g[k] *= - invError;
         if (useBinVolume) g[k] *= binVolume;
         if (h) {
            for (unsigned int l = 0; l <= k; l++) {  // use lower diagonal because I modify g[k]
               unsigned int idx = l + k * (k + 1) / 2;
               if (useFullHessian) {
                  h[idx] *= 2.* resval * (-invError);  // hessian of model function
                  if (useBinVolume) h[idx] *= binVolume;
               }
               else {
                  h[idx] = 0;
               }
               // add term depending on only gradient of model function
               h[idx] +=  2. * g[k]*g[l];
            }
         }
      }
   }


   return resval;

}

void FitUtil::EvaluateChi2Gradient(const IModelFunction &f, const BinData &data, const double *p, double *grad,
                                   unsigned int &nPoints, ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the gradient of the chi2 function
   // this function is used when the model function knows how to calculate the derivative and we can
   // avoid that the minimizer re-computes them
   //
   // case of chi2 effective (errors on coordinate) is not supported

   if (data.HaveCoordErrors()) {
      MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                     "Error on the coordinates are not used in calculating Chi2 gradient");
      return; // it will assert otherwise later in GetPoint
   }

   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a gradient function

   const IGradModelFunction &func = *fg;

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << nPoints << std::endl;
   std::cout << "evaluate chi2 using function gradient " << &func << "  " << p << std::endl;
#endif

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume) wrefVolume /= data.RefVolume();
   }

   ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT;
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      // do not use GSL integrator which is not thread safe
      igType = ROOT::Math::IntegrationOneDim::kGAUSS;
   }
   IntegralEvaluator<> igEval(func, p, useBinIntegral,igType);

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   std::vector<bool> isPointRejected(initialNPoints);

   auto mapFunction = [&](const unsigned int i) {
      // set all vector values to zero
      std::vector<double> gradFunc(npar);
      std::vector<double> pointContribution(npar);

      const auto x1 = data.GetCoordComponent(i, 0);
      const auto y = data.Value(i);
      auto invError = data.Error(i);

      invError = (invError != 0.0) ? 1.0 / invError : 1;

      double fval = 0;

      const double *x = nullptr;
      std::vector<double> xc;

      unsigned int ndim = data.NDim();
      double binVolume = 1;
      if (useBinVolume) {
         xc.resize(ndim);
         for (unsigned int j = 0; j < ndim; ++j) {
            double x1_j = *data.GetCoordComponent(i, j);
            double x2_j = data.GetBinUpEdgeComponent(i, j);
            binVolume *= std::abs(x2_j - x1_j);
            xc[j] = (useBinIntegral) ? x1_j : 0.5 * (x2_j + x1_j);
         }

         x = xc.data();

         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = *x1;
         for (unsigned int j = 1; j < ndim; ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
         x = x1;
      }

      if (!useBinIntegral) {
         fval = func(x, p);
         func.ParameterGradient(x, p, &gradFunc[0]);
      } else {
         std::vector<double> x2(data.NDim());
         data.GetBinUpEdgeCoordinates(i, x2.data());
         // calculate normalized integral and gradient (divided by bin volume)
         // need to set function and parameters here in case loop is parallelized
         fval = igEval(x, x2.data());
         CalculateGradientIntegral(func, x, x2.data(), p, &gradFunc[0]);
      }
      if (useBinVolume)
         fval *= binVolume;

#ifdef DEBUG
      std::cout << x[0] << "  " << y << "  " << 1. / invError << " params : ";
      for (unsigned int ipar = 0; ipar < npar; ++ipar)
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << std::endl;
#endif
      if (!CheckInfNaNValue(fval)) {
         isPointRejected[i] = true;
         // Return a zero contribution to all partial derivatives on behalf of the current point
         return pointContribution;
      }

      // loop on the parameters
      unsigned int ipar = 0;
      for (; ipar < npar; ++ipar) {

         // correct gradient for bin volumes
         if (useBinVolume)
            gradFunc[ipar] *= binVolume;

         // avoid singularity in the function (infinity and nan ) in the chi2 sum
         // eventually add possibility of excluding some points (like singularity)
         double dfval = gradFunc[ipar];
         if (!CheckInfNaNValue(dfval)) {
            break; // exit loop on parameters
         }

         // calculate derivative point contribution
         pointContribution[ipar] = -2.0 * (y - fval) * invError * invError * gradFunc[ipar];
      }

      if (ipar < npar) {
         // case loop was broken for an overflow in the gradient calculation
         isPointRejected[i] = true;
      }

      return pointContribution;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<double>> &pointContributions) {
      std::vector<double> result(npar);

      for (auto const &pointContribution : pointContributions) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContribution[parameterIndex];
      }

      return result;
   };

   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateChi2Gradient", "Multithread execution policy requires IMT, which is disabled. Changing "
                                               "to ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ROOT::EExecutionPolicy::kSequential) {
      std::vector<std::vector<double>> allGradients(initialNPoints);
      for (unsigned int i = 0; i < initialNPoints; ++i) {
         allGradients[i] = mapFunction(i);
      }
      g = redFunction(allGradients);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(initialNPoints);
      g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, initialNPoints), redFunction, chunks);
   }
#endif
   // else if(executionPolicy == ROOT::Fit::kMultiprocess){
   //    ROOT::TProcessExecutor pool;
   //    g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
   // }
   else {
      Error("FitUtil::EvaluateChi2Gradient",
            "Execution policy unknown. Available choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   //to fix compiler warning
   (void)nChunks;
#endif

   // correct the number of points
   nPoints = initialNPoints;

   if (std::any_of(isPointRejected.begin(), isPointRejected.end(), [](bool point) { return point; })) {
      unsigned nRejected = std::accumulate(isPointRejected.begin(), isPointRejected.end(), 0);
      assert(nRejected <= initialNPoints);
      nPoints = initialNPoints - nRejected;

      if (nPoints < npar)
         MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                        "Error - too many points rejected for overflow in gradient calculation");
   }

   // copy result
   std::copy(g.begin(), g.end(), grad);
}

//______________________________________________________________________________________________________
//
//  Log Likelihood functions
//_______________________________________________________________________________________________________

// utility function used by the likelihoods

// for LogLikelihood functions

double FitUtil::EvaluatePdf(const IModelFunction & func, const UnBinData & data, const double * p, unsigned int i, double * g, double * /*h*/, bool hasGrad, bool) {
   // evaluate the pdf contribution to the generic logl function in case of bin data
   // return actually the log of the pdf and its derivatives


   //func.SetParameters(p);

   const double * x = data.Coords(i);
   double fval = func ( x, p );
   double logPdf = ROOT::Math::Util::EvalLog(fval);
   //return
   if (g == nullptr) return logPdf;

   const IGradModelFunction * gfunc = (hasGrad) ?
      dynamic_cast<const IGradModelFunction *>( &func) : nullptr;

   // gradient  calculation
   if (gfunc) {
      //case function provides gradient
      gfunc->ParameterGradient(  x , p, g );
   }
   else {
      // estimate gradient numerically with simple 2 point rule
      // should probably calculate gradient of log(pdf) is more stable numerically
      SimpleGradientCalculator gc(func.NPar(), func);
      gc.ParameterGradient(x, p, fval, g );
   }
   // divide gradient by function value since returning the logs
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar) {
      g[ipar] /= fval; // this should be checked against infinities
   }

#ifdef DEBUG
   std::cout << x[i] << "\t";
   std::cout << "\tpar = [ " << func.NPar() << " ] =  ";
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
      std::cout << p[ipar] << "\t";
   std::cout << "\tfval = " << fval;
   std::cout << "\tgrad = [ ";
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
      std::cout << g[ipar] << "\t";
   std::cout << " ] "   << std::endl;
#endif


   return logPdf;
}

double FitUtil::EvaluateLogL(const IModelFunction &func, const UnBinData &data, const double *p,
                             int iWeight, bool extended, unsigned int &nPoints,
                             ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the LogLikelihood

   unsigned int n = data.Size();

   //unsigned int nRejected = 0;

   bool normalizeFunc = false;

   // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunctionTempl<double> &>(func)).SetParameters(p);
#endif

   nPoints = data.Size();  // npoints

#ifdef R__USE_IMT
         // in case parameter needs to be propagated to user function use trick to set parameters by calling one time the function
         // this will be done in sequential mode and parameters can be set in a thread safe manner
         if (!normalizeFunc) {
            if (data.NDim() == 1) {
               const double * x = data.GetCoordComponent(0,0);
               func( x, p);
            }
            else {
               std::vector<double> x(data.NDim());
               for (unsigned int j = 0; j < data.NDim(); ++j)
                  x[j] = *data.GetCoordComponent(0, j);
               func( x.data(), p);
            }
         }
#endif

         double norm = 1.0;
         if (normalizeFunc) {
            // compute integral of the function
            std::vector<double> xmin(data.NDim());
            std::vector<double> xmax(data.NDim());
            IntegralEvaluator<> igEval(func, p, true);
            // compute integral in the ranges where is defined
            if (data.Range().Size() > 0) {
               norm = 0;
               for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
                  data.Range().GetRange(&xmin[0], &xmax[0], ir);
                  norm += igEval.Integral(xmin.data(), xmax.data());
               }
            } else {
               // use (-inf +inf)
               data.Range().GetRange(&xmin[0], &xmax[0]);
               // check if funcition is zero at +- inf
               if (func(xmin.data(), p) != 0 || func(xmax.data(), p) != 0) {
                  MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood",
                                 "A range has not been set and the function is not zero at +/- inf");
                  return 0;
               }
               norm = igEval.Integral(&xmin[0], &xmax[0]);
            }
         }

         // needed to compute effective global weight in case of extended likelihood

         auto mapFunction = [&](const unsigned i) {
            double W = 0;
            double W2 = 0;
            double fval = 0;

            if (data.NDim() > 1) {
               std::vector<double> x(data.NDim());
               for (unsigned int j = 0; j < data.NDim(); ++j)
                  x[j] = *data.GetCoordComponent(i, j);
#ifdef USE_PARAMCACHE
               fval = func(x.data());
#else
               fval = func(x.data(), p);
#endif

               // one -dim case
            } else {
               const auto x = data.GetCoordComponent(i, 0);
#ifdef USE_PARAMCACHE
               fval = func(x);
#else
               fval = func(x, p);
#endif
            }

            if (normalizeFunc)
               fval = fval * (1 / norm);

            // function EvalLog protects against negative or too small values of fval
            double logval = ROOT::Math::Util::EvalLog(fval);
            if (iWeight > 0) {
               double weight = data.Weight(i);
               logval *= weight;
               if (iWeight == 2) {
                  logval *= weight; // use square of weights in likelihood
                  if (!extended) {
                     // needed sum of weights and sum of weight square if likelkihood is extended
                     W = weight;
                     W2 = weight * weight;
                  }
               }
            }
            return LikelihoodAux<double>(logval, W, W2);
         };

#ifdef R__USE_IMT
  // auto redFunction = [](const std::vector<LikelihoodAux<double>> & objs){
  //          return std::accumulate(objs.begin(), objs.end(), LikelihoodAux<double>(0.0,0.0,0.0),
  //                      [](const LikelihoodAux<double> &l1, const LikelihoodAux<double> &l2){
  //                          return l1+l2;
  //                 });
  // };
  // do not use std::accumulate to be sure to maintain always the same order
  auto redFunction = [](const std::vector<LikelihoodAux<double>> & objs){
     auto l0 =  LikelihoodAux<double>(0.0,0.0,0.0);
     for ( auto & l : objs ) {
        l0 = l0 + l;
     }
     return l0;
  };
#else
  (void)nChunks;

  // If IMT is disabled, force the execution policy to the serial case
  if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
     Warning("FitUtil::EvaluateLogL", "Multithread execution policy requires IMT, which is disabled. Changing "
                                      "to ROOT::EExecutionPolicy::kSequential.");
     executionPolicy = ROOT::EExecutionPolicy::kSequential;
  }
#endif

  double logl{};
  double sumW{};
  double sumW2{};
  if(executionPolicy == ROOT::EExecutionPolicy::kSequential){
    for (unsigned int i=0; i<n; ++i) {
      auto resArray = mapFunction(i);
      logl+=resArray.logvalue;
      sumW+=resArray.weight;
      sumW2+=resArray.weight2;
    }
#ifdef R__USE_IMT
  } else if(executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
    ROOT::TThreadExecutor pool;
    auto chunks = nChunks !=0? nChunks: setAutomaticChunking(data.Size());
    auto resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction, chunks);
    logl=resArray.logvalue;
    sumW=resArray.weight;
    sumW2=resArray.weight2;
#endif
//   } else if(executionPolicy == ROOT::Fit::kMultiProcess){
    // ROOT::TProcessExecutor pool;
    // res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
  } else{
    Error("FitUtil::EvaluateLogL","Execution policy unknown. Available choices:\n ROOT::EExecutionPolicy::kSequential (default)\n ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
  }

  if (extended) {
      // add Poisson extended term
      double extendedTerm = 0; // extended term in likelihood
      double nuTot = 0;
      // nuTot is integral of function in the range
      // if function has been normalized integral has been already computed
      if (!normalizeFunc) {
         IntegralEvaluator<> igEval( func, p, true);
         std::vector<double> xmin(data.NDim());
         std::vector<double> xmax(data.NDim());

         // compute integral in the ranges where is defined
         if (data.Range().Size() > 0 ) {
            nuTot = 0;
            for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
               data.Range().GetRange(&xmin[0],&xmax[0],ir);
               nuTot += igEval.Integral(xmin.data(),xmax.data());
            }
         } else {
            // use (-inf +inf)
            data.Range().GetRange(&xmin[0],&xmax[0]);
            // check if function is zero at +- inf
            if (func(xmin.data(), p) != 0 || func(xmax.data(), p) != 0) {
               MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood","A range has not been set and the function is not zero at +/- inf");
               return 0;
            }
            nuTot = igEval.Integral(&xmin[0],&xmax[0]);
         }

         // force to be last parameter value
         //nutot = p[func.NDim()-1];
         if (iWeight != 2)
            extendedTerm = - nuTot;  // no need to add in this case n log(nu) since is already computed before
         else {
            // case use weight square in likelihood : compute total effective weight = sw2/sw
            // ignore for the moment case when sumW is zero
            extendedTerm = - (sumW2 / sumW) * nuTot;
         }

      }
      else {
         nuTot = norm;
         extendedTerm = - nuTot + double(n) *  ROOT::Math::Util::EvalLog( nuTot);
         // in case of weights need to use here sum of weights (to be done)
      }
      logl += extendedTerm;

   }

#ifdef DEBUG
   std::cout << "Evaluated log L for parameters (";
   for (unsigned int ip = 0; ip < func.NPar(); ++ip)
      std::cout << " " << p[ip];
   std::cout << ")  fval = " << -logl << std::endl;
#endif

   return -logl;
}

void FitUtil::EvaluateLogLGradient(const IModelFunction &f, const UnBinData &data, const double *p, double *grad,
                                   unsigned int &nPoints, ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the gradient of the log likelihood function

   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   const IGradModelFunction &func = *fg;

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   (const_cast<IGradModelFunction &>(func)).SetParameters(p);

#ifdef DEBUG
    std::cout << "\n===> Evaluate Gradient for parameters ";
         for (unsigned int ip = 0; ip < npar; ++ip)
            std::cout << "  " << p[ip];
         std::cout << "\n";
#endif

   const double kdmax1 = std::sqrt(std::numeric_limits<double>::max());
   const double kdmax2 = std::numeric_limits<double>::max() / (4 * initialNPoints);

   auto mapFunction = [&](const unsigned int i) {
      std::vector<double> gradFunc(npar);
      std::vector<double> pointContribution(npar);


      const double * x = nullptr;
      std::vector<double> xc;
      if (data.NDim() > 1) {
         xc.resize(data.NDim() );
         for (unsigned int j = 0; j < data.NDim(); ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
         x = data.GetCoordComponent(i, 0);
      }

      double fval = func(x, p);
      func.ParameterGradient(x, p, &gradFunc[0]);

#ifdef DEBUG
      {
         R__LOCKGUARD(gROOTMutex);
         if (i < 5 || (i > data.Size()-5) ) {
            if (data.NDim() > 1) std::cout << i << "  x " << x[0] << " y " << x[1] << " func " << fval
                                           << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
            else std::cout << i << "  x " << x[0] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
         }
      }
#endif

      for (unsigned int kpar = 0; kpar < npar; ++kpar) {
         if (fval > 0)
            pointContribution[kpar] = -1. / fval * gradFunc[kpar];
         else if (gradFunc[kpar] != 0) {
            double gg = kdmax1 * gradFunc[kpar];
            if (gg > 0)
               gg = std::min(gg, kdmax2);
            else
               gg = std::max(gg, -kdmax2);
            pointContribution[kpar] = -gg;
         }
         // if func derivative is zero term is also zero so do not add in g[kpar]
      }

      return pointContribution;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<double>> &pointContributions) {
      std::vector<double> result(npar);

      for (auto const &pointContribution : pointContributions) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContribution[parameterIndex];
      }

      return result;
   };

   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateLogLGradient", "Multithread execution policy requires IMT, which is disabled. Changing "
                                               "to ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ROOT::EExecutionPolicy::kSequential) {
      std::vector<std::vector<double>> allGradients(initialNPoints);
      for (unsigned int i = 0; i < initialNPoints; ++i) {
         allGradients[i] = mapFunction(i);
      }
      g = redFunction(allGradients);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(initialNPoints);
      g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, initialNPoints), redFunction, chunks);
   }
#endif
   else {
      Error("FitUtil::EvaluateLogLGradient", "Execution policy unknown. Available choices:\n "
                                             "ROOT::EExecutionPolicy::kSequential (default)\n "
                                             "ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;
#endif

   // copy result
   std::copy(g.begin(), g.end(), grad);
   nPoints = data.Size();  // npoints

#ifdef DEBUG
   std::cout << "FitUtil.cxx : Final gradient ";
   for (unsigned int param = 0; param < npar; param++) {
      std::cout << "  " << grad[param];
   }
   std::cout << "\n";
#endif
}
//_________________________________________________________________________________________________
// for binned log likelihood functions
////////////////////////////////////////////////////////////////////////////////
/// evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
/// and its gradient (gradient of log(pdf))

double FitUtil::EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * p, unsigned int i, double * g, double * h, bool hasGrad, bool useFullHessian) {
   double y = 0;
   const double * x1 = data.GetPoint(i,y);

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   IntegralEvaluator<> igEval( func, p, useBinIntegral);
   const double * x2 = nullptr;
   // calculate the bin volume
   double binVolume = 1;
   std::vector<double> xc;
   if (useBinVolume) {
      unsigned int ndim = data.NDim();
      xc.resize(ndim);
      for (unsigned int j = 0; j < ndim; ++j) {
         double x2j = data.GetBinUpEdgeComponent(i, j);
         binVolume *= std::abs( x2j-x1[j] );
         xc[j] = 0.5*(x2j+ x1[j]);
      }
      // normalize the bin volume using a reference value
      binVolume /= data.RefVolume();
   }

   const double * x = (useBinVolume) ? &xc.front() : x1;

   double fval0 = 0;
   if (!useBinIntegral ) {
      fval0 = func ( x, p );
   }
   else {
      // calculate integral normalized (divided by bin volume)
      std::vector<double> vx2(data.NDim());
      data.GetBinUpEdgeCoordinates(i, vx2.data());
      fval0 = igEval( x1, vx2.data() ) ;
   }
   double fval = fval0;
   if (useBinVolume) fval = fval0*binVolume;

   // logPdf for Poisson: ignore constant term depending on N
   fval = std::max(fval, 0.0);  // avoid negative or too small values
   double nlogPdf =  fval;
   if (y > 0.0) {
      // include also constants due to saturate model (see Baker-Cousins paper)
      nlogPdf -= y * ROOT::Math::Util::EvalLog( fval / y) - y;
   }

   if (g == nullptr) return nlogPdf;

   unsigned int npar = func.NPar();
   const IGradModelFunction * gfunc = (hasGrad) ?
      dynamic_cast<const IGradModelFunction *>( &func) : nullptr;

   // for full Hessian we need a gradient function and not bin intgegral computation
   if (useFullHessian &&  (!gfunc || useBinIntegral || (gfunc && !gfunc->HasParameterHessian())))
      return std::numeric_limits<double>::quiet_NaN();

   // gradient  calculation
   if (gfunc) {
      //case function provides gradient
      if (!useBinIntegral ) {
         gfunc->ParameterGradient(  x , p, g );
         if (useFullHessian && h) {
            if (!gfunc->HasParameterHessian())
               return std::numeric_limits<double>::quiet_NaN();
            bool goodHessFunc = gfunc->ParameterHessian(x , p, h);
            if (!goodHessFunc) {
               return std::numeric_limits<double>::quiet_NaN();
            }
         }
      }
      else {
         // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral( *gfunc, x1, x2, p, g);
      }

   }
   else {
      SimpleGradientCalculator  gc(func.NPar(), func);
      if (!useBinIntegral )
         gc.ParameterGradient(x, p, fval0, g);
      else {
        // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral( gc, x1, x2, p, g);
      }
   }
   // correct g[] do be derivative of poisson term. We compute already derivative w.r.t. LL
   double coeffGrad = (fval > 0) ? (1. - y/fval) : ( (y > 0) ? std::sqrt( std::numeric_limits<double>::max() )  : 1. );
   double coeffHess = (fval > 0) ?  y/(fval*fval) : ( (y > 0) ? std::sqrt( std::numeric_limits<double>::max() )  : 0. );
   if (useBinVolume) {
      coeffGrad *= binVolume;
      coeffHess *= binVolume*binVolume;
   }
   for (unsigned int k = 0; k < npar; ++k) {
      // compute also approximate Hessian (excluding term with second derivative of model function)
      if (h) {
         for (unsigned int l = k; l < npar; ++l) {
            unsigned int idx = k + l * (l + 1) / 2;
            if (useFullHessian) {
               h[idx] *= coeffGrad;  // h contains first model function derivatives
            }
            else {
               h[idx] = 0;
            }
            // add term deoending on only gradient of model function
            h[idx] += coeffHess * g[k]*g[l];  // g are model function derivatives
         }
      }
      // compute gradient of NLL element
      // and apply bin volume correction if needed
      g[k] *= coeffGrad;
      if (useBinVolume)
         g[k] *= binVolume;
   }

#ifdef DEBUG
   std::cout << "x = " << x[0] << " y " << y << " fval " << fval << " logPdf = " << nlogPdf << " gradient : ";
   for (unsigned int ipar = 0; ipar < npar; ++ipar)
      std::cout << g[ipar] << "\t";
   if (h) {
      std::cout << "\thessian : ";
      for (unsigned int ipar = 0; ipar < npar; ++ipar) {
         std::cout << " {";
         for (unsigned int jpar = 0; jpar <= ipar; ++jpar) {
            std::cout << h[ipar + jpar * (jpar + 1) / 2] << "\t";
         }
         std::cout << "}";
      }
   }
   std::cout << std::endl;
#endif
#undef DEBUG

   return nlogPdf;
}

double FitUtil::EvaluatePoissonLogL(const IModelFunction &func, const BinData &data, const double *p, int iWeight,
                                    bool extended, unsigned int &nPoints, ROOT::EExecutionPolicy executionPolicy,
                                    unsigned nChunks)
{
   // evaluate the Poisson Log Likelihood
   // for binned likelihood fits
   // this is Sum ( f(x_i)  -  y_i * log( f (x_i) ) )
   // add as well constant term for saturated model to make it like a Chi2/2
   // by default is extended. If extended is false the fit is not extended and
   // the global poisson term is removed (i.e is a binomial fit)
   // (remember that in this case one needs to have a function with a fixed normalization
   // like in a non extended unbinned fit)
   //
   // if use Weight use a weighted dataset
   // iWeight = 1 ==> logL = Sum( w f(x_i) )
   // case of iWeight==1 is actually identical to weight==0
   // iWeight = 2 ==> logL = Sum( w*w * f(x_i) )
   //
   // nPoints returns the points where bin content is not zero


   unsigned int n = data.Size();

#ifdef USE_PARAMCACHE
   (const_cast<IModelFunction &>(func)).SetParameters(p);
#endif

   nPoints = data.Size();  // npoints


   // get fit option and check case of using integral of bins
   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());
   bool useW2 = (iWeight == 2);

   // normalize if needed by a reference volume value
   double wrefVolume = 1.0;
   if (useBinVolume) {
      if (fitOpt.fNormBinVolume) wrefVolume /= data.RefVolume();
   }

//#define DEBUG
#ifdef DEBUG
   std::cout << "Evaluate PoissonLogL for params = [ ";
   for (unsigned int j = 0; j < func.NPar(); ++j) std::cout << p[j] << " , ";
   std::cout << "]  - data size = " << n << " useBinIntegral " << useBinIntegral << " useBinVolume "
             << useBinVolume << " useW2 " << useW2 << " wrefVolume = " << wrefVolume << std::endl;
#endif


   ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT;
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      // do not use GSL integrator which is not thread safe
      igType = ROOT::Math::IntegrationOneDim::kGAUSS;
   }
#ifdef USE_PARAMCACHE
   IntegralEvaluator<> igEval(func, nullptr, useBinIntegral, igType);
#else
   IntegralEvaluator<> igEval(func, p, useBinIntegral, igType);
#endif

   auto mapFunction = [&](const unsigned i) {
      auto x1 = data.GetCoordComponent(i, 0);
      auto y = *data.ValuePtr(i);

      const double *x = nullptr;
      std::vector<double> xc;
      double fval = 0;
      double binVolume = 1.0;

      if (useBinVolume) {
         unsigned int ndim = data.NDim();
         xc.resize(data.NDim());
         for (unsigned int j = 0; j < ndim; ++j) {
            double xx = *data.GetCoordComponent(i, j);
            double x2 = data.GetBinUpEdgeComponent(i, j);
            binVolume *= std::abs(x2 - xx);
            xc[j] = (useBinIntegral) ? xx : 0.5 * (x2 + xx);
         }
         x = xc.data();
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if (data.NDim() > 1) {
         xc.resize(data.NDim());
         xc[0] = *x1;
         for (unsigned int j = 1; j < data.NDim(); ++j) {
            xc[j] = *data.GetCoordComponent(i, j);
         }
         x = xc.data();
      } else {
         x = x1;
      }

      if (!useBinIntegral) {
#ifdef USE_PARAMCACHE
         fval = func(x);
#else
         fval = func(x, p);
#endif
      } else {
         // calculate integral (normalized by bin volume)
         // need to set function and parameters here in case loop is parallelized
         std::vector<double> x2(data.NDim());
         data.GetBinUpEdgeCoordinates(i, x2.data());
         fval = igEval(x, x2.data());
      }
      if (useBinVolume) fval *= binVolume;



#ifdef DEBUG
      int NSAMPLE = 100;
      if (i % NSAMPLE == 0) {
         std::cout << "evt " << i << " x = [ ";
         for (unsigned int j = 0; j < func.NDim(); ++j) std::cout << x[j] << " , ";
         std::cout << "]  ";
         if (fitOpt.fIntegral) {
            std::cout << "x2 = [ ";
            for (unsigned int j = 0; j < func.NDim(); ++j) std::cout << data.GetBinUpEdgeComponent(i, j) << " , ";
            std::cout << "] ";
         }
         std::cout << "  y = " << y << " fval = " << fval << std::endl;
      }
#endif


      // EvalLog protects against 0 values of fval but don't want to add in the -log sum
      // negative values of fval
      fval = std::max(fval, 0.0);

      double nloglike = 0; // negative loglikelihood
      if (useW2) {
         // apply weight correction . Effective weight is error^2/ y
         // and expected events in bins is fval/weight
         // can apply correction only when y is not zero otherwise weight is undefined
         // (in case of weighted likelihood I don't care about the constant term due to
         // the saturated model)

         // use for the empty bins the global weight
         double weight = 1.0;
         if (y != 0) {
            double error = data.Error(i);
            weight = (error * error) / y; // this is the bin effective weight
            nloglike -= weight * y * ( ROOT::Math::Util::EvalLog(fval/y) );
         }
         else {
            // for empty bin use the average weight  computed from the total data weight
            weight = data.SumOfError2()/ data.SumOfContent();
         }
         if (extended) {
            nloglike += weight  *  ( fval - y);
         }

      } else {
         // standard case no weights or iWeight=1
         // this is needed for Poisson likelihood (which are extended and not for multinomial)
         // the formula below  include constant term due to likelihood of saturated model (f(x) = y)
         // (same formula as in Baker-Cousins paper, page 439 except a factor of 2
         if (extended) nloglike = fval - y;

         if (y >  0) {
            nloglike += y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval));
         }
      }
#ifdef DEBUG
      {
         R__LOCKGUARD(gROOTMutex);
         std::cout << " nll = " << nloglike << std::endl;
      }
#endif
      return nloglike;
   };

#ifdef R__USE_IMT
   auto redFunction = [](const std::vector<double> &objs) {
      return std::accumulate(objs.begin(), objs.end(), double{});
   };
#else
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluatePoissonLogL", "Multithread execution policy requires IMT, which is disabled. Changing "
                                              "to ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ROOT::EExecutionPolicy::kSequential;
   }
#endif

   double res{};
   if (executionPolicy == ROOT::EExecutionPolicy::kSequential) {
      for (unsigned int i = 0; i < n; ++i) {
         res += mapFunction(i);
      }
#ifdef R__USE_IMT
   } else if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size());
      res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction, chunks);
#endif
      //   } else if(executionPolicy == ROOT::Fit::kMultitProcess){
      // ROOT::TProcessExecutor pool;
      // res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
   } else {
      Error("FitUtil::EvaluatePoissonLogL",
            "Execution policy unknown. Available choices:\n ROOT::EExecutionPolicy::kSequential (default)\n ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

#ifdef DEBUG
   std::cout << "Loglikelihood  = " << res << std::endl;
#endif

   return res;
}

void FitUtil::EvaluatePoissonLogLGradient(const IModelFunction &f, const BinData &data, const double *p, double *grad,
                                          unsigned int &, ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the gradient of the Poisson log likelihood function

   const IGradModelFunction *fg = dynamic_cast<const IGradModelFunction *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   const IGradModelFunction &func = *fg;

#ifdef USE_PARAMCACHE
   (const_cast<IGradModelFunction &>(func)).SetParameters(p);
#endif

   const DataOptions &fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges();
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0;
   if (useBinVolume && fitOpt.fNormBinVolume)
      wrefVolume /= data.RefVolume();

   ROOT::Math::IntegrationOneDim::Type igType = ROOT::Math::IntegrationOneDim::kDEFAULT;
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      // do not use GSL integrator which is not thread safe
      igType = ROOT::Math::IntegrationOneDim::kGAUSS;
   }

   IntegralEvaluator<> igEval(func, p, useBinIntegral, igType);

   unsigned int npar = func.NPar();
   unsigned initialNPoints = data.Size();

   auto mapFunction = [&](const unsigned int i) {
      // set all vector values to zero
      std::vector<double> gradFunc(npar);
      std::vector<double> pointContribution(npar);

      const auto x1 = data.GetCoordComponent(i, 0);
      const auto y = data.Value(i);
      auto invError = data.Error(i);

      invError = (invError != 0.0) ? 1.0 / invError : 1;

      double fval = 0;

      const double *x = nullptr;
      std::vector<double> xc;

      unsigned ndim = data.NDim();
      double binVolume = 1.0;
      if (useBinVolume) {

         xc.resize(ndim);

         for (unsigned int j = 0; j < ndim; ++j) {
            double x1_j = *data.GetCoordComponent(i, j);
            double x2_j = data.GetBinUpEdgeComponent(i, j);
            binVolume *= std::abs(x2_j - x1_j);
            xc[j] = (useBinIntegral) ? x1_j : 0.5 * (x2_j + x1_j);
         }

         x = xc.data();

         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      } else if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = *x1;
         for (unsigned int j = 1; j < ndim; ++j)
            xc[j] = *data.GetCoordComponent(i, j);
         x = xc.data();
      } else {
         x = x1;
      }

      if (!useBinIntegral) {
         fval = func(x, p);
         func.ParameterGradient(x, p, &gradFunc[0]);
      } else {
         // calculate integral (normalized by bin volume)
         // need to set function and parameters here in case loop is parallelized
         std::vector<double> x2(data.NDim());
         data.GetBinUpEdgeCoordinates(i, x2.data());
         fval = igEval(x, x2.data());
         CalculateGradientIntegral(func, x, x2.data(), p, &gradFunc[0]);
      }
      if (useBinVolume)
         fval *= binVolume;

#ifdef DEBUG
      {
         R__LOCKGUARD(gROOTMutex);
         if (i < 5 || (i > data.Size()-5) ) {
            if (data.NDim() > 1) std::cout << i << "  x " << x[0] << " y " << x[1] << " func " << fval
                                           << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
            else std::cout << i << "  x " << x[0] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  " << gradFunc[3] << std::endl;
         }
      }
#endif

      // correct the gradient
      for (unsigned int ipar = 0; ipar < npar; ++ipar) {

         // correct gradient for bin volumes
         if (useBinVolume)
            gradFunc[ipar] *= binVolume;

         // df/dp * (1.  - y/f )
         if (fval > 0)
            pointContribution[ipar] = gradFunc[ipar] * (1. - y / fval);
         else if (gradFunc[ipar] != 0) {
            const double kdmax1 = std::sqrt(std::numeric_limits<double>::max());
            const double kdmax2 = std::numeric_limits<double>::max() / (4 * initialNPoints);
            double gg = kdmax1 * gradFunc[ipar];
            if (gg > 0)
               gg = std::min(gg, kdmax2);
            else
               gg = std::max(gg, -kdmax2);
            pointContribution[ipar] = -gg;
         }
      }


      return pointContribution;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<double>> &pointContributions) {
      std::vector<double> result(npar);

      for (auto const &pointContribution : pointContributions) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContribution[parameterIndex];
      }

      return result;
   };

   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluatePoissonLogLGradient",
              "Multithread execution policy requires IMT, which is disabled. Changing "
              "to ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ROOT::EExecutionPolicy::kSequential) {
      std::vector<std::vector<double>> allGradients(initialNPoints);
      for (unsigned int i = 0; i < initialNPoints; ++i) {
         allGradients[i] = mapFunction(i);
      }
      g = redFunction(allGradients);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(initialNPoints);
      g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, initialNPoints), redFunction, chunks);
   }
#endif

   // else if(executionPolicy == ROOT::Fit::kMultiprocess){
   //    ROOT::TProcessExecutor pool;
   //    g = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, n), redFunction);
   // }
   else {
      Error("FitUtil::EvaluatePoissonLogLGradient",
            "Execution policy unknown. Available choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
   }

#ifndef R__USE_IMT
   //to fix compiler warning
   (void)nChunks;
#endif

   // copy result
   std::copy(g.begin(), g.end(), grad);

#ifdef DEBUG
   std::cout << "***** Final gradient : ";
   for (unsigned int ii = 0; ii< npar; ++ii) std::cout << grad[ii] << "   ";
   std::cout << "\n";
#endif

}


unsigned FitUtil::setAutomaticChunking(unsigned nEvents){
      auto ncpu  = ROOT::GetThreadPoolSize();
      if (nEvents/ncpu < 1000) return ncpu;
      return nEvents/1000;
      //return ((nEvents/ncpu + 1) % 1000) *40 ; //arbitrary formula
}

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD
namespace FitUtil {

namespace {

template <class T>
T NumericMax()
{
   return T{std::numeric_limits<typename T::value_type>::max()};
}

template <typename V, typename S>
void Load(V &v, S const *ptr)
{
   for (size_t i = 0; i < V::size(); ++i)
      v[i] = ptr[i];
}

template <typename T>
auto ReduceAdd(const T &v)
{
   typename T::value_type result(0);
   for (size_t i = 0; i < T::size(); ++i) {
      result += v[i];
   }
   return result;
}

template <typename T>
bool MaskEmpty(T mask)
{
   for (size_t i = 0; i < T::size(); ++i)
      if (mask[i])
         return false;
   return true;
}

template <class T = ROOT::Double_v>
auto Int2Mask(unsigned i)
{
   T x;
   for (unsigned j = 0; j < T::size(); j++) {
      x[j] = j;
   }
   return x < T(i);
}

} // namespace

double Evaluate<Double_v>::EvalChi2(const IModelFunctionTempl<Double_v> &func, const BinData &data, const double *p,
                                    unsigned int &nPoints, ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the chi2 given a  vectorized function reference  , the data and returns the value and also in nPoints
   // the actual number of used points
   // normal chi2 using only error on values (from fitting histogram)
   // optionally the integral of function in the bin is used

   // Info("EvalChi2","Using vectorized implementation %d",(int) data.Opt().fIntegral);

   unsigned int n = data.Size();
   nPoints = data.Size(); // npoints

   // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunctionTempl<Double_v> &>(func)).SetParameters(p);
#endif
   // do not cache parameter values (it is not thread safe)
   // func.SetParameters(p);

   // get fit option and check case if using integral of bins
   const DataOptions &fitOpt = data.Opt();
   if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
      Error("FitUtil::EvaluateChi2",
            "The vectorized implementation doesn't support Integrals, BinVolume or ExpErrors\n. Aborting operation.");

   (const_cast<IModelFunctionTempl<Double_v> &>(func)).SetParameters(p);

   double maxResValue = std::numeric_limits<double>::max() / n;
   std::vector<double> ones{1., 1., 1., 1.};
   auto vecSize = Double_v::size();

   auto mapFunction = [&](unsigned int i) {
      // in case of no error in y invError=1 is returned
      Double_v x1, y, invErrorVec;
      Load(x1, data.GetCoordComponent(i * vecSize, 0));
      Load(y, data.ValuePtr(i * vecSize));
      const auto invError = data.ErrorPtr(i * vecSize);
      auto invErrorptr = (invError != nullptr) ? invError : &ones.front();
      Load(invErrorVec, invErrorptr);

      const Double_v *x;
      std::vector<Double_v> xc;
      if (data.NDim() > 1) {
         xc.resize(data.NDim());
         xc[0] = x1;
         for (unsigned int j = 1; j < data.NDim(); ++j)
            Load(xc[j], data.GetCoordComponent(i * vecSize, j));
         x = xc.data();
      } else {
         x = &x1;
      }

      Double_v fval{};

#ifdef USE_PARAMCACHE
      fval = func(x);
#else
      fval = func(x, p);
#endif

      Double_v tmp = (y - fval) * invErrorVec;
      Double_v chi2 = tmp * tmp;

      // avoid infinity or nan in chi2 values due to wrong function values
      where(chi2 > maxResValue, chi2) = maxResValue;

      return chi2;
   };

   auto redFunction = [](const std::vector<Double_v> &objs) {
      return std::accumulate(objs.begin(), objs.end(), Double_v{});
   };

#ifndef R__USE_IMT
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateChi2", "Multithread execution policy requires IMT, which is disabled. Changing "
                                       "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   Double_v res{};
   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      ROOT::TSequentialExecutor pool;
      res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction);
#ifdef R__USE_IMT
   } else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size() / vecSize);
      res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
   } else {
      Error("FitUtil::EvaluateChi2",
            "Execution policy unknown. Available choices:\n ::ROOT::EExecutionPolicy::kSequential (default)\n "
            "::ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

   // Last SIMD vector of elements (if padding needed)
   if (data.Size() % vecSize != 0)
      where(Int2Mask(data.Size() % vecSize), res) = res + mapFunction(data.Size() / vecSize);

   return ReduceAdd(res);
}

double Evaluate<Double_v>::EvalLogL(const IModelFunctionTempl<Double_v> &func, const UnBinData &data,
                                    const double *const p, int iWeight, bool extended, unsigned int &nPoints,
                                    ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the LogLikelihood
   unsigned int n = data.Size();
   nPoints = data.Size(); // npoints

   // unsigned int nRejected = 0;
   bool normalizeFunc = false;

   // set parameters of the function to cache integral value
#ifdef USE_PARAMCACHE
   (const_cast<IModelFunctionTempl<Double_v> &>(func)).SetParameters(p);
#endif

#ifdef R__USE_IMT
   // in case parameter needs to be propagated to user function use trick to set parameters by calling one time the
   // function this will be done in sequential mode and parameters can be set in a thread safe manner
   if (!normalizeFunc) {
      if (data.NDim() == 1) {
         Double_v x;
         Load(x, data.GetCoordComponent(0, 0));
         func(&x, p);
      } else {
         std::vector<Double_v> x(data.NDim());
         for (unsigned int j = 0; j < data.NDim(); ++j)
            Load(x[j], data.GetCoordComponent(0, j));
         func(x.data(), p);
      }
   }
#endif

   // this is needed if function must be normalized
   double norm = 1.0;
   if (normalizeFunc) {
      // compute integral of the function
      std::vector<double> xmin(data.NDim());
      std::vector<double> xmax(data.NDim());
      IntegralEvaluator<IModelFunctionTempl<Double_v>> igEval(func, p, true);
      // compute integral in the ranges where is defined
      if (data.Range().Size() > 0) {
         norm = 0;
         for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
            data.Range().GetRange(&xmin[0], &xmax[0], ir);
            norm += igEval.Integral(xmin.data(), xmax.data());
         }
      } else {
         // use (-inf +inf)
         data.Range().GetRange(&xmin[0], &xmax[0]);
         // check if function is zero at +- inf
         Double_v xmin_v;
         Double_v xmax_v;
         Load(xmin_v, xmin.data());
         Load(xmax_v, xmax.data());
         if (ReduceAdd(func(&xmin_v, p)) != 0 || ReduceAdd(func(&xmax_v, p)) != 0) {
            MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood",
                           "A range has not been set and the function is not zero at +/- inf");
            return 0;
         }
         norm = igEval.Integral(&xmin[0], &xmax[0]);
      }
   }

   // needed to compute effective global weight in case of extended likelihood

   auto vecSize = Double_v::size();
   unsigned int numVectors = n / vecSize;

   auto mapFunction = [&, p](const unsigned i) {
      Double_v W{};
      Double_v W2{};
      Double_v fval{};

      (void)p; /* avoid unused lambda capture warning if PARAMCACHE is disabled */

      Double_v x1;
      Load(x1, data.GetCoordComponent(i * vecSize, 0));
      const Double_v *x = nullptr;
      unsigned int ndim = data.NDim();
      std::vector<Double_v> xc;
      if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = x1;
         for (unsigned int j = 1; j < ndim; ++j)
            Load(xc[j], data.GetCoordComponent(i * vecSize, j));
         x = xc.data();
      } else {
         x = &x1;
      }

#ifdef USE_PARAMCACHE
      fval = func(x);
#else
      fval = func(x, p);
#endif

#ifdef DEBUG_FITUTIL
      if (i < 5 || (i > numVectors - 5)) {
         if (ndim == 1)
            std::cout << i << "  x " << x[0] << " fval = " << fval;
         else
            std::cout << i << "  x " << x[0] << " y " << x[1] << " fval = " << fval;
      }
#endif

      if (normalizeFunc)
         fval = fval * (1 / norm);

      // function EvalLog protects against negative or too small values of fval
      auto logval = ROOT::Math::Util::EvalLog(fval);
      if (iWeight > 0) {
         Double_v weight{};
         if (data.WeightsPtr(i) == nullptr)
            weight = 1;
         else
            Load(weight, data.WeightsPtr(i * vecSize));
         logval *= weight;
         if (iWeight == 2) {
            logval *= weight; // use square of weights in likelihood
            if (!extended) {
               // needed sum of weights and sum of weight square if likelkihood is extended
               W = weight;
               W2 = weight * weight;
            }
         }
      }
#ifdef DEBUG_FITUTIL
      if (i < 5 || (i > numVectors - 5)) {
         std::cout << "   " << fval << "  logfval " << logval << std::endl;
      }
#endif

      return LikelihoodAux<Double_v>(logval, W, W2);
   };

   auto redFunction = [](const std::vector<LikelihoodAux<Double_v>> &objs) {
      return std::accumulate(
         objs.begin(), objs.end(), LikelihoodAux<Double_v>(),
         [](const LikelihoodAux<Double_v> &l1, const LikelihoodAux<Double_v> &l2) { return l1 + l2; });
   };

#ifndef R__USE_IMT
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateLogL", "Multithread execution policy requires IMT, which is disabled. Changing "
                                       "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   Double_v logl_v{};
   Double_v sumW_v{};
   Double_v sumW2_v{};
   ROOT::Fit::FitUtil::LikelihoodAux<Double_v> resArray;
   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      ROOT::TSequentialExecutor pool;
      resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction);
#ifdef R__USE_IMT
   } else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(numVectors);
      resArray = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
   } else {
      Error("FitUtil::EvaluateLogL",
            "Execution policy unknown. Available choices:\n ::ROOT::EExecutionPolicy::kSequential (default)\n "
            "::ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

   logl_v = resArray.logvalue;
   sumW_v = resArray.weight;
   sumW2_v = resArray.weight2;

   // Compute the contribution from the remaining points ( Last padded SIMD vector of elements )
   unsigned int remainingPoints = n % vecSize;
   if (remainingPoints > 0) {
      auto remainingPointsContribution = mapFunction(numVectors);
      // Add the contribution from the valid remaining points and store the result in the output variable
      auto remainingMask = Int2Mask(remainingPoints);
      where(remainingMask, logl_v) = logl_v + remainingPointsContribution.logvalue;
      where(remainingMask, sumW_v) = sumW_v + remainingPointsContribution.weight;
      where(remainingMask, sumW2_v) = sumW2_v + remainingPointsContribution.weight2;
   }

   // reduce vector type to double.
   double logl = ReduceAdd(logl_v);
   double sumW = ReduceAdd(sumW_v);
   double sumW2 = ReduceAdd(sumW2_v);

   if (extended) {
      // add Poisson extended term
      double extendedTerm = 0; // extended term in likelihood
      double nuTot = 0;
      // nuTot is integral of function in the range
      // if function has been normalized integral has been already computed
      if (!normalizeFunc) {
         IntegralEvaluator<IModelFunctionTempl<Double_v>> igEval(func, p, true);
         std::vector<double> xmin(data.NDim());
         std::vector<double> xmax(data.NDim());

         // compute integral in the ranges where is defined
         if (data.Range().Size() > 0) {
            nuTot = 0;
            for (unsigned int ir = 0; ir < data.Range().Size(); ++ir) {
               data.Range().GetRange(&xmin[0], &xmax[0], ir);
               nuTot += igEval.Integral(xmin.data(), xmax.data());
            }
         } else {
            // use (-inf +inf)
            data.Range().GetRange(&xmin[0], &xmax[0]);
            // check if function is zero at +- inf
            Double_v xmin_v, xmax_v;
            Load(xmin_v, xmin.data());
            Load(xmax_v, xmax.data());
            if (ReduceAdd(func(&xmin_v, p)) != 0 || ReduceAdd(func(&xmax_v, p)) != 0) {
               MATH_ERROR_MSG("FitUtil::EvaluateLogLikelihood",
                              "A range has not been set and the function is not zero at +/- inf");
               return 0;
            }
            nuTot = igEval.Integral(&xmin[0], &xmax[0]);
         }

         // force to be last parameter value
         // nutot = p[func.NDim()-1];
         if (iWeight != 2)
            extendedTerm = -nuTot; // no need to add in this case n log(nu) since is already computed before
         else {
            // case use weight square in likelihood : compute total effective weight = sw2/sw
            // ignore for the moment case when sumW is zero
            extendedTerm = -(sumW2 / sumW) * nuTot;
         }

      } else {
         nuTot = norm;
         extendedTerm = -nuTot + double(n) * ROOT::Math::Util::EvalLog(nuTot);
         // in case of weights need to use here sum of weights (to be done)
      }

      logl += extendedTerm;
   }

#ifdef DEBUG_FITUTIL
   std::cout << "Evaluated log L for parameters (";
   for (unsigned int ip = 0; ip < func.NPar(); ++ip)
      std::cout << " " << p[ip];
   std::cout << ")  nll = " << -logl << std::endl;
#endif

   return -logl;
}

double Evaluate<Double_v>::EvalPoissonLogL(const IModelFunctionTempl<Double_v> &func, const BinData &data,
                                           const double *p, int iWeight, bool extended, unsigned int,
                                           ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the Poisson Log Likelihood
   // for binned likelihood fits
   // this is Sum ( f(x_i)  -  y_i * log( f (x_i) ) )
   // add as well constant term for saturated model to make it like a Chi2/2
   // by default is extended. If extended is false the fit is not extended and
   // the global poisson term is removed (i.e is a binomial fit)
   // (remember that in this case one needs to have a function with a fixed normalization
   // like in a non extended binned fit)
   //
   // if use Weight use a weighted dataset
   // iWeight = 1 ==> logL = Sum( w f(x_i) )
   // case of iWeight==1 is actually identical to weight==0
   // iWeight = 2 ==> logL = Sum( w*w * f(x_i) )
   //

#ifdef USE_PARAMCACHE
   (const_cast<IModelFunctionTempl<Double_v> &>(func)).SetParameters(p);
#endif
   auto vecSize = Double_v::size();
   // get fit option and check case of using integral of bins
   const DataOptions &fitOpt = data.Opt();
   if (fitOpt.fExpErrors || fitOpt.fIntegral)
      Error("FitUtil::EvaluateChi2",
            "The vectorized implementation doesn't support Integrals or BinVolume\n. Aborting operation.");
   bool useW2 = (iWeight == 2);

   auto mapFunction = [&](unsigned int i) {
      Double_v y;
      Load(y, data.ValuePtr(i * vecSize));
      Double_v fval{};

      if (data.NDim() > 1) {
         std::vector<Double_v> x(data.NDim());
         for (unsigned int j = 0; j < data.NDim(); ++j)
            Load(x[j], data.GetCoordComponent(i * vecSize, j));
#ifdef USE_PARAMCACHE
         fval = func(x.data());
#else
         fval = func(x.data(), p);
#endif
         // one -dim case
      } else {
         Double_v x;
         Load(x, data.GetCoordComponent(i * vecSize, 0));
#ifdef USE_PARAMCACHE
         fval = func(&x);
#else
         fval = func(&x, p);
#endif
      }

      // EvalLog protects against 0 values of fval but don't want to add in the -log sum
      // negative values of fval
      where(fval < 0.0, fval) = 0.0;

      Double_v nloglike{}; // negative loglikelihood

      if (useW2) {
         // apply weight correction . Effective weight is error^2/ y
         // and expected events in bins is fval/weight
         // can apply correction only when y is not zero otherwise weight is undefined
         // (in case of weighted likelihood I don't care about the constant term due to
         // the saturated model)
         assert(data.GetErrorType() != ROOT::Fit::BinData::ErrorType::kNoError);
         Double_v error = 0.0;
         Load(error, data.ErrorPtr(i * vecSize));
         // for empty bin use the average weight  computed from the total data weight
         auto m = y != 0.0;
         Double_v weight{};
         where(m, weight) = (error * error) / y;
         where(!m, weight) = Double_v{data.SumOfError2() / data.SumOfContent()};
         if (extended) {
            nloglike = weight * (fval - y);
         }
         where(y != 0, nloglike) =
            nloglike + weight * y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval));

      } else {
         // standard case no weights or iWeight=1
         // this is needed for Poisson likelihood (which are extended and not for multinomial)
         // the formula below  include constant term due to likelihood of saturated model (f(x) = y)
         // (same formula as in Baker-Cousins paper, page 439 except a factor of 2
         if (extended)
            nloglike = fval - y;

         where(y > 0, nloglike) = nloglike + y * (ROOT::Math::Util::EvalLog(y) - ROOT::Math::Util::EvalLog(fval));
      }

      return nloglike;
   };

#ifdef R__USE_IMT
   auto redFunction = [](const std::vector<Double_v> &objs) {
      return std::accumulate(objs.begin(), objs.end(), Double_v{});
   };
#else
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::Evaluate<T>::EvalPoissonLogL",
              "Multithread execution policy requires IMT, which is disabled. Changing "
              "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   Double_v res{};
   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      for (unsigned int i = 0; i < (data.Size() / vecSize); i++) {
         res += mapFunction(i);
      }
#ifdef R__USE_IMT
   } else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(data.Size() / vecSize);
      res = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, data.Size() / vecSize), redFunction, chunks);
#endif
   } else {
      Error("FitUtil::Evaluate<T>::EvalPoissonLogL",
            "Execution policy unknown. Available choices:\n ::ROOT::EExecutionPolicy::kSequential (default)\n "
            "::ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

   // Last padded SIMD vector of elements
   if (data.Size() % vecSize != 0)
      where(Int2Mask(data.Size() % vecSize), res) = res + mapFunction(data.Size() / vecSize);

   return ReduceAdd(res);
}

namespace {

// Compute a mask to filter out infinite numbers and NaN values.
// The argument rval is updated so infinite numbers and NaN values are replaced by
// maximum finite values (preserving the original sign).
auto CheckInfNaNValues(Double_v &rval)
{
   auto mask = rval > -NumericMax<Double_v>() && rval < NumericMax<Double_v>();

   // Case +inf or nan
   where(!mask, rval) = +NumericMax<Double_v>();

   // Case -inf
   where(!mask && rval < 0, rval) = -NumericMax<Double_v>();

   return mask;
}

} // namespace

void Evaluate<Double_v>::EvalChi2Gradient(const IModelFunctionTempl<Double_v> &f, const BinData &data, const double *p,
                                          double *grad, unsigned int &nPoints, ::ROOT::EExecutionPolicy executionPolicy,
                                          unsigned nChunks)
{
   // evaluate the gradient of the chi2 function
   // this function is used when the model function knows how to calculate the derivative and we can
   // avoid that the minimizer re-computes them
   //
   // case of chi2 effective (errors on coordinate) is not supported

   if (data.HaveCoordErrors()) {
      MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                     "Error on the coordinates are not used in calculating Chi2 gradient");
      return; // it will assert otherwise later in GetPoint
   }

   auto *fg = dynamic_cast<const IGradModelFunctionTempl<Double_v> *>(&f);
   assert(fg != nullptr); // must be called by a gradient function

   auto &func = *fg;

   const DataOptions &fitOpt = data.Opt();
   if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
      Error("FitUtil::EvaluateChi2Gradient", "The vectorized implementation doesn't support Integrals,"
                                             "BinVolume or ExpErrors\n. Aborting operation.");

   unsigned int npar = func.NPar();
   auto vecSize = Double_v::size();
   unsigned initialNPoints = data.Size();
   unsigned numVectors = initialNPoints / vecSize;

   // numVectors + 1 because of the padded data (call to mapFunction with i = numVectors after the main loop)
   std::vector<Double_v::mask_type> validPointsMasks(numVectors + 1);

   auto mapFunction = [&](const unsigned int i) {
      // set all vector values to zero
      std::vector<Double_v> gradFunc(npar);
      std::vector<Double_v> pointContributionVec(npar);

      Double_v x1, y, invError;

      Load(x1, data.GetCoordComponent(i * vecSize, 0));
      Load(y, data.ValuePtr(i * vecSize));
      const auto invErrorPtr = data.ErrorPtr(i * vecSize);

      if (invErrorPtr == nullptr)
         invError = 1;
      else
         Load(invError, invErrorPtr);

      // TODO: Check error options and invert if needed

      Double_v fval = 0;

      const Double_v *x = nullptr;

      unsigned int ndim = data.NDim();
      // need to declare vector outside if statement
      // otherwise pointer will be invalid
      std::vector<Double_v> xc;
      if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = x1;
         for (unsigned int j = 1; j < ndim; ++j)
            Load(xc[j], data.GetCoordComponent(i * vecSize, j));
         x = xc.data();
      } else {
         x = &x1;
      }

      fval = func(x, p);
      func.ParameterGradient(x, p, &gradFunc[0]);

      validPointsMasks[i] = CheckInfNaNValues(fval);
      if (MaskEmpty(validPointsMasks[i])) {
         // Return a zero contribution to all partial derivatives on behalf of the current points
         return pointContributionVec;
      }

      // loop on the parameters
      for (unsigned int ipar = 0; ipar < npar; ++ipar) {
         // avoid singularity in the function (infinity and nan ) in the chi2 sum
         // eventually add possibility of excluding some points (like singularity)
         validPointsMasks[i] = CheckInfNaNValues(gradFunc[ipar]);

         if (MaskEmpty(validPointsMasks[i])) {
            break; // exit loop on parameters
         }

         // calculate derivative point contribution (only for valid points)
         where(validPointsMasks[i], pointContributionVec[ipar]) =
            -2.0 * (y - fval) * invError * invError * gradFunc[ipar];
      }

      return pointContributionVec;
   };

   // Reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<Double_v>> &partialResults) {
      std::vector<Double_v> result(npar);

      for (auto const &pointContributionVec : partialResults) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContributionVec[parameterIndex];
      }

      return result;
   };

   std::vector<Double_v> gVec(npar);
   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateChi2Gradient", "Multithread execution policy requires IMT, which is disabled. Changing "
                                               "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      ROOT::TSequentialExecutor pool;
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(numVectors);
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction, chunks);
   }
#endif
   else {
      Error("FitUtil::EvaluateChi2Gradient",
            "Execution policy unknown. Available choices:\n 0: Serial (default)\n 1: MultiThread (requires IMT)\n");
   }

   // Compute the contribution from the remaining points
   unsigned int remainingPoints = initialNPoints % vecSize;
   if (remainingPoints > 0) {
      auto remainingPointsContribution = mapFunction(numVectors);
      // Add the contribution from the valid remaining points and store the result in the output variable
      auto remainingMask = Int2Mask(remainingPoints);
      for (unsigned int param = 0; param < npar; param++) {
         where(remainingMask, gVec[param]) = gVec[param] + remainingPointsContribution[param];
      }
   }
   // reduce final gradient result from T to double
   for (unsigned int param = 0; param < npar; param++) {
      grad[param] = ReduceAdd(gVec[param]);
   }

   // correct the number of points
   nPoints = initialNPoints;

   if (std::any_of(validPointsMasks.begin(), validPointsMasks.end(), [](auto validPoints) {
          // Check if the mask is not full
          for (size_t i = 0; i < Double_v::mask_type::size(); ++i)
             if (!validPoints[i])
                return true;
          return false;
       })) {
      unsigned nRejected = 0;

      for (const auto &mask : validPointsMasks) {
         for (unsigned int i = 0; i < vecSize; i++) {
            nRejected += !mask[i];
         }
      }

      assert(nRejected <= initialNPoints);
      nPoints = initialNPoints - nRejected;

      if (nPoints < npar) {
         MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient",
                        "Too many points rejected for overflow in gradient calculation");
      }
   }
}

void Evaluate<Double_v>::EvalPoissonLogLGradient(const IModelFunctionTempl<Double_v> &f, const BinData &data,
                                                 const double *p, double *grad, unsigned int &,
                                                 ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the gradient of the Poisson log likelihood function

   auto *fg = dynamic_cast<const IGradModelFunctionTempl<Double_v> *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   auto &func = *fg;

   (const_cast<IGradModelFunctionTempl<Double_v> &>(func)).SetParameters(p);

   const DataOptions &fitOpt = data.Opt();
   if (fitOpt.fBinVolume || fitOpt.fIntegral || fitOpt.fExpErrors)
      Error("FitUtil::EvaluatePoissonLogLGradient", "The vectorized implementation doesn't support Integrals,"
                                                    "BinVolume or ExpErrors\n. Aborting operation.");

   unsigned int npar = func.NPar();
   auto vecSize = Double_v::size();
   unsigned initialNPoints = data.Size();
   unsigned numVectors = initialNPoints / vecSize;

   auto mapFunction = [&](const unsigned int i) {
      // set all vector values to zero
      std::vector<Double_v> gradFunc(npar);
      std::vector<Double_v> pointContributionVec(npar);

      Double_v x1, y;

      Load(x1, data.GetCoordComponent(i * vecSize, 0));
      Load(y, data.ValuePtr(i * vecSize));

      Double_v fval = 0;

      const Double_v *x = nullptr;

      unsigned ndim = data.NDim();
      std::vector<Double_v> xc;
      if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = x1;
         for (unsigned int j = 1; j < ndim; ++j)
            Load(xc[j], data.GetCoordComponent(i * vecSize, j));
         x = xc.data();
      } else {
         x = &x1;
      }

      fval = func(x, p);
      func.ParameterGradient(x, p, &gradFunc[0]);

      // correct the gradient
      for (unsigned int ipar = 0; ipar < npar; ++ipar) {
         auto positiveValuesMask = fval > 0;

         // df/dp * (1.  - y/f )
         where(positiveValuesMask, pointContributionVec[ipar]) = gradFunc[ipar] * (1. - y / fval);

         auto validNegativeValuesMask = !positiveValuesMask && gradFunc[ipar] != 0;

         if (!MaskEmpty(validNegativeValuesMask)) {
            const Double_v kdmax1 = sqrt(NumericMax<Double_v>());
            const Double_v kdmax2 = NumericMax<Double_v>() / (4 * initialNPoints);
            Double_v gg = kdmax1 * gradFunc[ipar];
            auto mask = gg > 0;
            where(mask, pointContributionVec[ipar]) = min(gg, kdmax2);
            where(!mask, pointContributionVec[ipar]) = max(gg, -kdmax2);
            pointContributionVec[ipar] = -pointContributionVec[ipar];
         }
      }

#ifdef DEBUG_FITUTIL
      {
         R__LOCKGUARD(gROOTMutex);
         if (i < 5 || (i > data.Size() - 5)) {
            if (data.NDim() > 1)
               std::cout << i << "  x " << x[0] << " y " << x[1];
            else
               std::cout << i << "  x " << x[0];
            std::cout << " func " << fval << " gradient ";
            for (unsigned int ii = 0; ii < npar; ++ii)
               std::cout << "  " << pointContributionVec[ii];
            std::cout << "\n";
         }
      }
#endif

      return pointContributionVec;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<Double_v>> &partialResults) {
      std::vector<Double_v> result(npar);

      for (auto const &pointContributionVec : partialResults) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContributionVec[parameterIndex];
      }

      return result;
   };

   std::vector<Double_v> gVec(npar);

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluatePoissonLogLGradient",
              "Multithread execution policy requires IMT, which is disabled. Changing "
              "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      ROOT::TSequentialExecutor pool;
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(numVectors);
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction, chunks);
   }
#endif
   else {
      Error("FitUtil::EvaluatePoissonLogLGradient", "Execution policy unknown. Available choices:\n "
                                                    "::ROOT::EExecutionPolicy::kSequential (default)\n "
                                                    "::ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

   // Compute the contribution from the remaining points
   unsigned int remainingPoints = initialNPoints % vecSize;
   if (remainingPoints > 0) {
      auto remainingPointsContribution = mapFunction(numVectors);
      // Add the contribution from the valid remaining points and store the result in the output variable
      auto remainingMask = Int2Mask(remainingPoints);
      for (unsigned int param = 0; param < npar; param++) {
         where(remainingMask, gVec[param]) = gVec[param] + remainingPointsContribution[param];
      }
   }
   // reduce final gradient result from T to double
   for (unsigned int param = 0; param < npar; param++) {
      grad[param] = ReduceAdd(gVec[param]);
   }

#ifdef DEBUG_FITUTIL
   std::cout << "***** Final gradient : ";
   for (unsigned int ii = 0; ii < npar; ++ii)
      std::cout << grad[ii] << "   ";
   std::cout << "\n";
#endif
}

void Evaluate<Double_v>::EvalLogLGradient(const IModelFunctionTempl<Double_v> &f, const UnBinData &data,
                                          const double *p, double *grad, unsigned int &,
                                          ::ROOT::EExecutionPolicy executionPolicy, unsigned nChunks)
{
   // evaluate the gradient of the log likelihood function

   auto *fg = dynamic_cast<const IGradModelFunctionTempl<Double_v> *>(&f);
   assert(fg != nullptr); // must be called by a grad function

   auto &func = *fg;

   unsigned int npar = func.NPar();
   auto vecSize = Double_v::size();
   unsigned initialNPoints = data.Size();
   unsigned numVectors = initialNPoints / vecSize;

#ifdef DEBUG_FITUTIL
   std::cout << "\n===> Evaluate Gradient for parameters ";
   for (unsigned int ip = 0; ip < npar; ++ip)
      std::cout << "  " << p[ip];
   std::cout << "\n";
#endif

   (const_cast<IGradModelFunctionTempl<Double_v> &>(func)).SetParameters(p);

   const Double_v kdmax1 = sqrt(NumericMax<Double_v>());
   const Double_v kdmax2 = NumericMax<Double_v>() / (4 * initialNPoints);

   auto mapFunction = [&](const unsigned int i) {
      std::vector<Double_v> gradFunc(npar);
      std::vector<Double_v> pointContributionVec(npar);

      Double_v x1;
      Load(x1, data.GetCoordComponent(i * vecSize, 0));

      const Double_v *x = nullptr;

      unsigned int ndim = data.NDim();
      std::vector<Double_v> xc(ndim);
      if (ndim > 1) {
         xc.resize(ndim);
         xc[0] = x1;
         for (unsigned int j = 1; j < ndim; ++j)
            Load(xc[j], data.GetCoordComponent(i * vecSize, j));
         x = xc.data();
      } else {
         x = &x1;
      }

      Double_v fval = func(x, p);
      func.ParameterGradient(x, p, &gradFunc[0]);

#ifdef DEBUG_FITUTIL
      if (i < 5 || (i > numVectors - 5)) {
         if (ndim > 1)
            std::cout << i << "  x " << x[0] << " y " << x[1] << " gradient " << gradFunc[0] << "  " << gradFunc[1]
                      << "  " << gradFunc[3] << std::endl;
         else
            std::cout << i << "  x " << x[0] << " gradient " << gradFunc[0] << "  " << gradFunc[1] << "  "
                      << gradFunc[3] << std::endl;
      }
#endif

      auto positiveValues = fval > 0;

      for (unsigned int kpar = 0; kpar < npar; ++kpar) {
         if (!MaskEmpty(positiveValues)) {
            where(positiveValues, pointContributionVec[kpar]) = -1. / fval * gradFunc[kpar];
         }

         auto nonZeroGradientValues = !positiveValues && gradFunc[kpar] != 0;
         if (!MaskEmpty(nonZeroGradientValues)) {
            Double_v gg = kdmax1 * gradFunc[kpar];
            auto mask = nonZeroGradientValues && gg > 0;
            where(mask, pointContributionVec[kpar]) = -min(gg, kdmax2);
            where(!mask, pointContributionVec[kpar]) = -max(gg, -kdmax2);
         }
         // if func derivative is zero term is also zero so do not add in g[kpar]
      }

      return pointContributionVec;
   };

   // Vertically reduce the set of vectors by summing its equally-indexed components
   auto redFunction = [&](const std::vector<std::vector<Double_v>> &pointContributions) {
      std::vector<Double_v> result(npar);

      for (auto const &pointContributionVec : pointContributions) {
         for (unsigned int parameterIndex = 0; parameterIndex < npar; parameterIndex++)
            result[parameterIndex] += pointContributionVec[parameterIndex];
      }

      return result;
   };

   std::vector<Double_v> gVec(npar);
   std::vector<double> g(npar);

#ifndef R__USE_IMT
   // to fix compiler warning
   (void)nChunks;

   // If IMT is disabled, force the execution policy to the serial case
   if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      Warning("FitUtil::EvaluateLogLGradient", "Multithread execution policy requires IMT, which is disabled. Changing "
                                               "to ::ROOT::EExecutionPolicy::kSequential.");
      executionPolicy = ::ROOT::EExecutionPolicy::kSequential;
   }
#endif

   if (executionPolicy == ::ROOT::EExecutionPolicy::kSequential) {
      ROOT::TSequentialExecutor pool;
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction);
   }
#ifdef R__USE_IMT
   else if (executionPolicy == ::ROOT::EExecutionPolicy::kMultiThread) {
      ROOT::TThreadExecutor pool;
      auto chunks = nChunks != 0 ? nChunks : setAutomaticChunking(numVectors);
      gVec = pool.MapReduce(mapFunction, ROOT::TSeq<unsigned>(0, numVectors), redFunction, chunks);
   }
#endif
   else {
      Error("FitUtil::EvaluateLogLGradient", "Execution policy unknown. Available choices:\n "
                                             "::ROOT::EExecutionPolicy::kSequential (default)\n "
                                             "::ROOT::EExecutionPolicy::kMultiThread (requires IMT)\n");
   }

   // Compute the contribution from the remaining points
   unsigned int remainingPoints = initialNPoints % vecSize;
   if (remainingPoints > 0) {
      auto remainingPointsContribution = mapFunction(numVectors);
      // Add the contribution from the valid remaining points and store the result in the output variable
      auto remainingMask = Int2Mask(initialNPoints % vecSize);
      for (unsigned int param = 0; param < npar; param++) {
         where(remainingMask, gVec[param]) = gVec[param] + remainingPointsContribution[param];
      }
   }
   // reduce final gradient result from T to double
   for (unsigned int param = 0; param < npar; param++) {
      grad[param] = ReduceAdd(gVec[param]);
   }

#ifdef DEBUG_FITUTIL
   std::cout << "Final gradient ";
   for (unsigned int param = 0; param < npar; param++) {
      std::cout << "  " << grad[param];
   }
   std::cout << "\n";
#endif
}

} // namespace FitUtil

#endif // R__HAS_STD_EXPERIMENTAL_SIMD
}

} // end namespace ROOT
