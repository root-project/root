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
//#include "Fit/BinPoint.h"

#include "Math/IParamFunction.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/WrappedFunction.h"
#include "Math/OneDimFunctionAdapter.h"
#include "Math/RichardsonDerivator.h"

#include "Math/Error.h"
#include "Math/Util.h"  // for safe log(x)

#include <limits>
#include <cmath>
#include <cassert> 
//#include <memory>

//#define DEBUG
#ifdef DEBUG
#define NSAMPLE 10
#include <iostream> 
#endif

//todo: 

//  need to implement integral option

namespace ROOT { 

   namespace Fit { 

      namespace FitUtil { 

         // internal class to evaluate the function or the integral 
         // and cached internal integration details
         // if useIntegral is false no allocation is done
         // and this is a dummy class
         // class is templated on any parametric functor implementing operator()(x,p) and NDim()
         // contains a constant pointer to the function
         template <class ParamFunc = ROOT::Math::IParamMultiFunction> 
         class IntegralEvaluator { 

         public: 

            IntegralEvaluator(const ParamFunc & func, const double * p, bool useIntegral = true) :
               fDim(0),
               fParams(0),
               fFunc(0),
               fIg1Dim(0), 
               fIgNDim(0),
               fFunc1Dim(0),
               fFuncNDim(0)
            { 
               if (useIntegral) { 
                  SetFunction(func, p); 
               }
            }

            void SetFunction(const ParamFunc & func, const double * p = 0) { 
               // set the integrand function and create required wrapper 
               // to perform integral in (x) of a generic  f(x,p)
               fParams = p;
               fDim = func.NDim(); 
               // copy the function object to be able to modify the parameters 
               //fFunc = dynamic_cast<ROOT::Math::IParamMultiFunction *>( func.Clone() ); 
               fFunc = &func;
               assert(fFunc != 0); 
               // set parameters in function
               //fFunc->SetParameters(p); 
               if (fDim == 1) { 
                  fFunc1Dim = new ROOT::Math::WrappedMemFunction< IntegralEvaluator, double (IntegralEvaluator::*)(double ) const > (*this, &IntegralEvaluator::F1);
                  fIg1Dim = new ROOT::Math::IntegratorOneDim(); 
                  //fIg1Dim->SetFunction( static_cast<const ROOT::Math::IMultiGenFunction & >(*fFunc),false);
                  fIg1Dim->SetFunction( static_cast<const ROOT::Math::IGenFunction &>(*fFunc1Dim) );
               } 
               else if (fDim > 1) {
                  fFuncNDim = new ROOT::Math::WrappedMemMultiFunction< IntegralEvaluator, double (IntegralEvaluator::*)(const double *) const >  (*this, &IntegralEvaluator::FN, fDim);
                  fIgNDim = new ROOT::Math::IntegratorMultiDim();
                  fIgNDim->SetFunction(*fFuncNDim);
               }
               else
                  assert(fDim > 0);                
            }
            
            void SetParameters(const double *p) { 
               // copy just the pointer
               fParams = p; 
            }

            ~IntegralEvaluator() { 
               if (fIg1Dim) delete fIg1Dim; 
               if (fIgNDim) delete fIgNDim; 
               if (fFunc1Dim) delete fFunc1Dim; 
               if (fFuncNDim) delete fFuncNDim; 
               //if (fFunc) delete fFunc; 
            }
            
            // evaluation of integrand function (one-dim)
            double F1 (double x) const { 
               double xx[1]; xx[0] = x;
               return (*fFunc)( xx, fParams);
            }
            // evaluation of integrand function (multi-dim)
            double FN(const double * x) const { 
               return (*fFunc)( x, fParams);
            }

            double operator()(const double *x1, const double * x2) { 
               // return normalized integral, divided by bin volume (dx1*dx...*dxn) 
               if (fIg1Dim) { 
                  double dV = *x2 - *x1;
                  return fIg1Dim->Integral( *x1, *x2)/dV; 
               }
               else if (fIgNDim) { 
                  double dV = 1; 
                  for (unsigned int i = 0; i < fDim; ++i) 
                     dV *= ( x2[i] - x1[i] );                   
                  return fIgNDim->Integral( x1, x2)/dV; 
//                   std::cout << " do integral btw x " << x1[0] << "  " << x2[0] << " y " << x1[1] << "  " << x2[1] << " dV = " << dV << " result = " << result << std::endl;
//                   return result; 
               }
               else 
                  assert(1.); // should never be here
               return 0;
            }

         private: 

            unsigned int fDim; 
            const double * fParams;
            //ROOT::Math::IParamMultiFunction * fFunc;  // copy of function in order to be able to change parameters    
            const ParamFunc * fFunc;       //  reference to a generic parametric function  
            ROOT::Math::IntegratorOneDim * fIg1Dim; 
            ROOT::Math::IntegratorMultiDim * fIgNDim; 
            ROOT::Math::IGenFunction * fFunc1Dim; 
            ROOT::Math::IMultiGenFunction * fFuncNDim; 
         }; 


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
         bool CheckValue(double & rval) { 
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

double FitUtil::EvaluateChi2(const IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints) {  
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints 
   // the actual number of used points
   // normal chi2 using only error on values (from fitting histogram)
   // optionally the integral of function in the bin is used 
   
   unsigned int n = data.Size();

 
#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function " << &func << "  " << p << std::endl; 
#endif


   double chi2 = 0;
   //int nRejected = 0; 
   
   // do not cache parameter values (it is not thread safe)
   //func.SetParameters(p); 

   
   // get fit option and check case if using integral of bins
   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   IntegralEvaluator<> igEval( func, p, useBinIntegral); 

   double maxResValue = std::numeric_limits<double>::max() /n;
   double wrefVolume = 1.0; 
   std::vector<double> xc; 
   if (useBinVolume) { 
      wrefVolume /= data.RefVolume();
      xc.resize(data.NDim() );
   }

   for (unsigned int i = 0; i < n; ++ i) { 


      double y, invError; 
      // in case of no error in y invError=1 is returned
      const double * x1 = data.GetPoint(i,y, invError);

      double fval = 0;

      double binVolume = 1.0; 
      if (useBinVolume) { 
         unsigned int ndim = data.NDim(); 
         const double * x2 = data.BinUpEdge(i);  
         for (unsigned int j = 0; j < ndim; ++j) {
            binVolume *= std::abs( x2[j]-x1[j] );
            xc[j] = 0.5*(x2[j]+ x1[j]);
         }
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      }

      const double * x = (useBinVolume) ? &xc.front() : x1;

      if (!useBinIntegral) {
         fval = func ( x, p );
      }
      else {
         // calculate integral normalized by bin volume
         // need to set function and parameters here in case loop is parallelized
         fval = igEval( x1, data.BinUpEdge(i)) ; 
      }
      // normalize result if requested according to bin volume
      if (useBinVolume) fval *= binVolume;

//#define DEBUG
#ifdef DEBUG      
      std::cout << x[0] << "  " << y << "  " << 1./invError << " params : "; 
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar) 
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << " bin volume " << binVolume << " ref " << wrefVolume << std::endl; 
#endif
//#undef DEBUG


      double tmp = ( y -fval )* invError;  	  
      double resval = tmp * tmp;


      // avoid inifinity or nan in chi2 values due to wrong function values 
      if ( resval < maxResValue )  
         chi2 += resval; 
      else {  
         //nRejected++; 
         chi2 += maxResValue;
      }

      
   }
   
   // reset the number of fitting data points
   nPoints = n;  // no points are rejected 
   //if (nRejected != 0)  nPoints = n - nRejected;   

#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nPoints  /*<< " rejected = " << nRejected */ << std::endl;
#endif


   return chi2;
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

   assert(data.HaveCoordErrors() ); 

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
      const double * ex = 0; 
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
         // select optimal step size  (use 10--3 by default as was done in TF1:
         double kEps = 0.001;
         double kPrecision = 1.E-8;
         for (unsigned int icoord = 0; icoord < ndim; ++icoord) { 
            // calculate derivative for each coordinate 
            if (ex[icoord] > 0) {               
               //gradCalc.Gradient(x, p, fval, &grad[0]);
               f1D.SetCoord(icoord);
               // optimal spep size (maybe should take average or  range in points) 
               double x0= x[icoord];
               double h = std::max( kEps* std::abs(x0), 8.0*kPrecision*(std::abs(x0) + kPrecision) );
               double deriv = derivator.Derivative1(f1D, x[icoord], h);  
               double edx = ex[icoord] * deriv; 
               e2 += edx * edx;  
            }
         } 
      }
      double w2 = (e2 > 0) ? 1.0/e2 : 0;  
      double resval = w2 * ( y - fval ) *  ( y - fval); 

#ifdef DEBUG      
      std::cout << x[0] << "  " << y << " ex " << e2 << " ey  " << ey << " params : "; 
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
   std::cout << "chi2 = " << chi2 << " n = " << nPoints << /* " rejected = " << nRejected */ << std::endl;
#endif

   return chi2;

}


//___________________________________________________________________________________________________________________________
double FitUtil::EvaluateChi2Residual(const IModelFunction & func, const BinData & data, const double * p, unsigned int i, double * g) {  
   // evaluate the chi2 contribution (residual term) only for data with no coord-errors
   // This function is used in the specialized least square algorithms like FUMILI or L.M.
   // if we have error on the coordinates the method is not yet implemented 
   //  integral option is also not yet implemented
   //  one can use in that case normal chi2 method
  
   if (data.GetErrorType() == BinData::kCoordError && data.Opt().fCoordErrors ) { 
      MATH_ERROR_MSG("FitUtil::EvaluateChi2Residual","Error on the coordinates are not used in calculating Chi2 residual");
      return 0; // it will assert otherwise later in GetPoint
   }


   //func.SetParameters(p);

   double y, invError = 0; 

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   const double * x1 = data.GetPoint(i,y, invError);

   IntegralEvaluator<> igEval( func, p, useBinIntegral); 
   double fval = 0; 
   unsigned int ndim = data.NDim(); 
   double binVolume = 1.0; 
   const double * x2 = 0; 
   if (useBinVolume || useBinIntegral) x2 = data.BinUpEdge(i);  

   double * xc = 0;
 
   if (useBinVolume) { 
      xc = new double[ndim];
      for (unsigned int j = 0; j < ndim; ++j) {
         binVolume *= std::abs( x2[j]-x1[j] );
         xc[j] = 0.5*(x2[j]+ x1[j]);         
      }
      // normalize the bin volume using a reference value
      binVolume /= data.RefVolume();
   }

   const double * x = (useBinVolume) ? xc : x1;

   if (!useBinIntegral) {
      fval = func ( x, p );
   }
   else {
      // calculate integral (normalized by bin volume) 
      // need to set function and parameters here in case loop is parallelized
      fval = igEval( x1, x2) ; 
   }
   // normalize result if requested according to bin volume
   if (useBinVolume) fval *= binVolume;

   double resval =   ( y -fval )* invError;  	   

   // avoid infinities or nan in  resval
   resval = CorrectValue(resval);
   
   // estimate gradient 
   if (g != 0) {  

      unsigned int npar = func.NPar(); 
      const IGradModelFunction * gfunc = dynamic_cast<const IGradModelFunction *>( &func); 

      if (gfunc != 0) { 
         //case function provides gradient
         if (!useBinIntegral ) {
            gfunc->ParameterGradient(  x , p, g );  
         }
         else { 
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral( *gfunc, x1, x2, p, g); 
         }
      }
      else { 
         SimpleGradientCalculator  gc( npar, func); 
         if (!useBinIntegral ) 
            gc.ParameterGradient(x, p, fval, g); 
         else { 
            // needs to calculate the integral for each partial derivative
            CalculateGradientIntegral( gc, x1, x2, p, g); 
         }
      }
      // mutiply by - 1 * weight
      for (unsigned int k = 0; k < npar; ++k) {
         g[k] *= - invError;
         if (useBinVolume) g[k] *= binVolume;      
      }       
   }

   if (useBinVolume) delete [] xc;

   return resval; 

}

void FitUtil::EvaluateChi2Gradient(const IModelFunction & f, const BinData & data, const double * p, double * grad, unsigned int & nPoints) { 
   // evaluate the gradient of the chi2 function
   // this function is used when the model function knows how to calculate the derivative and we can  
   // avoid that the minimizer re-computes them 
   //
   // case of chi2 effective (errors on coordinate) is not supported

   if ( data.HaveCoordErrors() ) {
      MATH_ERROR_MSG("FitUtil::EvaluateChi2Residual","Error on the coordinates are not used in calculating Chi2 gradient");            return; // it will assert otherwise later in GetPoint
   }

   unsigned int nRejected = 0; 

   const IGradModelFunction * fg = dynamic_cast<const IGradModelFunction *>( &f); 
   assert (fg != 0); // must be called by a gradient function

   const IGradModelFunction & func = *fg; 
   unsigned int n = data.Size();


#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function gradient " << &func << "  " << p << std::endl; 
#endif

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0; 
   std::vector<double> xc; 
   if (useBinVolume) { 
      wrefVolume /= data.RefVolume();
      xc.resize(data.NDim() );
   }

   IntegralEvaluator<> igEval( func, p, useBinIntegral); 

   //int nRejected = 0; 
   // set values of parameters 

   unsigned int npar = func.NPar(); 
   //   assert (npar == NDim() );  // npar MUST be  Chi2 dimension
   std::vector<double> gradFunc( npar ); 
   // set all vector values to zero
   std::vector<double> g( npar); 

   for (unsigned int i = 0; i < n; ++ i) { 


      double y, invError = 0; 
      const double * x1 = data.GetPoint(i,y, invError);

      double fval = 0; 
      const double * x2 = 0; 

      double binVolume = 1; 
      if (useBinVolume) { 
         unsigned int ndim = data.NDim(); 
         x2 = data.BinUpEdge(i);  
         for (unsigned int j = 0; j < ndim; ++j) {
            binVolume *= std::abs( x2[j]-x1[j] );
            xc[j] = 0.5*(x2[j]+ x1[j]);
         }
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      }

      const double * x = (useBinVolume) ? &xc.front() : x1;

      if (!useBinIntegral ) {
         fval = func ( x, p ); 
         func.ParameterGradient(  x , p, &gradFunc[0] ); 
      }
      else { 
         x2 = data.BinUpEdge(i); 
         // calculate normalized integral and gradient (divided by bin volume)
         // need to set function and parameters here in case loop is parallelized 
         fval = igEval( x1, x2 ) ; 
         CalculateGradientIntegral( func, x1, x2, p, &gradFunc[0]); 
      }
      if (useBinVolume) fval *= binVolume;

#ifdef DEBUG      
      std::cout << x[0] << "  " << y << "  " << 1./invError << " params : "; 
      for (unsigned int ipar = 0; ipar < npar; ++ipar) 
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << std::endl; 
#endif
      if ( !CheckValue(fval) ) { 
         nRejected++; 
         continue;
      } 

      // loop on the parameters
      unsigned int ipar = 0; 
      for ( ; ipar < npar ; ++ipar) { 

         // correct gradient for bin volumes
         if (useBinVolume) gradFunc[ipar] *= binVolume;

         // avoid singularity in the function (infinity and nan ) in the chi2 sum 
         // eventually add possibility of excluding some points (like singularity) 
         double dfval = gradFunc[ipar];
         if ( !CheckValue(dfval) ) { 
               break; // exit loop on parameters
         } 
 
         // calculate derivative point contribution
         double tmp = - 2.0 * ( y -fval )* invError * invError * gradFunc[ipar];  	  
         g[ipar] += tmp;
      
      }

      if ( ipar < npar ) { 
          // case loop was broken for an overflow in the gradient calculation  
         nRejected++; 
         continue;
      } 


   } 

   // correct the number of points
   nPoints = n; 
   if (nRejected != 0)  {
      assert(nRejected <= n);
      nPoints = n - nRejected;
      if (nPoints < npar)  MATH_ERROR_MSG("FitUtil::EvaluateChi2Gradient","Error - too many points rejected for overflow in gradient calculation");
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

double FitUtil::EvaluatePdf(const IModelFunction & func, const UnBinData & data, const double * p, unsigned int i, double * g) {  
   // evaluate the pdf contribution to the generic logl function in case of bin data
   // return actually the log of the pdf and its derivatives 

   //func.SetParameters(p);


   const double * x = data.Coords(i);
   double fval = func ( x, p ); 
   double logPdf = ROOT::Math::Util::EvalLog(fval);
   //return 
   if (g == 0) return logPdf;

   const IGradModelFunction * gfunc = dynamic_cast<const IGradModelFunction *>( &func); 

   // gradient  calculation
   if (gfunc != 0) { 
      //case function provides gradient
      gfunc->ParameterGradient(  x , p, g );  
   }
   else { 
      // estimate gradieant numerically with simple 2 point rule 
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

double FitUtil::EvaluateLogL(const IModelFunction & func, const UnBinData & data, const double * p, unsigned int &nPoints) {  
   // evaluate the LogLikelihood 

   unsigned int n = data.Size();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "func pointer is " << typeid(func).name() << std::endl;
#endif

   double logl = 0;
   //unsigned int nRejected = 0; 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x = data.Coords(i);
      double fval = func ( x, p ); 

#ifdef DEBUG      
      std::cout << "x [ " << data.PointSize() << " ] = "; 
      for (unsigned int j = 0; j < data.PointSize(); ++j)
         std::cout << x[j] << "\t"; 
      std::cout << "\tpar = [ " << func.NPar() << " ] =  "; 
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar) 
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << std::endl; 
#endif
      // function EvalLog protects against negative or too small values of fval
      logl += ROOT::Math::Util::EvalLog( fval);       
   }
   
   // reset the number of fitting data points
   nPoints = n; 
//    if (nRejected != 0)  { 
//       assert(nRejected <= n);
//       nPoints = n - nRejected;
//       if ( nPoints < func.NPar() ) 
//          MATH_ERROR_MSG("FitUtil::EvaluateLogL","Error too many points rejected because of bad pdf values");

//    }

#ifdef DEBUG
   std::cout << "Logl = " << logl << " np = " << nPoints << std::endl;
#endif
   
   return -logl;
}

void FitUtil::EvaluateLogLGradient(const IModelFunction & f, const UnBinData & data, const double * p, double * grad, unsigned int & ) { 
   // evaluate the gradient of the log likelihood function

   const IGradModelFunction * fg = dynamic_cast<const IGradModelFunction *>( &f); 
   assert (fg != 0); // must be called by a grad function
   const IGradModelFunction & func = *fg; 

   unsigned int n = data.Size();
   //int nRejected = 0; 

   unsigned int npar = func.NPar(); 
   std::vector<double> gradFunc( npar ); 
   std::vector<double> g( npar); 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x = data.Coords(i);
      double fval = func ( x , p); 
      func.ParameterGradient( x, p, &gradFunc[0] );
      for (unsigned int kpar = 0; kpar < npar; ++ kpar) { 
         if (fval > 0)  
            g[kpar] -= 1./fval * gradFunc[ kpar ]; 
         else if (gradFunc [ kpar] != 0) { 
            const double kdmax1 = std::sqrt( std::numeric_limits<double>::max() );
            const double kdmax2 = std::numeric_limits<double>::max() / (4*n);
            double gg = kdmax1 * gradFunc[ kpar ];  
            if ( gg > 0) gg = std::min( gg, kdmax2);
            else gg = std::max(gg, - kdmax2);
            g[kpar] -= gg;
         }
         // if func derivative is zero term is also zero so do not add in g[kpar]
      }
            
    // copy result 
   std::copy(g.begin(), g.end(), grad);
   }
}
//_________________________________________________________________________________________________
// for binned log likelihood functions      
//------------------------------------------------------------------------------------------------
double FitUtil::EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * p, unsigned int i, double * g ) {  
   // evaluate the pdf (Poisson) contribution to the logl (return actually log of pdf)
   // and its gradient


   double y = 0; 
   const double * x1 = data.GetPoint(i,y);

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   IntegralEvaluator<> igEval( func, p, useBinIntegral); 
   double fval = 0; 
   const double * x2 = 0; 
   // calculate the bin volume
   double binVolume = 1; 
   std::vector<double> xc; 
   if (useBinVolume) { 
      unsigned int ndim = data.NDim(); 
      xc.resize(ndim);
      x2 = data.BinUpEdge(i);  
      for (unsigned int j = 0; j < ndim; ++j) {
         binVolume *= std::abs( x2[j]-x1[j] );
         xc[j] = 0.5*(x2[j]+ x1[j]);
      }
      // normalize the bin volume using a reference value
      binVolume /= data.RefVolume();
   }

   const double * x = (useBinVolume) ? &xc.front() : x1;

   if (!useBinIntegral ) {
      fval = func ( x, p ); 
   }
   else { 
      // calculate integral normalized (divided by bin volume) 
      x2 =  data.BinUpEdge(i);
      fval = igEval( x1, x2 ) ; 
   }
   if (useBinVolume) fval *= binVolume;

   // logPdf for Poisson: ignore constant term depending on N
   fval = std::max(fval, 0.0);  // avoid negative or too small values 
   double logPdf =   y * ROOT::Math::Util::EvalLog( fval) - fval;  
   // need to return the pdf contribution (not the log)

   //double pdfval =  std::exp(logPdf);

  //if (g == 0) return pdfval; 
   if (g == 0) return logPdf; 

   unsigned int npar = func.NPar(); 
   const IGradModelFunction * gfunc = dynamic_cast<const IGradModelFunction *>( &func); 

   // gradient  calculation
   if (gfunc != 0) { 
      //case function provides gradient
      if (!useBinIntegral ) 
         gfunc->ParameterGradient(  x , p, g );  
      else { 
         // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral( *gfunc, x1, x2, p, g); 
      }

   }
   else {       
      SimpleGradientCalculator  gc(func.NPar(), func); 
      if (!useBinIntegral ) 
         gc.ParameterGradient(x, p, fval, g); 
      else { 
        // needs to calculate the integral for each partial derivative
         CalculateGradientIntegral( gc, x1, x2, p, g);  
      } 
   }
   // correct g[] do be derivative of poisson term 
   for (unsigned int k = 0; k < npar; ++k) {
      // apply bin volume correction
      if (useBinVolume) g[k] *= binVolume;  

      // correct for Poisson term
      if ( fval > 0) 
         g[k] *= ( y/fval - 1.) ;//* pdfval; 
      else if (y > 0) { 
         const double kdmax1 = std::sqrt( std::numeric_limits<double>::max() );
         g[k] *= kdmax1; 
      }
      else   // y == 0 cannot have  negative y
         g[k] *= -1;         
   }       
     

#ifdef DEBUG
   std::cout << "x = " << x[0] << " logPdf = " << logPdf << " grad"; 
   for (unsigned int ipar = 0; ipar < npar; ++ipar) 
      std::cout << g[ipar] << "\t";
   std::cout << std::endl;
#endif 

//   return pdfval;
   return logPdf;
}

double FitUtil::EvaluatePoissonLogL(const IModelFunction & func, const BinData & data, const double * p, unsigned int &   nPoints ) {  
   // evaluate the Poisson Log Likelihood
   // for binned likelihood fits
   // this is Sum ( f(x_i)  -  y_i * log( f (x_i) ) )

   unsigned int n = data.Size();
#ifdef DEBUG
   std::cout << "Evaluate PoissonLogL for params = [ "; 
   for (unsigned int j=0; j < func.NPar(); ++j) std::cout << p[j] << " , ";
   std::cout << "]  - data size = " << n << std::endl;
#endif
   
   double loglike = 0;
   //int nRejected = 0; 

   // get fit option and check case of using integral of bins
   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0; 
   std::vector<double> xc; 
   if (useBinVolume) { 
      wrefVolume /= data.RefVolume();
      xc.resize(data.NDim() );
   }

   IntegralEvaluator<> igEval( func, p, fitOpt.fIntegral); 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x1 = data.Coords(i);
      double y = data.Value(i);

      double fval = 0;   
      double binVolume = 1.0; 

      if (useBinVolume) { 
         unsigned int ndim = data.NDim(); 
         const double * x2 = data.BinUpEdge(i);  
         for (unsigned int j = 0; j < ndim; ++j) {
            binVolume *= std::abs( x2[j]-x1[j] );
            xc[j] = 0.5*(x2[j]+ x1[j]);
         }
         // normalize the bin volume using a reefrence value
         binVolume *= wrefVolume;
      }

      const double * x = (useBinVolume) ? &xc.front() : x1;

      if (!useBinIntegral) {
         fval = func ( x, p );
      }
      else {
         // calculate integral (normalized by bin volume) 
         // need to set function and parameters here in case loop is parallelized
         fval = igEval( x1, data.BinUpEdge(i)) ; 
      }
      if (useBinVolume) fval *= binVolume;


#ifdef DEBUG
      int NSAMPLE = 100;
      if (i%NSAMPLE == 0) { 
         std::cout << "evt " << i << " x1 = [ "; 
         for (unsigned int j=0; j < func.NDim(); ++j) std::cout << x[j] << " , ";
         std::cout << "]  ";
         if (fitOpt.fIntegral) { 
            std::cout << "x2 = [ "; 
            for (unsigned int j=0; j < func.NDim(); ++j) std::cout << data.BinUpEdge(i)[j] << " , ";
            std::cout << "] ";
         }
         std::cout << "  y = " << y << " fval = " << fval << std::endl;
      }
#endif


      // EvalLog protects against 0 values of fval but don't want to add in the -log sum 
      // negative values of fval 
      fval = std::max(fval, 0.0);
      loglike +=  fval - y * ROOT::Math::Util::EvalLog( fval);  

   }
   
   // reset the number of fitting data points
   nPoints = n; 
   //if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "Loglikelihood  = " << loglike << std::endl;
#endif
   
   return loglike;  
}

void FitUtil::EvaluatePoissonLogLGradient(const IModelFunction & f, const BinData & data, const double * p, double * grad ) { 
   // evaluate the gradient of the Poisson log likelihood function

   const IGradModelFunction * fg = dynamic_cast<const IGradModelFunction *>( &f); 
   assert (fg != 0); // must be called by a grad function
   const IGradModelFunction & func = *fg; 

   unsigned int n = data.Size();

   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral && data.HasBinEdges(); 
   bool useBinVolume = (fitOpt.fBinVolume && data.HasBinEdges());

   double wrefVolume = 1.0;
   std::vector<double> xc;  
   if (useBinVolume) {
      wrefVolume /= data.RefVolume();
      xc.resize(data.NDim() );
   }

   IntegralEvaluator<> igEval( func, p, useBinIntegral); 

   unsigned int npar = func.NPar(); 
   std::vector<double> gradFunc( npar ); 
   std::vector<double> g( npar); 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x1 = data.Coords(i);
      double y = data.Value(i);
      double fval = 0; 
      const double * x2 = 0; 

      double binVolume = 1.0; 
      if (useBinVolume) { 
         x2 = data.BinUpEdge(i);  
         unsigned int ndim = data.NDim(); 
         for (unsigned int j = 0; j < ndim; ++j) { 
            binVolume *= std::abs( x2[j]-x1[j] );
            xc[j] = 0.5*(x2[j]+ x1[j]);
         }
         // normalize the bin volume using a reference value
         binVolume *= wrefVolume;
      }

      const double * x = (useBinVolume) ? &xc.front() : x1;

      if (!useBinIntegral) {
         fval = func ( x, p );
         func.ParameterGradient(  x , p, &gradFunc[0] ); 
      }
      else {
         // calculate integral (normalized by bin volume) 
         // need to set function and parameters here in case loop is parallelized
         x2 = data.BinUpEdge(i);
         fval = igEval( x1, x2) ; 
         CalculateGradientIntegral( func, x1, x2, p, &gradFunc[0]); 
      }
      if (useBinVolume) fval *= binVolume;
      
      // correct the gradient
      for (unsigned int kpar = 0; kpar < npar; ++ kpar) { 

         // correct gradient for bin volumes
         if (useBinVolume) gradFunc[kpar] *= binVolume; 

         // df/dp * (1.  - y/f )
         if (fval > 0)  
            g[kpar] += gradFunc[ kpar ] * ( 1. - y/fval ); 
         else if (gradFunc [ kpar] != 0) { 
            const double kdmax1 = std::sqrt( std::numeric_limits<double>::max() );
            const double kdmax2 = std::numeric_limits<double>::max() / (4*n);
            double gg = kdmax1 * gradFunc[ kpar ];  
            if ( gg > 0) gg = std::min( gg, kdmax2);
            else gg = std::max(gg, - kdmax2);
            g[kpar] -= gg;
         }
      }            

      // copy result 
      std::copy(g.begin(), g.end(), grad);
   }
}
   
}

} // end namespace ROOT

