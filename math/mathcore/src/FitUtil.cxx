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

#include "Math/Error.h"
#include "Math/Util.h"  // for safe log(x)

#include <limits>
#include <cmath>
#include <cassert> 
//#include <memory>

//#define DEBUG
#ifdef DEBUG
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
         struct IntegralEvaluator { 

            IntegralEvaluator(const ROOT::Math::IParamMultiFunction & func, const double * p, bool useIntegral = true) :
               fDim(0),
               fFunc(0),
               fIg1Dim(0), 
               fIgNDim(0) 
            { 
               if (useIntegral) { 
                  // copy the function object to be able to modify the parameters 
                  fDim = func.NDim(); 
                  fFunc = dynamic_cast<ROOT::Math::IParamMultiFunction *>( func.Clone() ); 
                  assert(fFunc != 0); 
                  // set parameters in function
                  fFunc->SetParameters(p); 
                  if (fDim == 1) { 
                     fIg1Dim = new ROOT::Math::IntegratorOneDim(); 
                     fIg1Dim->SetFunction( static_cast<const ROOT::Math::IMultiGenFunction & >(*fFunc),false);
                  } 
                  else if (fDim > 1) {
                     fIgNDim = new ROOT::Math::IntegratorMultiDim();
                     fIgNDim->SetFunction(*fFunc);
                  }
                  else
                     assert(fDim > 0); 
               }
            }

            ~IntegralEvaluator() { 
               if (fIg1Dim) delete fIg1Dim; 
               if (fIgNDim) delete fIgNDim; 
               if (fFunc) delete fFunc; 
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
               }
               else 
                  assert(1.); // should never be here
               return 0;
            }

            unsigned int fDim; 
            ROOT::Math::IParamMultiFunction * fFunc;  // copy of function in order to be able to change parameters    
            ROOT::Math::IntegratorOneDim * fIg1Dim; 
            ROOT::Math::IntegratorMultiDim * fIgNDim; 
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


            // calculate gradient at point (x,p) knnowing already value f0 (we gain a function eval.)
            void ParameterGradient(const double * x, const double * p, double f0, double * g) { 
               // fVec are the cached parameter values
               std::copy(p, p+fN, fVec.begin()); 
               for (unsigned int k = 0; k < fN; ++k) {
                  double p0 = p[k];
                  double h = std::max( fEps* std::abs(p0), 8.0*fPrecision*(std::abs(p0) + fPrecision) );
                  fVec[k] += h;
                  // t.b.d : treat case of infinities 
                  //if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() ) 
                  double f1 = fFunc(x, &fVec.front() );
                  if (fStrategy > 1) { 
                     fVec[k] = p0 - h; 
                     double f2 = fFunc(x, &fVec.front() );
                     g[k] = 0.5 * ( f2 - f1 )/h;
                  }
                  else 
                     g[k] = ( f1 - f0 )/h;

                  fVec[k] = p[k]; // restore original p value
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
            std::vector<double> fVec; // cached coordinates (or parameter values in case of gradientpar)
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

      } // end namespace  FitUtil      


//___________________________________________________________________________________________________________________________
// for chi2 functions
//___________________________________________________________________________________________________________________________

double FitUtil::EvaluateChi2(const IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints) {  
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints 
   // the actual number of used points
   // normal chi2 using only error on values (from fitting histogram)
   // optionally intergal of function in the bin is used 
   
   unsigned int n = data.Size();

 
#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   std::cout << "evaluate chi2 using function " << &func << "  " << p << std::endl; 
#endif

   double chi2 = 0;
   int nRejected = 0; 
   
   // do not cache parameter values (it is not thread safe)
   //func.SetParameters(p); 

   
   // get fit option and check case if using integral of bins
   const DataOptions & fitOpt = data.Opt();
   bool useBinIntegral = fitOpt.fIntegral; 
   IntegralEvaluator igEval( func, p, useBinIntegral); 


   for (unsigned int i = 0; i < n; ++ i) { 


      double y, invError; 
      // in case of no error in y invError=1 is returned
      const double * x = data.GetPoint(i,y, invError);


      double fval = 0;

      if (!useBinIntegral ) 
         fval = func ( x, p ); 
      else { 
         // calculate normalized integral (divided by bin volume)
         // need to set function and parameters here in case loop is parallelized 
         fval = igEval( x, data.Coords(i+1) ) ; 
      }


#ifdef DEBUG      
      std::cout << x[0] << "  " << y << "  " << 1./invError << " params : "; 
      for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar) 
         std::cout << p[ipar] << "\t";
      std::cout << "\tfval = " << fval << std::endl; 
#endif


      double tmp = ( y -fval )* invError;  	  
      double resval = tmp * tmp;


      // avoid inifinity or nan in chi2 values due to wrong function values 
      if ( resval < std::numeric_limits<double>::max() )  
         chi2 += resval; 
      else 
         nRejected++; 

      
   }
   
   // reset the number of fitting data points
   if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nRejected << std::endl;
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
   int nRejected = 0; 
   

   //func.SetParameters(p); 

   unsigned int ndim = func.NDim();
   SimpleGradientCalculator gradCalc(ndim, func );  
   std::vector<double> grad( ndim ); 


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
         for (unsigned int icoord = 0; icoord < ndim; ++icoord) { 
            if (ex[icoord] != 0) 
               gradCalc.Gradient(x, p, fval, &grad[0]);
            double edx = ex[icoord] * grad[icoord]; 
            e2 += edx * edx;  
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
      if ( resval < std::numeric_limits<double>::max() )  
         chi2 += resval; 
      else 
         nRejected++; 
      
   }
   
   // reset the number of fitting data points
   if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nRejected << std::endl;
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

   if (data.Opt().fIntegral )
      MATH_WARN_MSG("FitUtil::EvaluateChi2Residual","Bin integral is not used in calculating Chi2 residual");


   //func.SetParameters(p);

   double y, invError = 0; 
   const double * x = data.GetPoint(i,y, invError);

   double fval = func ( x , p); 

//       // calculate normalized integral (divided by bin volume)
//       IntegralEvaluator igEval( func, fitOpt.fIntegral); 
//       fval = igEval( x, data.Coords(i+1) ) ; 

   double resval =   ( y -fval )* invError;  	   

   // avoid infinities or nan in  resval
   resval = CorrectValue(resval);
   
   // estimate gradient 
   if (g == 0) return resval; 

   unsigned int npar = func.NPar(); 
   const IGradModelFunction * gfunc = dynamic_cast<const IGradModelFunction *>( &func); 

   if (gfunc != 0) { 
      //case function provides gradient
      gfunc->ParameterGradient(  x , p, g );  
   }
   else { 
      SimpleGradientCalculator  gc( func.NPar(), func); 
      gc.ParameterGradient(x, p, fval, g); 
   }
   // mutiply by - 1 * weihgt
   for (unsigned int k = 0; k < npar; ++k) {
      g[k] *= - invError;
   }       


   return resval; 

}

void FitUtil::EvaluateChi2Gradient(const IModelFunction & f, const BinData & data, const double * p, double * grad, unsigned int & nPoints) { 
   // evaluate gradient of chi2
   // this function is used when the model function knows how to calculate the derivative and we can  
   // avoid that the minimizer re-computes them 
   //
   // need to implement cases with integral option and errors on the coordinates
   
   if (data.Opt().fIntegral )
      MATH_WARN_MSG("FitUtil::EvaluateChi2Residual","Bin integral is not used in calculating Chi2 gradient");

   if ( data.GetErrorType() == BinData::kCoordError && data.Opt().fCoordErrors ) {
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

   //int nRejected = 0; 
   // set values of parameters 

   unsigned int npar = func.NPar(); 
//   assert (npar == NDim() );  // npar MUST be  Chi2 dimension
   std::vector<double> gradFunc( npar ); 
   // set all vector values to zero
   std::vector<double> g( npar); 

   for (unsigned int i = 0; i < n; ++ i) { 


      double y, invError = 0; 
      const double * x = data.GetPoint(i,y, invError);

      double fval = func ( x, p ); 
      func.ParameterGradient(  x , p, &gradFunc[0] );  

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
      for (unsigned int ipar = 0; ipar < npar ; ++ipar) { 

         // avoid singularity in the function (infinity and nan ) in the chi2 sum 
         // eventually add possibility of excluding some points (like singularity) 
         double dfval = gradFunc[ipar];
         if ( !CheckValue(dfval) ) { 
               nRejected++; 
               break; // exit loop on parameters
         } 
 
         // calculate derivative point contribution
         double tmp = - 2.0 * ( y -fval )* invError * invError * gradFunc[ipar];  	  
         g[ipar] += tmp;
      
      }

   } 

   // correct the number of points
   if (nRejected != 0)  nPoints = n - nRejected;

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
   // evaluate the pdf contribution to the logl
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
   int nRejected = 0; 

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
      if (fval < 0) { 
         nRejected++; // reject points with negative pdf (cannot exist)
      }
      else 
         logl += ROOT::Math::Util::EvalLog( fval); 
      
   }
   
   // reset the number of fitting data points
   if (nRejected != 0)  nPoints = n - nRejected;

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
      if (fval > 0) { 
         func.ParameterGradient( x, p, &gradFunc[0] );
         for (unsigned int kpar = 0; kpar < npar; ++ kpar) { 
            g[kpar] -= 1./fval * gradFunc[ kpar ]; 
         }
            
      }
    // copy result 
   std::copy(g.begin(), g.end(), grad);
   }
}
//_________________________________________________________________________________________________
// for binned log likelihood functions      
//------------------------------------------------------------------------------------------------
double FitUtil::EvaluatePoissonBinPdf(const IModelFunction & func, const BinData & data, const double * p, unsigned int i, double * g ) {  
   // evaluate the pdf contribution to the logl (return actually log of pdf)
   // t.b.d. implement integral option


   if (data.Opt().fIntegral )
      MATH_WARN_MSG("FitUtil::EvaluatePoissonBinPdf","Bin integral is not used in calculating pdf for Poisson pdf for the bin");


   double y = 0; 
   const double * x = data.GetPoint(i,y);


   double fval = func ( x , p); 

   // logPdf for Poisson: ignore constant term depending on N
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
      gfunc->ParameterGradient(  x , p, g );  
      // correct g[] do be derivative of poisson term 
      for (unsigned int k = 0; k < npar; ++k) {
         g[k] *= ( y/fval - 1.) ;//* pdfval; 
      }       
   }
   else { 
//       // estimate gradient numerically with simple 2 point rule 
//       static const double kEps = 1.0E-6;      
//       std::vector<double> p2(p,p+npar);
//       for (unsigned int k = 0; k < npar; ++k) {
//          p2[k] += kEps*p2[k];
//          // t.b.d : treat case of infinities 
//          // make derivatives of the log (more stables ) and correct afterwards
//          double fval2 = func(x,&p2.front() );    
//          double logPdf2 = y * ROOT::Math::Util::EvalLog( fval2) - fval2; 
//          double dlogPdf = (logPdf2 - logPdf)/kEps;
//          g[k] = dlogPdf;// * pdfval;
//          p2[k] = p[k]; // restore original p value
      
      SimpleGradientCalculator  gc(func.NPar(), func); 
      gc.ParameterGradient(x, p, fval, g); 
      // correct g[] do be derivative of poisson term 
      for (unsigned int k = 0; k < npar; ++k) {
         g[k] *= ( y/fval - 1.) ; 
      }       
     
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

double FitUtil::EvaluatePoissonLogL(const IModelFunction & func, const BinData & data, const double * p, unsigned int &nPoints) {  
   // evaluate the Poisson Log Likelihood
   // for binned likelihood fits

   unsigned int n = data.Size();

   
   double loglike = 0;
   int nRejected = 0; 

   // get fit option and check case of using integral of bins
   const DataOptions & fitOpt = data.Opt();
   IntegralEvaluator igEval( func, p, fitOpt.fIntegral); 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x = data.Coords(i);
      double y = data.Value(i);

      double fval = 0;   
      if (!fitOpt.fIntegral )
         fval = func ( x, p ); 
      else { 
         // calculate normalized integral (divided by bin volume)
         fval = igEval( x, data.Coords(i+1) ) ; 
      }


      loglike +=  fval - y * ROOT::Math::Util::EvalLog( fval);  
      
      
   }
   
   // reset the number of fitting data points
   if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   std::cout << "Loglikelihood  = " << loglike << " np = " << nPoints << std::endl;
#endif
   
   return loglike;  
}

void FitUtil::EvaluatePoissonLogLGradient(const IModelFunction & f, const BinData & data, const double * p, double * grad ) { 
   // evaluate the gradient of the log likelihood function
   // t.b.d. add integral option

   const IGradModelFunction * fg = dynamic_cast<const IGradModelFunction *>( &f); 
   assert (fg != 0); // must be called by a grad function
   const IGradModelFunction & func = *fg; 

   unsigned int n = data.Size();

   unsigned int npar = func.NPar(); 
   std::vector<double> gradFunc( npar ); 
   std::vector<double> g( npar); 

   for (unsigned int i = 0; i < n; ++ i) { 
      const double * x = data.Coords(i);
      double y = data.Value(i);
      double fval = func ( x, p ); 
      if (fval > 0) { 
         func.ParameterGradient( x, p, &gradFunc[0] );
         for (unsigned int kpar = 0; kpar < npar; ++ kpar) { 
            // df/dp * (1.  - y/f )
            g[kpar] += gradFunc[ kpar ] * ( 1. - y/fval ); 
         }            
      }
    // copy result 
   std::copy(g.begin(), g.end(), grad);
   }
}
   
}

} // end namespace ROOT

