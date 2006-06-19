// @(#)root/mathcore:$Name:  $:$Id: DistFunc.cxx,v 1.6 2005/12/08 21:56:39 moneta Exp $
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



#include <cmath>
#include "Math/SpecFuncMathCore.h"

#ifndef M_PI
#define M_PI        3.14159265358979323846   /* pi */
#endif


namespace ROOT {
namespace Math {




  double binomial_pdf(unsigned int k, double p, unsigned int n) {
     
     if (k > n) {
        return 0.0;
     } else {
        
        double coeff = ROOT::Math::lgamma(n+1) - ROOT::Math::lgamma(k+1) - ROOT::Math::lgamma(n-k+1);
        return std::exp(coeff + k * std::log(p) + (n - k) * std::log(1 - p));
     }
  }
   
   
   double breitwigner_pdf(double x, double gamma, double x0) {
      
      double gammahalf = gamma/2.0;
      return gammahalf/(M_PI * ((x-x0)*(x-x0) + gammahalf*gammahalf));
      
      
   }
   
   
   
   double cauchy_pdf(double x, double b, double x0) {
      
      return b/(M_PI * ((x-x0)*(x-x0) + b*b));
      
   }
   
   
   
   double chisquared_pdf(double x, double r, double x0) {
      
      if ((x-x0) <= 0) {
         return 0.0;
      } else {
         return std::exp ((r/2 - 1) * std::log((x-x0)/2) - (x-x0)/2 - ROOT::Math::lgamma(r/2))/2;
      }
      
   }
   
   
   
   double exponential_pdf(double x, double lambda, double x0) {
      
      if ((x-x0) < 0) {
         return 0.0;
      } else {
         return lambda * std::exp (-lambda * (x-x0));
      }
      
   }
   
   
   
   double fdistribution_pdf(double x, double n, double m, double x0) {
      
      if ((x-x0) < 0) {
         return 0.0;
      } else {
         
         return std::exp((n/2) * std::log(n) + (m/2) * std::log(m) + ROOT::Math::lgamma((n+m)/2) - ROOT::Math::lgamma(n/2) - ROOT::Math::lgamma(m/2))
         * std::pow((x-x0), n/2-1) * std::pow (m + n*(x-x0), -(n+m)/2);
         
      }
      
   }
   
   
   
   double gamma_pdf(double x, double alpha, double theta, double x0) {
      
      if ((x-x0) < 0) {
         return 0.0;
      } else if ((x-x0) == 0) {
         
         if (alpha == 1) {
            return 1.0/theta;
         } else {
            return 0.0;
         }
         
      } else if (alpha == 1) {
         return std::exp(-(x-x0)/theta)/theta;
      } else {
         return std::exp((alpha - 1) * std::log((x-x0)/theta) - (x-x0)/theta - ROOT::Math::lgamma(alpha))/theta;
      }
      
   }
   
   
   
   double gaussian_pdf(double x, double sigma, double x0) {
      
      double tmp = (x-x0)/sigma;
      return (1.0/(std::sqrt(2 * M_PI) * std::fabs(sigma))) * std::exp(-tmp*tmp/2);
   }
   
   
   /**
   double landau_pdf(double x) {
      
      return gsl_ran_landau_pdf(x);
      
   }
    */
   
   
   double lognormal_pdf(double x, double m, double s, double x0) {
      
      if ((x-x0) <= 0) {
         return 0.0;
      } else {
         double tmp = (std::log((x-x0)) - m)/s;
         return 1.0 / ((x-x0) * std::fabs(s) * std::sqrt(2 * M_PI)) * std::exp(-(tmp * tmp) /2);
      }
      
   }
   
   
   
   double normal_pdf(double x, double sigma, double x0) {
      
      double tmp = (x-x0)/sigma;
      return (1.0/(std::sqrt(2 * M_PI) * std::fabs(sigma))) * std::exp(-tmp*tmp/2);
      
   }
   
   
   
   double poisson_pdf(unsigned int n, double mu) {
      
      return std::exp (n*std::log(mu) - ROOT::Math::lgamma(n+1) - mu);
      
   }
   
   
   
   double tdistribution_pdf(double x, double r, double x0) {
      
      return (std::exp (ROOT::Math::lgamma((r + 1.0)/2.0) - ROOT::Math::lgamma(r/2.0)) / std::sqrt (M_PI * r)) 
      * std::pow ((1.0 + (x-x0)*(x-x0)/r), -(r + 1.0)/2.0);
      
   }
   
   
   
   double uniform_pdf(double x, double a, double b, double x0) {
      
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! when a=b
      
      if ((x-x0) < b && (x-x0) >= a) {
         return 1.0/(b - a);
      } else {
         return 0.0;
      }
      
   }
   


} // namespace Math
} // namespace ROOT





