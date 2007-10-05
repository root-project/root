// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   06/2005 


#include <cmath>
#include "Math/ProbFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"

#ifndef M_PI
#define M_PI        3.14159265358979323846   /* pi */
#endif


namespace ROOT {
namespace Math {

  


  double beta_cdf_c(double x, double a, double b) {
     // use the fact that I(x,a,b) = 1. - I(1-x,b,a)
     return ROOT::Math::inc_beta(1-x, b, a);
  }


  double beta_cdf(double x, double a, double b ) {

     return ROOT::Math::inc_beta(x, a, b);

  }

   

   double breitwigner_cdf_c(double x, double gamma, double x0) {
      
      return 0.5 - std::atan(2.0 * (x-x0) / gamma) / M_PI;
      
   }
   
   
   
   double breitwigner_cdf(double x, double gamma, double x0) {
      
      return 0.5 + std::atan(2.0 * (x-x0) / gamma) / M_PI;
      
   }
   
   
   
   double cauchy_cdf_c(double x, double b, double x0) {
      
      return 0.5 - std::atan( (x-x0) / b) / M_PI;
      
   }
   
   
   
   double cauchy_cdf(double x, double b, double x0) {
      
      return 0.5 + std::atan( (x-x0) / b) / M_PI;
      
   }
   
   
   
   double chisquared_cdf_c(double x, double r, double x0) {
      
      return ROOT::Math::inc_gamma_c ( 0.5 * r , 0.5* (x-x0) );
      
   }

   double chisquared_cdf(double x, double r, double x0) {
       
      return ROOT::Math::inc_gamma ( 0.5 * r , 0.5* (x-x0) );
       
   }
   
      
   
   double exponential_cdf_c(double x, double lambda, double x0) {
      
      if ((x-x0) < 0) {
         
         return 1.0;
         
      } else {
         
         return std::exp(- lambda * (x-x0));
         
      }
      
   }
   
   
   
   double exponential_cdf(double x, double lambda, double x0) {
      
      if ((x-x0) < 0) {
         
         return 0.0;
         
      } else {
         
         return 1.0 - std::exp(- lambda * (x-x0));
         
      }
      
   }
   
  
   double fdistribution_cdf_c(double x, double n, double m, double x0) {

      return ROOT::Math::inc_beta(m/(m + n*(x-x0)), .5*m, .5*n);
  
   }


   double fdistribution_cdf(double x, double n, double m, double x0) {

      return ROOT::Math::inc_beta(n*(x-x0)/(m + n*(x-x0)), .5*n, .5*m);
   }



   double gamma_cdf_c(double x, double alpha, double theta, double x0) {

      return ROOT::Math::inc_gamma_c(alpha, (x-x0)/theta);

   }



   double gamma_cdf(double x, double alpha, double theta, double x0) {

      return ROOT::Math::inc_gamma(alpha, (x-x0)/theta);
   }


   
   
   double gaussian_cdf_c(double x, double sigma, double x0) {
      
      return 0.5*(1.0 - ROOT::Math::erf((x-x0)/(sigma*std::sqrt(2.0))));
      
   }
   
   
   
   double gaussian_cdf(double x, double sigma, double x0) {
      
      return 0.5*(1.0 + ROOT::Math::erf((x-x0)/(sigma*std::sqrt(2.0))));
      
   }
   
   
   
   double lognormal_cdf_c(double x, double m, double s, double x0) {
      
      return 0.5*(1.0 - ROOT::Math::erf((std::log((x-x0))-m)/(s*std::sqrt(2.0))));
      
   }
   
   
   
   double lognormal_cdf(double x, double m, double s, double x0) {
      
      return 0.5*(1.0 + ROOT::Math::erf((std::log((x-x0))-m)/(s*std::sqrt(2.0))));
      
   }
   
   
   
   double normal_cdf_c(double x, double sigma, double x0) {
      
      return 0.5*(1.0 - ROOT::Math::erf((x-x0)/(sigma*std::sqrt(2.0))));
      
   }
   
   
   
   double normal_cdf(double x, double sigma, double x0) {
      
      return 0.5*(1 + ROOT::Math::erf((x-x0)/(sigma*std::sqrt(2.0))));
      
   }
   
   

   double tdistribution_cdf_c(double x, double r, double x0) {

      double p = x-x0;
      double sign = (p>0) ? 1. : -1;
      return .5 - .5*ROOT::Math::inc_beta(p*p/(r + p*p), .5, .5*r)*sign;

   }



   double tdistribution_cdf(double x, double r, double x0) {

      double p = x-x0;
      double sign = (p>0) ? 1. : -1;
      return  .5 + .5*ROOT::Math::inc_beta(p*p/(r + p*p), .5, .5*r)*sign;

   }
   
   
   double uniform_cdf_c(double x, double a, double b, double x0) {
      
      if ((x-x0) < a) {
         return 1.0;
      } else if ((x-x0) >= b) {
         return 0.0;
      } else {
         return (b-(x-x0))/(b-a);
      }
   }
   
   
   
   double uniform_cdf(double x, double a, double b, double x0) {
      
      if ((x-x0) < a) {
         return 0.0;
      } else if ((x-x0) >= b) {
         return 1.0;
      } else {
         return ((x-x0)-a)/(b-a);
      }    
   }
   
   /// discrete distributions

   double poisson_cdf_c(unsigned int n, double mu) {
      // mu must be >= 0  . Use poisson - gamma relation
      //  Pr ( x <= n) = Pr( y >= a)   where x is poisson and y is gamma distributed ( a = n+1)
      double a = (double) n + 1.0;           
      return ROOT::Math::gamma_cdf(mu, a, 1.0);
   }

   double poisson_cdf(unsigned int n, double mu) {
      // mu must be >= 0  . Use poisson - gamma relation
      //  Pr ( x <= n) = Pr( y >= a)   where x is poisson and y is gamma distributed ( a = n+1)
      double a = (double) n + 1.0; 
      return ROOT::Math::gamma_cdf_c(mu, a, 1.0);
   }

   double binomial_cdf_c(unsigned int k, double p, unsigned int n) {
      // use relation with in beta distribution
      // p must be 0 <=p <= 1
      if ( k >= n) return 0; 

      double a = (double) k + 1.0; 
      double b = (double) n - k; 
      return ROOT::Math::beta_cdf(p, a, b);
   }

   double binomial_cdf(unsigned int k, double p, unsigned int n) {
      // use relation with in beta distribution
      // p must be 0 <=p <= 1
      if ( k >= n) return 1.0; 

      double a = (double) k + 1.0; 
      double b = (double) n - k; 
      return ROOT::Math::beta_cdf_c(p, a, b);
   }




} // namespace Math
} // namespace ROOT



