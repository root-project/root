// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



#include "Math/Math.h"
#include "Math/SpecFuncMathCore.h"
#include <limits>


namespace ROOT {
namespace Math {

   
   double beta_pdf(double x, double a, double b) {
      if (x < 0 || x > 1.0) return 0;
      if (x == 0 ) {
         // need this wor Windows 
         if (a < 1) return  std::numeric_limits<double>::infinity();
         else if (a > 1) return  0;
         else if ( a == 1) return b; // to avoid a nan from log(0)*0 
      }
      if (x == 1 ) {
         // need this wor Windows 
         if (b < 1) return  std::numeric_limits<double>::infinity();
         else if (b > 1) return  0;
         else if ( b == 1) return a; // to avoid a nan from log(0)*0 
      }
      return std::exp( ROOT::Math::lgamma(a + b) - ROOT::Math::lgamma(a) - ROOT::Math::lgamma(b) + 
                       std::log(x) * (a -1.) + ROOT::Math::log1p(-x ) * (b - 1.) ); 
   }
   
   double binomial_pdf(unsigned int k, double p, unsigned int n) {
      
      if (k > n) {
         return 0.0;
      } else {
         
         double coeff = ROOT::Math::lgamma(n+1) - ROOT::Math::lgamma(k+1) - ROOT::Math::lgamma(n-k+1);
         return std::exp(coeff + k * std::log(p) + (n - k) * ROOT::Math::log1p(-p));
      }
   }

   double negative_binomial_pdf(unsigned int k, double p, double n) {
      // impelment in term of gamma function 
      
      if (n < 0)  return 0.0;
      if (p < 0 || p > 1.0) return 0.0;

      double coeff = ROOT::Math::lgamma(k+n) - ROOT::Math::lgamma(k+1.0) - ROOT::Math::lgamma(n);
      return std::exp(coeff + n * std::log(p) + double(k) * ROOT::Math::log1p(-p));

   }
   
   
   double breitwigner_pdf(double x, double gamma, double x0) {
      
      double gammahalf = gamma/2.0;
      return gammahalf/(M_PI * ((x-x0)*(x-x0) + gammahalf*gammahalf));
      
      
   }
   
   
   
   double cauchy_pdf(double x, double b, double x0) {
      
      return b/(M_PI * ((x-x0)*(x-x0) + b*b));
      
   }
   
   
   
   double chisquared_pdf(double x, double r, double x0) {
      
      if ((x-x0) <  0) {
         return 0.0;
      }
      double a = r/2 -1.; 
      // let return inf for case x  = x0 and treat special case of r = 2 otherwise will return nan
      if (x == x0 && a == 0) return 0.5;

      return std::exp ((r/2 - 1) * std::log((x-x0)/2) - (x-x0)/2 - ROOT::Math::lgamma(r/2))/2;
      
   }
   
   
   
   double exponential_pdf(double x, double lambda, double x0) {
      
      if ((x-x0) < 0) {
         return 0.0;
      } else {
         return lambda * std::exp (-lambda * (x-x0));
      }
      
   }
   
   
   
   double fdistribution_pdf(double x, double n, double m, double x0) {

      // function is defined only for both n and m > 0
      if (n < 0 || m < 0)  
         return std::numeric_limits<double>::quiet_NaN(); 
      if ((x-x0) < 0) 
         return 0.0;
         
      return std::exp((n/2) * std::log(n) + (m/2) * std::log(m) + ROOT::Math::lgamma((n+m)/2) - ROOT::Math::lgamma(n/2) - ROOT::Math::lgamma(m/2) 
                         + (n/2 -1) * std::log(x-x0) - ((n+m)/2) * std::log(m +  n*(x-x0)) );
         
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
   
   
   
   double landau_pdf(double x, double xi, double x0) {
      // LANDAU pdf : algorithm from CERNLIB G110 denlan
      // same algorithm is used in GSL 

      static double p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
      static double q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

      static double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
      static double q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

      static double p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
      static double q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

      static double p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
      static double q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

      static double p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
      static double q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

      static double p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
      static double q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

      static double a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};

      static double a2[2] = {-1.845568670,-4.284640743};

      if (xi <= 0) return 0; 
      double v = (x - x0)/xi;
      double u, ue, us, denlan;
      if (v < -5.5) {
         u   = std::exp(v+1.0);
         if (u < 1e-10) return 0.0;
         ue  = std::exp(-1/u);
         us  = std::sqrt(u);
         denlan = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
      } else if(v < -1) {
         u   = std::exp(-v-1);
         denlan = std::exp(-u)*std::sqrt(u)*
            (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
            (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      } else if(v < 1) {
         denlan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
            (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
      } else if(v < 5) {
         denlan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
            (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
      } else if(v < 12) {
         u   = 1/v;
         denlan = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
            (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
      } else if(v < 50) {
         u   = 1/v;
         denlan = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
            (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
      } else if(v < 300) {
         u   = 1/v;
         denlan = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
            (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
      } else {
         u   = 1/(v-v*std::log(v)/(v+1));
         denlan = u*u*(1+(a2[0]+a2[1]*u)*u);
      }
      return denlan/xi;
      
   }
   
      
    
   
   
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
      
      if (n >  0) 
         return std::exp (n*std::log(mu) - ROOT::Math::lgamma(n+1) - mu);
      else  {
         //  when  n = 0 and mu = 0,  1 is returned 
         if (mu >= 0) return std::exp(-mu);
         // return a nan for mu < 0 since it does not make sense
         return std::log(mu);
      }
      
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





