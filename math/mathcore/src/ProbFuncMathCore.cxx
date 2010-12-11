// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   06/2005 


#include "Math/Math.h"
#include "Math/ProbFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"

#include <limits>

namespace ROOT {
namespace Math {

  
   static const double kSqrt2 = 1.41421356237309515; // sqrt(2.)

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
         // use expm1 function to avoid errors at small x
         return - ROOT::Math::expm1( - lambda * (x-x0) ) ;
         
      }
      
   }
   
  
   double fdistribution_cdf_c(double x, double n, double m, double x0) {
      // f distribution  is defined only for both n and m > 0
      if (n < 0 || m < 0)  
         return std::numeric_limits<double>::quiet_NaN(); 

      double z = m/(m + n*(x-x0));
      // fox z->1 and large a and b IB looses precision use complement function 
      if (z > 0.9 && n > 1 && m > 1) return 1.-  fdistribution_cdf(x,n,m,x0);

      // for the complement use the fact that IB(x,a,b) = 1. - IB(1-x,b,a)     
      return ROOT::Math::inc_beta(m/(m + n*(x-x0)), .5*m, .5*n);
  
   }


   double fdistribution_cdf(double x, double n, double m, double x0) {
      // f distribution  is defined only for both n and m > 0
      if (n < 0 || m < 0)  
         return std::numeric_limits<double>::quiet_NaN(); 

      double z = n*(x-x0)/(m + n*(x-x0));
      // fox z->1 and large a and b IB looses precision use complement function
      if (z > 0.9 && n > 1 && m > 1) return 1. - fdistribution_cdf_c(x,n,m,x0);

      return ROOT::Math::inc_beta(z, .5*n, .5*m);
   }



   double gamma_cdf_c(double x, double alpha, double theta, double x0) {

      return ROOT::Math::inc_gamma_c(alpha, (x-x0)/theta);

   }



   double gamma_cdf(double x, double alpha, double theta, double x0) {

      return ROOT::Math::inc_gamma(alpha, (x-x0)/theta);
   }

  
   

   
   double lognormal_cdf_c(double x, double m, double s, double x0) {
      
      double z = (std::log((x-x0))-m)/(s*kSqrt2); 
      if ( z > 1. )
         return 0.5*ROOT::Math::erfc(z);
      else 
         return 0.5*(1.0 - ROOT::Math::erf(z));
      
   }
   
   
   
   double lognormal_cdf(double x, double m, double s, double x0) {
      
      double z = (std::log((x-x0))-m)/(s*kSqrt2); 
      if ( z < -1. )
         return 0.5*ROOT::Math::erfc(-z);
      else 
         return 0.5*(1.0 + ROOT::Math::erf(z));
      
   }
   
   
   
   double normal_cdf_c(double x, double sigma, double x0) {
      
      double z = (x-x0)/(sigma*kSqrt2);
      if ( z > 1. )
         return 0.5*ROOT::Math::erfc(z);
      else 
         return 0.5*(1.0 - ROOT::Math::erf(z));

      
   }
   
   
   
   double normal_cdf(double x, double sigma, double x0) {
    
      double z = (x-x0)/(sigma*kSqrt2);
      if ( z < -1. )
         return 0.5*ROOT::Math::erfc(-z);
      else 
         return 0.5*(1.0 + ROOT::Math::erf(z));
      
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

   double negative_binomial_cdf(unsigned int k, double p, double n) {
      // use relation with in beta distribution
      // p must be 0 <=p <= 1
      if ( n < 0) return 0; 
      if ( p < 0 || p > 1) return 0; 

      return ROOT::Math::beta_cdf(p, n, k+1.0);
   }

   double negative_binomial_cdf_c(unsigned int k, double p, double n) {
      // use relation with in beta distribution
      // p must be 0 <=p <= 1
      if ( n < 0) return 0; 
      if ( p < 0 || p > 1) return 0; 

      return ROOT::Math::beta_cdf_c(p, n, k+1.0);
   }



   double landau_cdf(double x, double xi, double x0) { 
      // implementation of landau distribution (from DISLAN)
   //The algorithm was taken from the Cernlib function dislan(G110)
   //Reference: K.S.Kolbig and B.Schorr, "A program package for the Landau
   //distribution", Computer Phys.Comm., 31(1984), 97-111

      static double p1[5] = {0.2514091491e+0,-0.6250580444e-1, 0.1458381230e-1, -0.2108817737e-2, 0.7411247290e-3};
      static double q1[5] = {1.0            ,-0.5571175625e-2, 0.6225310236e-1, -0.3137378427e-2, 0.1931496439e-2};

      static double p2[4] = {0.2868328584e+0, 0.3564363231e+0, 0.1523518695e+0, 0.2251304883e-1};
      static double q2[4] = {1.0            , 0.6191136137e+0, 0.1720721448e+0, 0.2278594771e-1};

      static double p3[4] = {0.2868329066e+0, 0.3003828436e+0, 0.9950951941e-1, 0.8733827185e-2};
      static double q3[4] = {1.0            , 0.4237190502e+0, 0.1095631512e+0, 0.8693851567e-2};

      static double p4[4] = {0.1000351630e+1, 0.4503592498e+1, 0.1085883880e+2, 0.7536052269e+1};
      static double q4[4] = {1.0            , 0.5539969678e+1, 0.1933581111e+2, 0.2721321508e+2};

      static double p5[4] = {0.1000006517e+1, 0.4909414111e+2, 0.8505544753e+2, 0.1532153455e+3};
      static double q5[4] = {1.0            , 0.5009928881e+2, 0.1399819104e+3, 0.4200002909e+3};

      static double p6[4] = {0.1000000983e+1, 0.1329868456e+3, 0.9162149244e+3, -0.9605054274e+3};
      static double q6[4] = {1.0            , 0.1339887843e+3, 0.1055990413e+4, 0.5532224619e+3};

      static double a1[4] = {0, -0.4583333333e+0, 0.6675347222e+0,-0.1641741416e+1};

      static double a2[4] = {0,  1.0            ,-0.4227843351e+0,-0.2043403138e+1};

      double v = (x - x0)/xi; 
      double u;
      double lan;

      if (v < -5.5) {
         u = std::exp(v+1);
         lan = 0.3989422803*std::exp(-1./u)*std::sqrt(u)*(1+(a1[1]+(a1[2]+a1[3]*u)*u)*u);
      }
      else if (v < -1 ) {
         u = std::exp(-v-1);
         lan = (std::exp(-u)/std::sqrt(u))*(p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
            (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      }
      else if (v < 1)
         lan = (p2[0]+(p2[1]+(p2[2]+p2[3]*v)*v)*v)/(q2[0]+(q2[1]+(q2[2]+q2[3]*v)*v)*v);
      else if (v < 4)
         lan = (p3[0]+(p3[1]+(p3[2]+p3[3]*v)*v)*v)/(q3[0]+(q3[1]+(q3[2]+q3[3]*v)*v)*v);
      else if (v < 12) {
         u = 1./v;
         lan = (p4[0]+(p4[1]+(p4[2]+p4[3]*u)*u)*u)/(q4[0]+(q4[1]+(q4[2]+q4[3]*u)*u)*u);
      }
      else if (v < 50) {
         u = 1./v;
         lan = (p5[0]+(p5[1]+(p5[2]+p5[3]*u)*u)*u)/(q5[0]+(q5[1]+(q5[2]+q5[3]*u)*u)*u);
      }
      else if (v < 300) {
         u = 1./v;
         lan = (p6[0]+(p6[1]+(p6[2]+p6[3]*u)*u)*u)/(q6[0]+(q6[1]+(q6[2]+q6[3]*u)*u)*u);
      }
      else {
         u = 1./(v-v*std::log(v)/(v+1));
         lan = 1-(a2[1]+(a2[2]+a2[3]*u)*u)*u;
      }
      return lan;
      
   }


   
   double landau_xm1(double x, double xi, double x0) {
      // implementation of first momentum of Landau distribution
      // translated from Cernlib (XM1LAN function) by Benno List  

      static double p1[5] = {
       -0.8949374280E+0, 0.4631783434E+0,-0.4053332915E-1,
        0.1580075560E-1,-0.3423874194E-2};
      static double q1[5] = {
        1.0            , 0.1002930749E+0, 0.3575271633E-1,
       -0.1915882099E-2, 0.4811072364E-4};
      static double p2[5] = {
       -0.8933384046E+0, 0.1161296496E+0, 0.1200082940E+0,
        0.2185699725E-1, 0.2128892058E-2};
      static double q2[5] = {
        1.0            , 0.4935531886E+0, 0.1066347067E+0,
        0.1250161833E-1, 0.5494243254E-3};
      static double p3[5] = {
       -0.8933322067E+0, 0.2339544896E+0, 0.8257653222E-1,
        0.1411226998E-1, 0.2892240953E-3};
      static double q3[5] = {
        1.0            , 0.3616538408E+0, 0.6628026743E-1,
        0.4839298984E-2, 0.5248310361E-4};
      static double p4[4] = {
        0.9358419425E+0, 0.6716831438E+2,-0.6765069077E+3,
        0.9026661865E+3};
      static double q4[4] = {
        1.0            , 0.7752562854E+2,-0.5637811998E+3,
       -0.5513156752E+3};
      static double p5[4] = {
        0.9489335583E+0, 0.5561246706E+3, 0.3208274617E+5,
       -0.4889926524E+5};
      static double q5[4] = {
        1.0            , 0.6028275940E+3, 0.3716962017E+5,
        0.3686272898E+5};
      static double a0[6] = {
       -0.4227843351E+0,-0.1544313298E+0, 0.4227843351E+0,
        0.3276496874E+1, 0.2043403138E+1,-0.8681296500E+1};
      static double a1[4] = {0,
       -0.4583333333E+0, 0.6675347222E+0,-0.1641741416E+1};
      static double a2[5] = {0,
       -0.1958333333E+1, 0.5563368056E+1,-0.2111352961E+2,
        0.1006946266E+3};

      double v = (x-x0)/xi;
      double xm1lan;
      if (v < -4.5) {
         double u = std::exp(v+1);
         xm1lan = v-u*(1+(a2[1]+(a2[2]+(a2[3]+a2[4]*u)*u)*u)*u)/
                      (1+(a1[1]+(a1[2]+a1[3]*u)*u)*u);
      } else if(v < -2) {
         xm1lan = (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
                  (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      } else if(v < 2) {
         xm1lan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
                  (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
      } else if(v < 10) {
         xm1lan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
                  (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
      } else if(v < 40) {
         double u = 1/v;
         xm1lan = std::log(v)*(p4[0]+(p4[1]+(p4[2]+p4[3]*u)*u)*u)/
                              (q4[0]+(q4[1]+(q4[2]+q4[3]*u)*u)*u);
      } else if(v < 200) {
         double u = 1/v;
         xm1lan= std::log(v)*(p5[0]+(p5[1]+(p5[2]+p5[3]*u)*u)*u)/
                             (q5[0]+(q5[1]+(q5[2]+q5[3]*u)*u)*u);
      } else {
       double u = v-v*std::log(v)/(v+1);
       v = 1/(u-u*(u+ std::log(u)-v)/(u+1));
       u = -std::log(v);
       xm1lan = (u+a0[0]+(-u+a0[1]+(a0[2]*u+a0[3]+(a0[4]*u+a0[5])*v)*v)*v)/
                (1-(1-(a0[2]+a0[4]*v)*v)*v);
      }
      return xm1lan*xi + x0;
      
   }
   
   
   
   double landau_xm2(double x, double xi, double x0) {
      // implementation of second momentum of Landau distribution
      // translated from Cernlib (XM2LAN function) by Benno List  

      static double p1[5] = {
        0.1169837582E+1,-0.4834874539E+0, 0.4383774644E+0,
        0.3287175228E-2, 0.1879129206E-1};
      static double q1[5] = {
        1.0            , 0.1795154326E+0, 0.4612795899E-1,
        0.2183459337E-2, 0.7226623623E-4};
      static double p2[5] = {
        0.1157939823E+1,-0.3842809495E+0, 0.3317532899E+0,
        0.3547606781E-1, 0.6725645279E-2};
      static double q2[5] = {
        1.0            , 0.2916824021E+0, 0.5259853480E-1,
        0.3840011061E-2, 0.9950324173E-4};
      static double p3[4] = {
        0.1178191282E+1, 0.1011623342E+2,-0.1285585291E+2,
        0.3641361437E+2};
      static double q3[4] = {
        1.0            , 0.8614160194E+1, 0.3118929630E+2,
        0.1514351300E+0};
      static double p4[5] = {
        0.1030763698E+1, 0.1216758660E+3, 0.1637431386E+4,
       -0.2171466507E+4, 0.7010168358E+4};
      static double q4[5] = {
        1.0            , 0.1022487911E+3, 0.1377646350E+4,
        0.3699184961E+4, 0.4251315610E+4};
      static double p5[4] = {
        0.1010084827E+1, 0.3944224824E+3, 0.1773025353E+5,
       -0.7075963938E+5};
      static double q5[4] = {
        1.0            , 0.3605950254E+3, 0.1392784158E+5,
       -0.1881680027E+5};
      static double a0[7] = {
       -0.2043403138E+1,-0.8455686702E+0,-0.3088626596E+0,
        0.5821346754E+1, 0.4227843351E+0, 0.6552993748E+1,
       -0.1076714945E+2};
      static double a1[4] = {0,
       -0.4583333333E+0, 0.6675347222E+0,-0.1641741416E+1};
      static double a2[4] = {
       -0.1958333333E+1, 0.5563368056E+1,-0.2111352961E+2,
        0.1006946266E+3};
      static double a3[4] = {
       -1.0            , 0.4458333333E+1,-0.2116753472E+2,
        0.1163674359E+3};

      double v = (x-x0)/xi;
      double xm2lan;
      if (v < -4.5) {
         double u = std::exp(v+1);
         xm2lan = v*v-2*u*u*
                  (v/u+a2[0]*v+a3[0]+(a2[1]*v+a3[1]+(a2[2]*v+a3[2]+
                  (a2[3]*v+a3[3])*u)*u)*u)/
                  (1+(a1[1]+(a1[2]+a1[3]*u)*u)*u);
      } else if (v < -2) {
         xm2lan = (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
                  (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      } else if (v < 2) {
         xm2lan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
                  (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
      } else if (v < 5) {
         double u = 1/v;
         xm2lan = v*(p3[0]+(p3[1]+(p3[2]+p3[3]*u)*u)*u)/
                    (q3[0]+(q3[1]+(q3[2]+q3[3]*u)*u)*u);
      } else if (v < 50) {
         double u = 1/v;
         xm2lan = v*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
                    (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
      } else if (v < 200) {
         double u = 1/v;
         xm2lan = v*(p5[0]+(p5[1]+(p5[2]+p5[3]*u)*u)*u)/
                    (q5[0]+(q5[1]+(q5[2]+q5[3]*u)*u)*u);
      } else { 
         double u = v-v*std::log(v)/(v+1);
         v = 1/(u-u*(u+log(u)-v)/(u+1));
         u = -std::log(v);
         xm2lan = (1/v+u*u+a0[0]+a0[1]*u+(-u*u+a0[2]*u+a0[3]+
                  (a0[4]*u*u+a0[5]*u+a0[6])*v)*v)/(1-(1-a0[4]*v)*v);
      }
      if (x0 == 0) return xm2lan*xi*xi;
      double xm1lan = ROOT::Math::landau_xm1(x, xi, x0);
      return xm2lan*xi*xi + (2*xm1lan-x0)*x0;
   }

} // namespace Math
} // namespace ROOT



