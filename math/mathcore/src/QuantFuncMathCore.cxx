// @(#)root/mathcore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005 


#include "Math/Math.h"
#include "Math/QuantFuncMathCore.h"
#include "SpecFuncCephes.h"
#include <limits>


namespace ROOT {
namespace Math {



   double beta_quantile_c(double z, double a, double b) {
      // use Cephes and proprety of icomplete beta function
      if ( z < 0.5) 
         return 1. - ROOT::Math::Cephes::incbi(b,a,z);
      else 
         return ROOT::Math::Cephes::incbi(a,b,1.0-z);

   }


   double beta_quantile(double z, double a, double b ) {
      // use Cephes function
      return ROOT::Math::Cephes::incbi(a,b,z);

   }


   double cauchy_quantile_c(double z, double b) {
      // inverse of Caucy is simply the tan(PI(z-0.5))
      if (z == 0) return std::numeric_limits<double>::infinity(); 
      if (z == 1) return - std::numeric_limits<double>::infinity();
      if (z < 0.5)  
         // use fact that tan(PI(0.5-z)) = 1/tan(PI*z)
         return  b / std::tan( M_PI * z ); 
      else 
         return  b * std::tan( M_PI * (0.5 -  z ) ); 
   }



   double cauchy_quantile(double z, double b) {
      // inverse of Caucy is simply the tan(PI(z-0.5))
      if (z == 0) return - std::numeric_limits<double>::infinity(); 
      if (z == 1) return + std::numeric_limits<double>::infinity();
      if (z < 0.5)  
         // use fact that tan(PI(0.5-z)) = 1/tan(PI*z)
         return  - b / std::tan( M_PI * z ); 
      else 
         return  b * std::tan( M_PI * ( z -  0.5 ) ); 

   }



   double chisquared_quantile_c(double z, double r) {
      // use Cephes igami which return inverse of complemented regularized gamma
      return 2.* ROOT::Math::Cephes::igami( 0.5 *r, z); 

   }


#ifndef R__HAS_MATHMORE
   double chisquared_quantile(double z, double r) {
      // use Cephes (probably large error for z approx 1) 
      return 2.* ROOT::Math::Cephes::igami( 0.5 *r, 1. - z); 
   }
#endif


   double exponential_quantile_c(double z, double lambda) {

      return - std::log(z)/ lambda; 

   }



   double exponential_quantile(double z, double lambda) {
      // use log1p for avoid errors at small z
      return - ROOT::Math::log1p(-z)/lambda;

   }


   double fdistribution_quantile_c(double z, double n, double m) {
      // use cephes incbi function and use propreties of incomplete beta for case <> 0.5
      if (n == 0) return 0;  // is value of cdf for n = 0 
      if (z < 0.5) { 
         double y =  ROOT::Math::Cephes::incbi( .5*m, .5*n, z); 
         return m/(n * y) - m/n; 
      }
      else { 
         double y =  ROOT::Math::Cephes::incbi( .5*n, .5*m, 1.0 - z); 
         // will lose precision for y approx to 1
         return  m * y /(n * ( 1. - y) ); 
      }
   }

   double fdistribution_quantile(double z, double n, double m) {
      // use cephes incbi function
      if (n == 0) return 0;  // is value of cdf for n = 0 
      double y =  ROOT::Math::Cephes::incbi( .5*n, .5*m, z); 
      // will lose precision for y approx to 1
      return  m * y /(n * ( 1. - y) ); 
   }


   double gamma_quantile_c(double z, double alpha, double theta) {

      return theta * ROOT::Math::Cephes::igami( alpha, z); 

   }

#ifndef R__HAS_MATHMORE
   double gamma_quantile(double z, double alpha, double theta) {
      // use gamma_quantile_c (large error for z close to 1)
      return theta * ROOT::Math::Cephes::igami( alpha, 1.- z); 
   }
#endif



   double normal_quantile_c(double z, double sigma) {
      // use cephes and fact that ntri(1.-x) = - ndtri(x)
      return - sigma * ROOT::Math::Cephes::ndtri(z);

   }



   double normal_quantile(double z, double sigma) {
      // use cephes ndtri function
      return  sigma * ROOT::Math::Cephes::ndtri(z);

   }




   double lognormal_quantile_c(double z, double m, double s) {
      // if y is log normal, u = exp(y) is log-normal distributed  
      double y = - s * ROOT::Math::Cephes::ndtri(z) + m;
      return std::exp(y);
   }



   double lognormal_quantile(double z, double m, double s) {
      // if y is log normal, u = exp(y) is log-normal distributed  
      double y = s * ROOT::Math::Cephes::ndtri(z) + m;
      return std::exp(y);

   }


//   double tdistribution_quantile_c(double z, double r) {

//     return gsl_cdf_tdist_Qinv(z, r);

//   }



//   double tdistribution_quantile(double z, double r) {

//     return gsl_cdf_tdist_Pinv(z, r);

//   }



   double uniform_quantile_c(double z, double a, double b) {

      return a * z  + b * (1.0 - z);  

   }



   double uniform_quantile(double z, double a, double b) {

      return b * z + a * (1.0 - z);

   }





} // namespace Math
} // namespace ROOT
