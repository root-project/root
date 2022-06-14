// @(#)root/mathmore:$Id$
// Authors: L. Moneta    10/2010

#include <cmath>

#include "Math/SpecFuncMathMore.h"
#include "Math/SpecFuncMathCore.h"
#include "Math/PdfFuncMathCore.h"
#include "Math/DistFuncMathMore.h"


#include "gsl/gsl_sf_hyperg.h"  // needed for 0F1


namespace ROOT {
namespace Math {


//non central chisquare pdf (impelmentation from Kyle Cranmer)
// formula from Wikipedia http://en.wikipedia.org/wiki/Noncentral_chi-square_distribution
// but use hybergeometric form for k < 2
double noncentral_chisquared_pdf(double x, double k, double lambda) {

   // special case (form below doesn't work when lambda==0)
   if(lambda==0){
     return ROOT::Math::chisquared_pdf(x,k);
   }
   double ret = 0;
   if (k < 2.0) {

      //  expression using regularized confluent hypergeometric limit function.
      //see  http://mathworld.wolfram.com/NoncentralChi-SquaredDistribution.html
      //  (note  0\tilde{F}(a,x)  = 0F1(a,x)/ Gamma(a)
      // or  wikipedia
      // NOTE : this has problems for large k (so use only fr k <= 2)

      ret = std::exp( - 0.5 *(x + lambda) ) * 1./(std::pow(2.0, 0.5 * k) * ROOT::Math::tgamma(0.5*k)) * std::pow( x, 0.5 * k - 1.0)
      * gsl_sf_hyperg_0F1( 0.5 * k, 0.25 * lambda * x );

   }
   else {

      // SECOND FORM
      // 1/2 exp(-(x+lambda)/2) (x/lambda)^(k/4-1/2) I_{k/2 -1}(\sqrt(lamba x))
      // where I_v(z) is modified bessel of the first kind
      // bessel defined only for nu > 0

      ret = 0.5 * std::exp(-0.5 * (x+lambda) ) * std::pow(x/lambda, 0.25*k-0.5)
         * ROOT::Math::cyl_bessel_i (0.5*k-1., std::sqrt(lambda*x));

//       ret = 0.5 * exp(-(_x+lambda)/2.) * pow(_x/lambda, k/4.-0.5)
//      * ROOT::Math::cyl_bessel_i (k/2.-1., sqrt(lambda*_x));

   }

   return ret;
}


} // namespace Math


} // namespace ROOT

#include "Math/Error.h"

// dummy method called to force auto-loading of MathMore Library
//
// note: a typdef MathMoreLIb to MathMoreLibrary has been introduced because for unknown reasons
// loading on the MathMoreLibrary does not work while it works for the class name MathMoreLib
// see ROOT-8455

void ROOT::Math::MathMoreLib::Load() {
    MATH_INFO_MSG("MathMoreLibrary","libMathMore has been loaded.");
 }

