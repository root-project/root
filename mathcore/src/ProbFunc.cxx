// @(#)root/mathcore:$Name:  $:$Id: ProbFunc.cxx,v 1.1 2005/06/24 18:54:24 brun Exp $
// Authors: L. Moneta, A. Zsenei   06/2005 


#include <cmath>
#include "MathCore/ProbFunc.h"
#include "MathCore/SpecFunc.h"

#ifndef M_PI
#define M_PI        3.14159265358979323846   /* pi */
#endif


namespace ROOT {
namespace Math {

  


  double breitwigner_prob(double x, double gamma) {

    return 0.5 - std::atan(2.0 * x / gamma) / M_PI;

  }



  double breitwigner_quant(double x, double gamma) {

    return 0.5 + std::atan(2.0 * x / gamma) / M_PI;

  }

  

  double cauchy_prob(double x, double gamma) {

    return 0.5 - std::atan(2.0 * x / gamma) / M_PI;

  }



  double cauchy_quant(double x, double gamma) {

    return 0.5 + std::atan(2.0 * x / gamma) / M_PI;

  }

  /**

  double chisquared_prob(double x, double r) {

    return gsl_cdf_chisq_Q(x, r);

  }



  double chisquared_quant(double x, double r) {

    return gsl_cdf_chisq_P(x, r);

  }
  */


  double exponential_prob(double x, double lambda) {

    if (x < 0) {

      return 1.0;

    } else {

      return std::exp(- lambda * x);

    }

  }



  double exponential_quant(double x, double lambda) {

    if (x < 0) {

      return 0.0;

    } else {

      return 1.0 - std::exp(- lambda * x);

    }

  }


  /**
  double fdistribution_prob(double x, double n, double m) {

    return gsl_cdf_fdist_Q(x, n, m);

  }



  double fdistribution_quant(double x, double n, double m) {

    return gsl_cdf_fdist_P(x, n, m);

  }



  double gamma_prob(double x, double alpha, double theta) {

    return gsl_cdf_gamma_Q(x, alpha, theta);

  }



  double gamma_quant(double x, double alpha, double theta) {

    return gsl_cdf_gamma_P(x, alpha, theta);

  }
  */



  double gaussian_prob(double x, double sigma) {

    return 0.5*(1.0 - ROOT::Math::erf(x/(sigma*std::sqrt(2.0))));

  }



  double gaussian_quant(double x, double sigma) {

    return 0.5*(1.0 + ROOT::Math::erf(x/(sigma*std::sqrt(2.0))));

  }


  
  double lognormal_prob(double x, double m, double s) {

    return 0.5*(1.0 - ROOT::Math::erf((std::log(x)-m)/(s*std::sqrt(2.0))));

  }



  double lognormal_quant(double x, double m, double s) {

    return 0.5*(1.0 + ROOT::Math::erf((std::log(x)-m)/(s*std::sqrt(2.0))));

  }
  


  double normal_prob(double x, double sigma) {

    return 0.5*(1.0 - ROOT::Math::erf(x/(sigma*std::sqrt(2.0))));

  }



  double normal_quant(double x, double sigma) {

    return 0.5*(1 + ROOT::Math::erf(x/(sigma*std::sqrt(2.0))));

  }


  /**
  double tdistribution_prob(double x, double r) {

    return gsl_cdf_tdist_Q(x, r);

  }



  double tdistribution_quant(double x, double r) {

    return gsl_cdf_tdist_P(x, r);

  }
  */


  double uniform_prob(double x, double a, double b) {

    if (x < a) {
      return 1.0;
    } else if (x >= b) {
      return 0.0;
    } else {
      return (b-x)/(b-a);
    }
  }



  double uniform_quant(double x, double a, double b) {

    if (x < a) {
      return 0.0;
    } else if (x >= b) {
      return 1.0;
    } else {
      return (x-a)/(b-a);
    }    
  }





} // namespace Math
} // namespace ROOT



