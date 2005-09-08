// @(#)root/mathmore:$Name:  $:$Id: ProbFunc.cxxv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

// @(#)Root/mathcore:$Name:  $:$Id: ProbFunc.cxx,v 1.1 2005/06/28 09:55:13 brun Exp $
// Authors: L. Moneta, A. Zsenei   06/2005 


#include <cmath>
#include "MathMore/ProbFunc.h"
#include "cdf/gsl_cdf.h"


namespace ROOT {
namespace Math {

  




  double chisquared_prob(double x, double r, double x0 = 0) {

    return gsl_cdf_chisq_Q(x-x0, r);

  }



  double chisquared_quant(double x, double r, double x0 = 0) {

    return gsl_cdf_chisq_P(x-x0, r);

  }





  
  double fdistribution_prob(double x, double n, double m, double x0 = 0) {

    return gsl_cdf_fdist_Q(x-x0, n, m);

  }



  double fdistribution_quant(double x, double n, double m, double x0 = 0) {

    return gsl_cdf_fdist_P(x-x0, n, m);

  }



  double gamma_prob(double x, double alpha, double theta, double x0 = 0) {

    return gsl_cdf_gamma_Q(x-x0, alpha, theta);

  }



  double gamma_quant(double x, double alpha, double theta, double x0 = 0) {

    return gsl_cdf_gamma_P(x-x0, alpha, theta);

  }





  double tdistribution_prob(double x, double r, double x0 = 0) {

    return gsl_cdf_tdist_Q(x-x0, r);

  }



  double tdistribution_quant(double x, double r, double x0 = 0) {

    return gsl_cdf_tdist_P(x-x0, r);

  }





} // namespace Math
} // namespace ROOT



