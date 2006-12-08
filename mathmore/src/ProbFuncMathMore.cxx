// @(#)root/mathmore:$Name:  $:$Id: ProbFuncMathMore.cxx,v 1.4 2006/12/06 17:53:47 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 


#include <cmath>
#include "Math/ProbFuncMathMore.h"
#include "gsl/gsl_cdf.h"


namespace ROOT {
namespace Math {

  





  double chisquared_cdf_c(double x, double r, double x0) {

    return gsl_cdf_chisq_Q(x-x0, r);

  }



  double chisquared_cdf(double x, double r, double x0) {

    return gsl_cdf_chisq_P(x-x0, r);

  }


  
  double fdistribution_cdf_c(double x, double n, double m, double x0) {

    return gsl_cdf_fdist_Q(x-x0, n, m);

  }



  double fdistribution_cdf(double x, double n, double m, double x0) {

    return gsl_cdf_fdist_P(x-x0, n, m);

  }



  double gamma_cdf_c(double x, double alpha, double theta, double x0) {

    return gsl_cdf_gamma_Q(x-x0, alpha, theta);

  }



  double gamma_cdf(double x, double alpha, double theta, double x0) {

    return gsl_cdf_gamma_P(x-x0, alpha, theta);

  }





  double tdistribution_cdf_c(double x, double r, double x0) {

    return gsl_cdf_tdist_Q(x-x0, r);

  }



  double tdistribution_cdf(double x, double r, double x0) {

    return gsl_cdf_tdist_P(x-x0, r);

  }


  double beta_cdf_c(double x, double a, double b) {

    return gsl_cdf_beta_Q(x, a, b);

  }


  double beta_cdf(double x, double a, double b ) {

    return gsl_cdf_beta_P(x, a, b);

  }

   double poisson_cdf_c(unsigned int n, double mu) {

          return gsl_cdf_poisson_Q(n, mu);

   }

   double poisson_cdf(unsigned int n, double mu) {

          return gsl_cdf_poisson_P(n, mu);

   }

   double binomial_cdf_c(unsigned int k, double p, unsigned int n) {

          return gsl_cdf_binomial_Q(k, p, n);

   }

   double binomial_cdf(unsigned int k, double p, unsigned int n) {

          return gsl_cdf_binomial_P(k, p, n);

   }

} // namespace Math
} // namespace ROOT



