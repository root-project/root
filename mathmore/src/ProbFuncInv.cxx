// @(#)root/mathmore:$Name:  $:$Id: ProbFuncInv.cxx,v 1.3 2006/12/06 17:53:47 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

#include "gsl/gsl_cdf.h"


namespace ROOT {
namespace Math {



  double breitwigner_quantile_c(double z, double gamma) {

    return gsl_cdf_cauchy_Qinv(z, gamma/2.0);

  }



  double breitwigner_quantile(double z, double gamma) {

    return gsl_cdf_cauchy_Pinv(z, gamma/2.0);

  }



  double cauchy_quantile_c(double z, double b) {

    return gsl_cdf_cauchy_Qinv(z, b);

  }



  double cauchy_quantile(double z, double b) {

    return gsl_cdf_cauchy_Pinv(z, b);

  }



  double chisquared_quantile_c(double z, double r) {

    return gsl_cdf_chisq_Qinv(z, r);

  }



  double chisquared_quantile(double z, double r) {

    return gsl_cdf_chisq_Pinv(z, r);

  }



  double exponential_quantile_c(double z, double lambda) {

    return gsl_cdf_exponential_Qinv(z, 1.0/lambda);

  }



  double exponential_quantile(double z, double lambda) {

    return gsl_cdf_exponential_Pinv(z, 1.0/lambda);

  }


  double fdistribution_quantile_c(double z, double n, double m) {

    return gsl_cdf_fdist_Qinv(z, n, m);

  }

  double fdistribution_quantile(double z, double n, double m) {

    return gsl_cdf_fdist_Pinv(z, n, m);

  }


  double gamma_quantile_c(double z, double alpha, double theta) {

    return gsl_cdf_gamma_Qinv(z, alpha, theta);

  }

  double gamma_quantile(double z, double alpha, double theta) {

    return gsl_cdf_gamma_Pinv(z, alpha, theta);

  }



  double gaussian_quantile_c(double z, double sigma) {

    return gsl_cdf_gaussian_Qinv(z, sigma);

  }



  double gaussian_quantile(double z, double sigma) {

    return gsl_cdf_gaussian_Pinv(z, sigma);

  }



  double lognormal_quantile_c(double x, double m, double s) {

    return gsl_cdf_lognormal_Qinv(x, m, s);

  }



  double lognormal_quantile(double x, double m, double s) {

    return gsl_cdf_lognormal_Pinv(x, m, s);

  }



  double normal_quantile_c(double z, double sigma) {

    return gsl_cdf_gaussian_Qinv(z, sigma);

  }



  double normal_quantile(double z, double sigma) {

    return gsl_cdf_gaussian_Pinv(z, sigma);

  }



  double tdistribution_quantile_c(double z, double r) {

    return gsl_cdf_tdist_Qinv(z, r);

  }



  double tdistribution_quantile(double z, double r) {

    return gsl_cdf_tdist_Pinv(z, r);

  }



  double uniform_quantile_c(double z, double a, double b) {

    return gsl_cdf_flat_Qinv(z, a, b);

  }



  double uniform_quantile(double z, double a, double b) {

    return gsl_cdf_flat_Pinv(z, a, b);

  }


  double beta_quantile_c(double x, double a, double b) {

    return gsl_cdf_beta_Qinv(x, a, b);

  }


  double beta_quantile(double x, double a, double b ) {

    return gsl_cdf_beta_Pinv(x, a, b);

  }




} // namespace Math
} // namespace ROOT
