// @(#)root/mathmore:$Name:  $:$Id: ProbFuncInv.cxxv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

#include "gsl/gsl_cdf.h"


namespace ROOT {
namespace Math {



  double breitwigner_prob_inv(double z, double gamma) {

    return gsl_cdf_cauchy_Qinv(z, gamma/2.0);

  }



  double breitwigner_quant_inv(double z, double gamma) {

    return gsl_cdf_cauchy_Pinv(z, gamma/2.0);

  }



  double cauchy_prob_inv(double z, double gamma) {

    return gsl_cdf_cauchy_Qinv(z, gamma/2.0);

  }



  double cauchy_quant_inv(double z, double gamma) {

    return gsl_cdf_cauchy_Pinv(z, gamma/2.0);

  }



  double chisquared_prob_inv(double z, double r) {

    return gsl_cdf_chisq_Qinv(z, r);

  }



  double chisquared_quant_inv(double z, double r) {

    return gsl_cdf_chisq_Pinv(z, r);

  }



  double exponential_prob_inv(double z, double lambda) {

    return gsl_cdf_exponential_Qinv(z, 1.0/lambda);

  }



  double exponential_quant_inv(double z, double lambda) {

    return gsl_cdf_exponential_Pinv(z, 1.0/lambda);

  }



  double gamma_prob_inv(double z, double alpha, double theta) {

    return gsl_cdf_gamma_Qinv(z, alpha, theta);

  }

  double gamma_quant_inv(double z, double alpha, double theta) {

    return gsl_cdf_gamma_Pinv(z, alpha, theta);

  }



  double gaussian_prob_inv(double z, double sigma) {

    return gsl_cdf_gaussian_Qinv(z, sigma);

  }



  double gaussian_quant_inv(double z, double sigma) {

    return gsl_cdf_gaussian_Pinv(z, sigma);

  }



  double lognormal_prob_inv(double x, double m, double s) {

    return gsl_cdf_lognormal_Qinv(x, m, s);

  }



  double lognormal_quant_inv(double x, double m, double s) {

    return gsl_cdf_lognormal_Pinv(x, m, s);

  }



  double normal_prob_inv(double z, double sigma) {

    return gsl_cdf_gaussian_Qinv(z, sigma);

  }



  double normal_quant_inv(double z, double sigma) {

    return gsl_cdf_gaussian_Pinv(z, sigma);

  }



  double tdistribution_prob_inv(double z, double r) {

    return gsl_cdf_tdist_Qinv(z, r);

  }



  double tdistribution_quant_inv(double z, double r) {

    return gsl_cdf_tdist_Pinv(z, r);

  }



  double uniform_prob_inv(double z, double a, double b) {

    return gsl_cdf_flat_Qinv(z, a, b);

  }



  double uniform_quant_inv(double z, double a, double b) {

    return gsl_cdf_flat_Pinv(z, a, b);

  }





} // namespace Math
} // namespace ROOT
