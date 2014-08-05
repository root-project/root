// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

#include "gsl/gsl_cdf.h"


namespace ROOT {
namespace Math {



  double tdistribution_quantile_c(double z, double r) {

    return gsl_cdf_tdist_Qinv(z, r);

  }



  double tdistribution_quantile(double z, double r) {

    return gsl_cdf_tdist_Pinv(z, r);

  }

} // namespace Math

namespace MathMore  {
   // re-impelment some function already existing in MathCore (defined in ROOT::Math namespace)

  double chisquared_quantile(double z, double r) {

    return gsl_cdf_chisq_Pinv(z, r);

  }


  double gamma_quantile(double z, double alpha, double theta) {

    return gsl_cdf_gamma_Pinv(z, alpha, theta);

  }

} // namespace MathMore

} // namespace ROOT
