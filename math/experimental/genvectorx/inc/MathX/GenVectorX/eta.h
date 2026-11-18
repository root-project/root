// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , FNAL MathLib Team                             *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header source file for function calculating eta
//
// Created by: Lorenzo Moneta  at 14 Jun 2007

#ifndef ROOT_MathX_GenVectorX_eta
#define ROOT_MathX_GenVectorX_eta 1

#include "MathX/GenVectorX/etaMax.h"

#include "MathX/GenVectorX/MathHeaders.h"

#include "MathX/GenVectorX/AccHeaders.h"

using namespace ROOT::ROOT_MATH_ARCH;

#include <limits>
#include <cmath>

namespace ROOT {

namespace ROOT_MATH_ARCH {

namespace Impl {

/**
    Calculate eta given rho and zeta.
    This formula is faster than the standard calculation (below) from log(tan(theta/2)
    but one has to be careful when rho is much smaller than z (large eta values)
    Formula is  eta = log( zs + sqrt(zs^2 + 1) )  where zs = z/rho

    For large value of z_scaled (tan(theta) ) one can approximate the sqrt via a Taylor expansion
    We do the approximation of the sqrt if the numerical error is of the same order of second term of
    the sqrt.expansion:
    eps > 1/zs^4   =>   zs > 1/(eps^0.25)

    When rho == 0 we use etaMax (see definition in etaMax.h)

 */
template <typename Scalar>
inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
{
   if (rho > 0) {

      // value to control Taylor expansion of sqrt
      Scalar big_z_scaled = math_pow(std::numeric_limits<Scalar>::epsilon(), static_cast<Scalar>(-.25));

      Scalar z_scaled = z / rho;

      // eta is odd in zs: use |zs| in the formula, which then has no cancellation, and apply the sign of zs.
      // For large |zs|, use the first order Taylor expansion of sqrt.
      const Scalar abs_z_scaled = math_fabs(z_scaled);
      const Scalar abs_eta = abs_z_scaled < big_z_scaled
                                ? math_log(abs_z_scaled + math_sqrt(abs_z_scaled * abs_z_scaled + 1.0))
                                : math_log(2.0 * abs_z_scaled + 0.5 / abs_z_scaled);
      return z_scaled < 0 ? -abs_eta : abs_eta;
   }
   // case vector has rho = 0
   else if (z == 0) {
      return 0;
   } else if (z > 0) {
      return z + etaMax<Scalar>();
   } else {
      return z - etaMax<Scalar>();
   }
}

/**
   Implementation of eta from -log(tan(theta/2)).
   This is convenient when theta is already known (for example in a polar coorindate system)
*/
template <typename Scalar>
inline Scalar Eta_FromTheta(Scalar theta, Scalar r)
{
   Scalar tanThetaOver2 = tan(theta / 2.);
   if (tanThetaOver2 == 0) {
      return r + etaMax<Scalar>();
   } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -r - etaMax<Scalar>();
   } else {
      return -math_log(tanThetaOver2);
   }
}

} // end namespace Impl

} // namespace ROOT_MATH_ARCH

} // namespace ROOT

#endif /* ROOT_MathX_GenVectorX_etaMax  */
