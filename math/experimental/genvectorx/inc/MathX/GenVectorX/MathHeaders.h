/*
 * Project: Math
 * Authors:
 *   Monica Dessole, CERN, 2024
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_MathHeaders_H
#define ROOT_MathHeaders_H

#include "MathX/GenVectorX/AccHeaders.h"

#include <limits>

namespace ROOT {

namespace ROOT_MATH_ARCH {

#if defined(ROOT_MATH_SYCL)
template <class Scalar>
inline Scalar math_fmod(Scalar x, Scalar y)
{
   return sycl::fmod(x, y);
}

template <class Scalar>
inline Scalar math_sin(Scalar x)
{
   return sycl::sin(x);
}

template <class Scalar>
inline Scalar math_cos(Scalar x)
{
   return sycl::cos(x);
}

template <class Scalar>
inline Scalar math_asin(Scalar x)
{
   return sycl::asin(x);
}

template <class Scalar>
inline Scalar math_acos(Scalar x)
{
   return sycl::acos(x);
}

template <class Scalar>
inline Scalar math_sinh(Scalar x)
{
   return sycl::sinh(x);
}

template <class Scalar>
inline Scalar math_cosh(Scalar x)
{
   return sycl::cosh(x);
}

template <class Scalar>
inline Scalar math_atan2(Scalar x, Scalar y)
{
   return sycl::atan2(x, y);
}

template <class Scalar>
inline Scalar math_atan(Scalar x)
{
   return sycl::atan(x);
}

template <class Scalar>
inline Scalar math_sqrt(Scalar x)
{
   return sycl::sqrt(x);
}

template <class Scalar>
inline Scalar math_floor(Scalar x)
{
   return sycl::floor(x);
}

template <class Scalar>
inline Scalar math_exp(Scalar x)
{
   return sycl::exp(x);
}

template <class Scalar>
inline Scalar math_log(Scalar x)
{
   return sycl::log(x);
}

inline long double math_log(long double x)
{
   double castx = x;
   double castres = sycl::log(castx);
   return (long double)castres;
}

template <class Scalar>
inline Scalar math_tan(Scalar x)
{
   return sycl::tan(x);
}

template <class Scalar>
inline Scalar math_fabs(Scalar x)
{
   return sycl::fabs(x);
}

template <class Scalar>
inline Scalar math_pow(Scalar x, Scalar y)
{
   return sycl::pow(x, y);
}

// template <class T>
// T etaMax2()
// {
//    return static_cast<T>(22756.0);
// }

// template <typename Scalar>
// inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
// {
//    if (rho > 0) {
//       // value to control Taylor expansion of sqrt
//       // static const Scalar
//       Scalar epsilon = static_cast<Scalar>(2e-16);
//       const Scalar big_z_scaled = sycl::pow(epsilon, static_cast<Scalar>(-.25));

//       Scalar z_scaled = z / rho;
//       if (sycl::fabs(z_scaled) < big_z_scaled) {
//          return sycl::log(z_scaled + sycl::sqrt(z_scaled * z_scaled + 1.0));
//       } else {
//          // apply correction using first order Taylor expansion of sqrt
//          return z > 0 ? sycl::log(2.0 * z_scaled + 0.5 / z_scaled) : -sycl::log(-2.0 * z_scaled);
//       }
//       return z_scaled;
//    }
//    // case vector has rho = 0
//    else if (z == 0) {
//       return 0;
//    } else if (z > 0) {
//       return z + etaMax2<Scalar>();
//    } else {
//       return z - etaMax2<Scalar>();
//    }
// }

// /**
//    Implementation of eta from -log(tan(theta/2)).
//    This is convenient when theta is already known (for example in a polar
//    coorindate system)
// */
// template <typename Scalar>
// inline Scalar Eta_FromTheta(Scalar theta, Scalar r)
// {
//    Scalar tanThetaOver2 = tan(theta / 2.);
//    if (tanThetaOver2 == 0) {
//       return r + etaMax2<Scalar>();
//    } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
//       return -r - etaMax2<Scalar>();
//    } else {
//       return -log(tanThetaOver2);
//    }
// }

#elif defined(ROOT_MATH_CUDA)
template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_fmod(Scalar x, Scalar y)
{
   return std::fmod(x, y);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_sin(Scalar x)
{
   return std::sin(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_cos(Scalar x)
{
   return std::cos(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_asin(Scalar x)
{
   return std::asin(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_acos(Scalar x)
{
   return std::acos(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_sinh(Scalar x)
{
   return std::sinh(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_cosh(Scalar x)
{
   return std::cosh(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_atan2(Scalar x, Scalar y)
{
   return std::atan2(x, y);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_atan(Scalar x)
{
   return std::atan(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_sqrt(Scalar x)
{
   return std::sqrt(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_floor(Scalar x)
{
   return std::floor(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_exp(Scalar x)
{
   return std::exp(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_log(Scalar x)
{
   return std::log(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_tan(Scalar x)
{
   return std::tan(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_fabs(Scalar x)
{
   return std::fabs(x);
}

template <class Scalar>
__roohost__ __roodevice__ inline Scalar math_pow(Scalar x, Scalar y)
{
   return std::pow(x, y);
}

template <class T>
__roohost__ __roodevice__ inline T etaMax2()
{
   return static_cast<T>(22756.0);
}

template <typename Scalar>
__roohost__ __roodevice__ inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
{
   if (rho > 0) {
      // value to control Taylor expansion of sqrt
      // static const Scalar
      Scalar epsilon = static_cast<Scalar>(2e-16);
      const Scalar big_z_scaled = pow(epsilon, static_cast<Scalar>(-.25));

      Scalar z_scaled = z / rho;
      if (fabs(z_scaled) < big_z_scaled) {
         return log(z_scaled + sqrt(z_scaled * z_scaled + 1.0));
      } else {
         // apply correction using first order Taylor expansion of sqrt
         return z > 0 ? log(2.0 * z_scaled + 0.5 / z_scaled) : log(-2.0 * z_scaled);
      }
      return z_scaled;
   }
   // case vector has rho = 0
   else if (z == 0) {
      return 0;
   } else if (z > 0) {
      return z + etaMax2<Scalar>();
   } else {
      return z - etaMax2<Scalar>();
   }
}

/**
   Implementation of eta from -log(tan(theta/2)).
   This is convenient when theta is already known (for example in a polar
   coorindate system)
*/
template <typename Scalar>
__roohost__ __roodevice__ inline Scalar Eta_FromTheta(Scalar theta, Scalar r)
{
   Scalar tanThetaOver2 = tan(theta / 2.);
   if (tanThetaOver2 == 0) {
      return r + etaMax2<Scalar>();
   } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -r - etaMax2<Scalar>();
   } else {
      return -log(tanThetaOver2);
   }
}

#else

template <class Scalar>
inline Scalar math_fmod(Scalar x, Scalar y)
{
   return std::fmod(x, y);
}

template <class Scalar>
inline Scalar math_sin(Scalar x)
{
   return std::sin(x);
}

template <class Scalar>
inline Scalar math_cos(Scalar x)
{
   return std::cos(x);
}

template <class Scalar>
inline Scalar math_asin(Scalar x)
{
   return std::asin(x);
}

template <class Scalar>
inline Scalar math_acos(Scalar x)
{
   return std::acos(x);
}

template <class Scalar>
inline Scalar math_sinh(Scalar x)
{
   return std::sinh(x);
}

template <class Scalar>
inline Scalar math_cosh(Scalar x)
{
   return std::cosh(x);
}

template <class Scalar>
inline Scalar math_atan2(Scalar x, Scalar y)
{
   return std::atan2(x, y);
}

template <class Scalar>
inline Scalar math_atan(Scalar x)
{
   return std::atan(x);
}

template <class Scalar>
inline Scalar math_sqrt(Scalar x)
{
   return std::sqrt(x);
}

template <class Scalar>
inline Scalar math_floor(Scalar x)
{
   return std::floor(x);
}

template <class Scalar>
inline Scalar math_exp(Scalar x)
{
   return std::exp(x);
}

template <class Scalar>
inline Scalar math_log(Scalar x)
{
   return std::log(x);
}

template <class Scalar>
inline Scalar math_tan(Scalar x)
{
   return std::tan(x);
}

template <class Scalar>
inline Scalar math_fabs(Scalar x)
{
   return std::fabs(x);
}

template <class Scalar>
inline Scalar math_pow(Scalar x, Scalar y)
{
   return std::pow(x, y);
}

template <class T>
inline T etaMax2()
{
   return static_cast<T>(22756.0);
}

template <typename Scalar>
inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
{
   if (rho > 0) {

      // value to control Taylor expansion of sqrt
      static const Scalar big_z_scaled = pow(std::numeric_limits<Scalar>::epsilon(), static_cast<Scalar>(-.25));

      Scalar z_scaled = z / rho;
      if (fabs(z_scaled) < big_z_scaled) {
         return log(z_scaled + sqrt(z_scaled * z_scaled + 1.0));
      } else {
         // apply correction using first order Taylor expansion of sqrt
         return z > 0 ? log(2.0 * z_scaled + 0.5 / z_scaled) : -log(-2.0 * z_scaled);
      }
   }
   // case vector has rho = 0
   else if (z == 0) {
      return 0;
   } else if (z > 0) {
      return z + etaMax2<Scalar>();
   } else {
      return z - etaMax2<Scalar>();
   }
}

/**
   Implementation of eta from -log(tan(theta/2)).
   This is convenient when theta is already known (for example in a polar
   coorindate system)
*/
template <typename Scalar>
inline Scalar Eta_FromTheta(Scalar theta, Scalar r)
{
   Scalar tanThetaOver2 = tan(theta / 2.);
   if (tanThetaOver2 == 0) {
      return r + etaMax2<Scalar>();
   } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -r - etaMax2<Scalar>();
   } else {
      return -log(tanThetaOver2);
   }
}

#endif

} // namespace ROOT_MATH_ARCH

} // end namespace ROOT

#endif