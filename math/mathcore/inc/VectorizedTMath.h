#ifndef ROOT_VectorizedTMath
#define ROOT_VectorizedTMath

#include "Math/Types.h"
#include "RConfigure.h" // for R__HAS_STD_EXPERIMENTAL_SIMD

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD

#include <cmath>

namespace TMath {

template <class T, class Abi>
auto Log2(std::experimental::simd<T, Abi> &x)
{
   return log2(x);
}

/// Calculate a Breit Wigner function with mean and gamma.
template <class T, class Abi>
auto BreitWigner(std::experimental::simd<T, Abi> &x, double mean = 0, double gamma = 1)
{
   return 0.5 * M_1_PI * (gamma / (0.25 * gamma * gamma + (x - mean) * (x - mean)));
}

/// Calculate a gaussian function with mean and sigma.
/// If norm=kTRUE (default is kFALSE) the result is divided
/// by sqrt(2*Pi)*sigma.
template <class T, class Abi>
auto Gaus(std::experimental::simd<T, Abi> &x, double mean = 0, double sigma = 1, Bool_t norm = false)
{
   if (sigma == 0)
      return std::experimental::simd<T, Abi>{1.e30};

   auto inv_sigma = 1.0 / std::experimental::simd<T, Abi>(sigma);
   auto arg = (x - std::experimental::simd<T, Abi>(mean)) * inv_sigma;

   // For those entries of |arg| > 39 result is zero in double precision
   std::experimental::simd<T, Abi> out{};
   where(abs(arg) < 39.0, out) = exp(std::experimental::simd<T, Abi>(-0.5) * arg * arg);

   if (norm)
      out *= 0.3989422804014327 * inv_sigma; // 1/sqrt(2*Pi)=0.3989422804014327
   return out;
}

/// Computes the probability density function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
template <class T, class Abi>
auto LaplaceDist(std::experimental::simd<T, Abi> &x, double alpha = 0, double beta = 1)
{
   auto beta_v_inv = std::experimental::simd<T, Abi>(1.0 / beta);
   auto out = exp(-abs((x - alpha) * beta_v_inv));
   out *= 0.5 * beta_v_inv;
   return out;
}

/// Computes the distribution function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
template <class T, class Abi>
auto LaplaceDistI(std::experimental::simd<T, Abi> &x, double alpha = 0, double beta = 1)
{
   auto alpha_v = std::experimental::simd<T, Abi>(alpha);
   auto beta_v_inv = std::experimental::simd<T, Abi>(1.0) / std::experimental::simd<T, Abi>(beta);
   auto mask = x <= alpha_v;
   std::experimental::simd<T, Abi> out{};
   where(mask, out) = 0.5 * exp(-abs((x - alpha_v) * beta_v_inv));
   where(!mask, out) = 1 - 0.5 * exp(-abs((x - alpha_v) * beta_v_inv));
   return out;
}

/// Computation of the normal frequency function freq(x).
/// Freq(x) = (1/sqrt(2pi)) Integral(exp(-t^2/2))dt between -infinity and x.
///
/// Translated from CERNLIB C300 by Rene Brun.
template <class T, class Abi>
auto Freq(std::experimental::simd<T, Abi> &x)
{
   double c1 = 0.56418958354775629;
   double w2 = 1.41421356237309505;

   double p10 = 2.4266795523053175e+2, q10 = 2.1505887586986120e+2, p11 = 2.1979261618294152e+1,
          q11 = 9.1164905404514901e+1, p12 = 6.9963834886191355e+0, q12 = 1.5082797630407787e+1,
          p13 = -3.5609843701815385e-2;

   double p20 = 3.00459261020161601e+2, q20 = 3.00459260956983293e+2, p21 = 4.51918953711872942e+2,
          q21 = 7.90950925327898027e+2, p22 = 3.39320816734343687e+2, q22 = 9.31354094850609621e+2,
          p23 = 1.52989285046940404e+2, q23 = 6.38980264465631167e+2, p24 = 4.31622272220567353e+1,
          q24 = 2.77585444743987643e+2, p25 = 7.21175825088309366e+0, q25 = 7.70001529352294730e+1,
          p26 = 5.64195517478973971e-1, q26 = 1.27827273196294235e+1, p27 = -1.36864857382716707e-7;

   double p30 = -2.99610707703542174e-3, q30 = 1.06209230528467918e-2, p31 = -4.94730910623250734e-2,
          q31 = 1.91308926107829841e-1, p32 = -2.26956593539686930e-1, q32 = 1.05167510706793207e+0,
          p33 = -2.78661308609647788e-1, q33 = 1.98733201817135256e+0, p34 = -2.23192459734184686e-2, q34 = 1;

   auto v = abs(x) / w2;

   std::experimental::simd<T, Abi> result{};

   auto mask1 = v < 0.5;
   auto mask2 = !mask1 && v < 4.0;
   auto mask3 = !(mask1 || mask2);

   auto v2 = v * v;
   auto v3 = v2 * v;
   auto v4 = v3 * v;
   auto v5 = v4 * v;
   auto v6 = v5 * v;
   auto v7 = v6 * v;
   auto v8 = v7 * v;

   where(mask1, result) = v * (p10 + p11 * v2 + p12 * v4 + p13 * v6) / (q10 + q11 * v2 + q12 * v4 + v6);
   where(mask2, result) =
      1.0 - (p20 + p21 * v + p22 * v2 + p23 * v3 + p24 * v4 + p25 * v5 + p26 * v6 + p27 * v7) /
               (exp(v2) * (q20 + q21 * v + q22 * v2 + q23 * v3 + q24 * v4 + q25 * v5 + q26 * v6 + v7));
   where(mask3, result) = 1.0 - (c1 + (p30 * v8 + p31 * v6 + p32 * v4 + p33 * v2 + p34) /
                                         ((q30 * v8 + q31 * v6 + q32 * v4 + q33 * v2 + q34) * v2)) /
                                   (v * exp(v2));

   auto out = 0.5 * (1 - result);
   where(x > 0, out) = 0.5 + 0.5 * result;
   return out;
}

/// Vectorized implementation of Bessel function I_0(x) for a vector x.
template <class T, class Abi>
auto BesselI0_Split_More(std::experimental::simd<T, Abi> &ax)
{
   auto y = 3.75 / ax;
   return (exp(ax) / sqrt(ax)) *
          (0.39894228 +
           y * (1.328592e-2 +
                y * (2.25319e-3 +
                     y * (-1.57565e-3 +
                          y * (9.16281e-3 +
                               y * (-2.057706e-2 + y * (2.635537e-2 + y * (-1.647633e-2 + y * 3.92377e-3))))))));
}

template <class T, class Abi>
auto BesselI0_Split_Less(std::experimental::simd<T, Abi> &x)
{
   auto y = x * x * 0.071111111;

   return 1.0 +
          y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (3.60768e-2 + y * 4.5813e-3)))));
}

template <class T, class Abi>
auto BesselI0(std::experimental::simd<T, Abi> &x)
{
   auto ax = abs(x);

   auto out = BesselI0_Split_More(ax);
   where(ax <= 3.75, out) = BesselI0_Split_Less(x);
   return out;
}

///  Vectorized implementation of modified Bessel function I_1(x) for a vector x.
template <class T, class Abi>
auto BesselI1_Split_More(std::experimental::simd<T, Abi> &ax, std::experimental::simd<T, Abi> &x)
{
   auto y = 3.75 / ax;
   auto result = (exp(ax) / sqrt(ax)) *
                 (0.39894228 +
                  y * (-3.988024e-2 +
                       y * (-3.62018e-3 +
                            y * (1.63801e-3 +
                                 y * (-1.031555e-2 +
                                      y * (2.282967e-2 + y * (-2.895312e-2 + y * (1.787654e-2 + y * -4.20059e-3))))))));
   where(x < 0, result) = -result;
   return result;
}

template <class T, class Abi>
auto BesselI1_Split_Less(std::experimental::simd<T, Abi> &x)
{
   auto y = x * x * 0.071111111;

   return x * (0.5 + y * (0.87890594 +
                          y * (0.51498869 + y * (0.15084934 + y * (2.658733e-2 + y * (3.01532e-3 + y * 3.2411e-4))))));
}

template <class T, class Abi>
auto BesselI1(std::experimental::simd<T, Abi> &x)
{
   auto ax = abs(x);

   auto out = BesselI1_Split_More(ax, x);
   where(ax <= 3.75, out) = BesselI1_Split_Less(x);
   return out;
}

///  Vectorized implementation of Bessel function J0(x) for a vector x.
template <class T, class Abi>
auto BesselJ0_Split1_More(std::experimental::simd<T, Abi> &ax)
{
   auto z = 8 / ax;
   auto y = z * z;
   auto xx = ax - 0.785398164;
   auto result1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
   auto result2 =
      -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
   return sqrt(0.636619772 / ax) * (cos(xx) * result1 - z * sin(xx) * result2);
}

template <class T, class Abi>
auto BesselJ0_Split1_Less(std::experimental::simd<T, Abi> &x)
{
   auto y = x * x;
   return (57568490574.0 +
           y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))) /
          (57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))));
}

template <class T, class Abi>
auto BesselJ0(std::experimental::simd<T, Abi> &x)
{
   auto ax = abs(x);
   auto out = BesselJ0_Split1_More(ax);
   where(ax < 8, out) = BesselJ0_Split1_Less(x);
   return out;
}

///  Vectorized implementation of Bessel function J1(x) for a vector x.
template <class T, class Abi>
auto BesselJ1_Split1_More(std::experimental::simd<T, Abi> &ax, std::experimental::simd<T, Abi> &x)
{
   auto z = 8 / ax;
   auto y = z * z;
   auto xx = ax - 2.356194491;
   auto result1 = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
   auto result2 =
      0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 - y * 0.105787412e-6)));
   auto result = sqrt(0.636619772 / ax) * (cos(xx) * result1 - z * sin(xx) * result2);
   where(x < 0, result) = -result;
   return result;
}

template <class T, class Abi>
auto BesselJ1_Split1_Less(std::experimental::simd<T, Abi> &x)
{
   auto y = x * x;
   return x *
          (72362614232.0 +
           y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))))) /
          (144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))));
}

template <class T, class Abi>
auto BesselJ1(std::experimental::simd<T, Abi> &x)
{
   auto ax = abs(x);
   auto out = BesselJ1_Split1_More(ax, x);
   where(ax < 8, out) = BesselJ1_Split1_Less(x);
   return out;
}

} // namespace TMath

#endif // R__HAS_STD_EXPERIMENTAL_SIMD

#endif
