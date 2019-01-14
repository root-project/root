#include "VectorizedTMath.h"

#if defined(R__HAS_VECCORE) && defined(R__HAS_VC)

namespace TMath {
////////////////////////////////////////////////////////////////////////////////
::ROOT::Double_v Log2(::ROOT::Double_v &x)
{
   return vecCore::math::Log2(x);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate a Breit Wigner function with mean and gamma.
::ROOT::Double_v BreitWigner(::ROOT::Double_v &x, Double_t mean, Double_t gamma)
{
   return 0.5 * M_1_PI * (gamma / (0.25 * gamma * gamma + (x - mean) * (x - mean)));
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate a gaussian function with mean and sigma.
/// If norm=kTRUE (default is kFALSE) the result is divided
/// by sqrt(2*Pi)*sigma.
::ROOT::Double_v Gaus(::ROOT::Double_v &x, Double_t mean, Double_t sigma, Bool_t norm)
{
   if (sigma == 0)
      return ::ROOT::Double_v(1.e30);

   ::ROOT::Double_v inv_sigma = 1.0 / ::ROOT::Double_v(sigma);
   ::ROOT::Double_v arg = (x - ::ROOT::Double_v(mean)) * inv_sigma;

   // For those entries of |arg| > 39 result is zero in double precision
   ::ROOT::Double_v out =
      vecCore::Blend<::ROOT::Double_v>(vecCore::math::Abs(arg) < ::ROOT::Double_v(39.0),
                                       vecCore::math::Exp(::ROOT::Double_v(-0.5) * arg * arg), ::ROOT::Double_v(0.0));
   if (norm)
      out *= 0.3989422804014327 * inv_sigma; // 1/sqrt(2*Pi)=0.3989422804014327
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the probability density function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
::ROOT::Double_v LaplaceDist(::ROOT::Double_v &x, Double_t alpha, Double_t beta)
{
   ::ROOT::Double_v beta_v_inv = ::ROOT::Double_v(1.0) / ::ROOT::Double_v(beta);
   ::ROOT::Double_v out = vecCore::math::Exp(-vecCore::math::Abs((x - ::ROOT::Double_v(alpha)) * beta_v_inv));
   out *= ::ROOT::Double_v(0.5) * beta_v_inv;
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the distribution function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
::ROOT::Double_v LaplaceDistI(::ROOT::Double_v &x, Double_t alpha, Double_t beta)
{
   ::ROOT::Double_v alpha_v = ::ROOT::Double_v(alpha);
   ::ROOT::Double_v beta_v_inv = ::ROOT::Double_v(1.0) / ::ROOT::Double_v(beta);
   return vecCore::Blend<::ROOT::Double_v>(
      x <= alpha_v, 0.5 * vecCore::math::Exp(-vecCore::math::Abs((x - alpha_v) * beta_v_inv)),
      1 - 0.5 * vecCore::math::Exp(-vecCore::math::Abs((x - alpha_v) * beta_v_inv)));
}

////////////////////////////////////////////////////////////////////////////////
/// Computation of the normal frequency function freq(x).
/// Freq(x) = (1/sqrt(2pi)) Integral(exp(-t^2/2))dt between -infinity and x.
///
/// Translated from CERNLIB C300 by Rene Brun.
::ROOT::Double_v Freq(::ROOT::Double_v &x)
{
   Double_t c1 = 0.56418958354775629;
   Double_t w2 = 1.41421356237309505;

   Double_t p10 = 2.4266795523053175e+2, q10 = 2.1505887586986120e+2, p11 = 2.1979261618294152e+1,
            q11 = 9.1164905404514901e+1, p12 = 6.9963834886191355e+0, q12 = 1.5082797630407787e+1,
            p13 = -3.5609843701815385e-2;

   Double_t p20 = 3.00459261020161601e+2, q20 = 3.00459260956983293e+2, p21 = 4.51918953711872942e+2,
            q21 = 7.90950925327898027e+2, p22 = 3.39320816734343687e+2, q22 = 9.31354094850609621e+2,
            p23 = 1.52989285046940404e+2, q23 = 6.38980264465631167e+2, p24 = 4.31622272220567353e+1,
            q24 = 2.77585444743987643e+2, p25 = 7.21175825088309366e+0, q25 = 7.70001529352294730e+1,
            p26 = 5.64195517478973971e-1, q26 = 1.27827273196294235e+1, p27 = -1.36864857382716707e-7;

   Double_t p30 = -2.99610707703542174e-3, q30 = 1.06209230528467918e-2, p31 = -4.94730910623250734e-2,
            q31 = 1.91308926107829841e-1, p32 = -2.26956593539686930e-1, q32 = 1.05167510706793207e+0,
            p33 = -2.78661308609647788e-1, q33 = 1.98733201817135256e+0, p34 = -2.23192459734184686e-2, q34 = 1;

   ::ROOT::Double_v v = vecCore::math::Abs(x) / w2;

   ::ROOT::Double_v ap, aq, h, hc, y, result;

   vecCore::Mask<::ROOT::Double_v> mask1 = v < ::ROOT::Double_v(0.5);
   vecCore::Mask<::ROOT::Double_v> mask2 = !mask1 && v < ::ROOT::Double_v(4.0);
   vecCore::Mask<::ROOT::Double_v> mask3 = !(mask1 || mask2);

   ::ROOT::Double_v v2 = v * v;
   ::ROOT::Double_v v3 = v2 * v;
   ::ROOT::Double_v v4 = v3 * v;
   ::ROOT::Double_v v5 = v4 * v;
   ::ROOT::Double_v v6 = v5 * v;
   ::ROOT::Double_v v7 = v6 * v;
   ::ROOT::Double_v v8 = v7 * v;

   vecCore::MaskedAssign<::ROOT::Double_v>(
      result, mask1, v * (p10 + p11 * v2 + p12 * v4 + p13 * v6) / (q10 + q11 * v2 + q12 * v4 + v6));
   vecCore::MaskedAssign<::ROOT::Double_v>(
      result, mask2,
      ::ROOT::Double_v(1.0) -
         (p20 + p21 * v + p22 * v2 + p23 * v3 + p24 * v4 + p25 * v5 + p26 * v6 + p27 * v7) /
            (vecCore::math::Exp(v2) * (q20 + q21 * v + q22 * v2 + q23 * v3 + q24 * v4 + q25 * v5 + q26 * v6 + v7)));
   vecCore::MaskedAssign<::ROOT::Double_v>(result, mask3,
                                           ::ROOT::Double_v(1.0) -
                                              (c1 + (p30 * v8 + p31 * v6 + p32 * v4 + p33 * v2 + p34) /
                                                       ((q30 * v8 + q31 * v6 + q32 * v4 + q33 * v2 + q34) * v2)) /
                                                 (v * vecCore::math::Exp(v2)));

   return vecCore::Blend<::ROOT::Double_v>(x > 0, ::ROOT::Double_v(0.5) + ::ROOT::Double_v(0.5) * result,
                                           ::ROOT::Double_v(0.5) * (::ROOT::Double_v(1) - result));
}

////////////////////////////////////////////////////////////////////////////////
/// Vectorized implementation of Bessel function I_0(x) for a vector x.
::ROOT::Double_v BesselI0_Split_More(::ROOT::Double_v &ax)
{
   ::ROOT::Double_v y = 3.75 / ax;
   return (vecCore::math::Exp(ax) / vecCore::math::Sqrt(ax)) *
          (0.39894228 +
           y * (1.328592e-2 +
                y * (2.25319e-3 +
                     y * (-1.57565e-3 +
                          y * (9.16281e-3 +
                               y * (-2.057706e-2 + y * (2.635537e-2 + y * (-1.647633e-2 + y * 3.92377e-3))))))));
}

::ROOT::Double_v BesselI0_Split_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x * x * 0.071111111;

   return 1.0 +
          y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (3.60768e-2 + y * 4.5813e-3)))));
}

::ROOT::Double_v BesselI0(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);

   return vecCore::Blend<::ROOT::Double_v>(ax <= 3.75, BesselI0_Split_Less(x), BesselI0_Split_More(ax));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of modified Bessel function I_1(x) for a vector x.
::ROOT::Double_v BesselI1_Split_More(::ROOT::Double_v &ax, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = 3.75 / ax;
   ::ROOT::Double_v result =
      (vecCore::math::Exp(ax) / vecCore::math::Sqrt(ax)) *
      (0.39894228 + y * (-3.988024e-2 +
                         y * (-3.62018e-3 +
                              y * (1.63801e-3 + y * (-1.031555e-2 +
                                                     y * (2.282967e-2 + y * (-2.895312e-2 +
                                                                             y * (1.787654e-2 + y * -4.20059e-3))))))));
   return vecCore::Blend<::ROOT::Double_v>(x < 0, ::ROOT::Double_v(-1.0) * result, result);
}

::ROOT::Double_v BesselI1_Split_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x * x * 0.071111111;

   return x * (0.5 + y * (0.87890594 +
                          y * (0.51498869 + y * (0.15084934 + y * (2.658733e-2 + y * (3.01532e-3 + y * 3.2411e-4))))));
}

::ROOT::Double_v BesselI1(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);

   return vecCore::Blend<::ROOT::Double_v>(ax <= 3.75, BesselI1_Split_Less(x), BesselI1_Split_More(ax, x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function J0(x) for a vector x.
::ROOT::Double_v BesselJ0_Split1_More(::ROOT::Double_v &ax)
{
   ::ROOT::Double_v z = 8 / ax;
   ::ROOT::Double_v y = z * z;
   ::ROOT::Double_v xx = ax - 0.785398164;
   ::ROOT::Double_v result1 =
      1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
   ::ROOT::Double_v result2 =
      -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
   return vecCore::math::Sqrt(0.636619772 / ax) *
          (vecCore::math::Cos(xx) * result1 - z * vecCore::math::Sin(xx) * result2);
}

::ROOT::Double_v BesselJ0_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x * x;
   return (57568490574.0 +
           y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))) /
          (57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))));
}

::ROOT::Double_v BesselJ0(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);
   return vecCore::Blend<::ROOT::Double_v>(ax < 8, BesselJ0_Split1_Less(x), BesselJ0_Split1_More(ax));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function J1(x) for a vector x.
::ROOT::Double_v BesselJ1_Split1_More(::ROOT::Double_v &ax, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v z = 8 / ax;
   ::ROOT::Double_v y = z * z;
   ::ROOT::Double_v xx = ax - 2.356194491;
   ::ROOT::Double_v result1 =
      1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
   ::ROOT::Double_v result2 =
      0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 - y * 0.105787412e-6)));
   ::ROOT::Double_v result =
      vecCore::math::Sqrt(0.636619772 / ax) * (vecCore::math::Cos(xx) * result1 - z * vecCore::math::Sin(xx) * result2);
   vecCore::MaskedAssign<::ROOT::Double_v>(result, x < 0, -result);
   return result;
}

::ROOT::Double_v BesselJ1_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x * x;
   return x *
          (72362614232.0 +
           y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))))) /
          (144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))));
}

::ROOT::Double_v BesselJ1(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);
   return vecCore::Blend<::ROOT::Double_v>(ax < 8, BesselJ1_Split1_Less(x), BesselJ1_Split1_More(ax, x));
}

} // namespace TMath

#endif
