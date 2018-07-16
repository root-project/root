#ifndef ROOT_VectorizedTMath
#define ROOT_VectorizedTMath

#include "Rtypes.h"
#include "Math/Types.h"
#include "TMath.h"

#if defined(R__HAS_VECCORE) && defined(R__HAS_VC)

namespace TMath {
////////////////////////////////////////////////////////////////////////////////
::ROOT::Double_v Log2(::ROOT::Double_v &x)
{
   return vecCore::math::Log2(x);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate a Breit Wigner function with mean and gamma.
::ROOT::Double_v BreitWigner(::ROOT::Double_v &x, Double_t mean=0, Double_t gamma=1)
{
   return (gamma / (gamma*gamma / 4.0f + (x-mean) * (x-mean)))/(2*Pi());
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate a gaussian function with mean and sigma.
/// If norm=kTRUE (default is kFALSE) the result is divided
/// by sqrt(2*Pi)*sigma.
::ROOT::Double_v Gaus(::ROOT::Double_v &x, Double_t mean=0, Double_t sigma=1, Bool_t norm=kFALSE)
{
   if (sigma == 0)
      return ::ROOT::Double_v(1.e30f);

   ::ROOT::Double_v arg = (x-mean)/sigma;

   // For those entries of |arg| > 39 result is zero in double precision
   ::ROOT::Double_v out = vecCore::Blend<::ROOT::Double_v>(vecCore::math::Abs(arg) < 39.0f,
                                                       vecCore::math::Exp(-0.5f * arg * arg),
                                                       0.0f);
   if (!norm)
      return out;
   return out/(2.50662827463100024f * sigma); //sqrt(2*Pi)=2.50662827463100024
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the probability density function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
::ROOT::Double_v LaplaceDist(::ROOT::Double_v &x, Double_t alpha=0, Double_t beta=1)
{
   ::ROOT::Double_v out = vecCore::math::Exp(-vecCore::math::Abs((x-alpha)/beta));
   out /= (::ROOT::Double_v(2.0f)*beta);
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the distribution function of Laplace distribution
/// at point x, with location parameter alpha and shape parameter beta.
/// By default, alpha=0, beta=1
/// This distribution is known under different names, most common is
/// double exponential distribution, but it also appears as
/// the two-tailed exponential or the bilateral exponential distribution
::ROOT::Double_v LaplaceDistI(::ROOT::Double_v &x, Double_t alpha=0, Double_t beta=1)
{
   return vecCore::Blend<::ROOT::Double_v>(x <= alpha,
                                         0.5 * vecCore::math::Exp(-vecCore::math::Abs((x-alpha)/beta)),
                                         1 - 0.5 * vecCore::math::Exp(-vecCore::math::Abs((x-alpha)/beta)));
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

    Double_t p10 = 2.4266795523053175e+2,  q10 = 2.1505887586986120e+2,
                  p11 = 2.1979261618294152e+1,  q11 = 9.1164905404514901e+1,
                  p12 = 6.9963834886191355e+0,  q12 = 1.5082797630407787e+1,
                  p13 =-3.5609843701815385e-2;

    Double_t p20 = 3.00459261020161601e+2, q20 = 3.00459260956983293e+2,
                  p21 = 4.51918953711872942e+2, q21 = 7.90950925327898027e+2,
                  p22 = 3.39320816734343687e+2, q22 = 9.31354094850609621e+2,
                  p23 = 1.52989285046940404e+2, q23 = 6.38980264465631167e+2,
                  p24 = 4.31622272220567353e+1, q24 = 2.77585444743987643e+2,
                  p25 = 7.21175825088309366e+0, q25 = 7.70001529352294730e+1,
                  p26 = 5.64195517478973971e-1, q26 = 1.27827273196294235e+1,
                  p27 =-1.36864857382716707e-7;

    Double_t p30 =-2.99610707703542174e-3, q30 = 1.06209230528467918e-2,
                  p31 =-4.94730910623250734e-2, q31 = 1.91308926107829841e-1,
                  p32 =-2.26956593539686930e-1, q32 = 1.05167510706793207e+0,
                  p33 =-2.78661308609647788e-1, q33 = 1.98733201817135256e+0,
                  p34 =-2.23192459734184686e-2, q34 = 1;

   ::ROOT::Double_v v = vecCore::math::Abs(x)/::ROOT::Double_v(w2);

   ::ROOT::Double_v ap, aq, h, hc, y, result;

   vecCore::Mask<::ROOT::Double_v> mask1 = v < ::ROOT::Double_v(0.5f);
   vecCore::Mask<::ROOT::Double_v> mask2 = !mask1 && v < ::ROOT::Double_v(4.0f);
   vecCore::Mask<::ROOT::Double_v> mask3 = !(mask1 || mask2);

   ::ROOT::Double_v v2 = v*v;
   ::ROOT::Double_v v3 = v2*v;
   ::ROOT::Double_v v4 = v3*v;
   ::ROOT::Double_v v5 = v4*v;
   ::ROOT::Double_v v6 = v5*v;
   ::ROOT::Double_v v7 = v6*v;
   ::ROOT::Double_v v8 = v7*v;

   vecCore::MaskedAssign<::ROOT::Double_v>(result, mask1,
      v*(p10 + p11*v2 + p12*v4 + p13*v6)/(q10 + q11*v2 + q12*v4 + v6));
   vecCore::MaskedAssign<::ROOT::Double_v>(result, mask2,
      ::ROOT::Double_v(1.0f) - (p20 + p21*v + p22*v2 + p23*v3 + p24*v4 + p25*v5 + p26*v6 + p27*v7)/
                             (vecCore::math::Exp(v2) * (q20 + q21*v + q22*v2 + q23*v3 + q24*v4 + q25*v5 + q26*v6 + v7)));
   vecCore::MaskedAssign<::ROOT::Double_v>(result, mask3,
      ::ROOT::Double_v(1.0f) - (c1 + (p30*v8 + p31*v6 + p32*v4 + p33*v2 + p34)/
                             ((q30*v8 + q31*v6 + q32*v4 + q33*v2 + q34)*v2))/(v * vecCore::math::Exp(v2)));

   return vecCore::Blend<::ROOT::Double_v>(x > 0,
                                         ::ROOT::Double_v(0.5) + ::ROOT::Double_v(0.5) * result,
                                         ::ROOT::Double_v(0.5)*(::ROOT::Double_v(1) - result));
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the Kolmogorov distribution function,
///
/// \f[
/// P(z) = 2 \sum_{j=1}^{\infty} (-1)^{j-1} e^{-2 j^2 z^2}
/// \f]
///
/// which gives the probability that Kolmogorov's test statistic will exceed
/// the value z assuming the null hypothesis. This gives a very powerful
/// test for comparing two one-dimensional distributions.
/// see, for example, Eadie et al, "statistical Methods in Experimental
/// Physics', pp 269-270).
///
/// This function returns the confidence level for the null hypothesis, where:
///  - \f$ z = dn \sqrt{n} \f$, and
///     - \f$ dn \f$  is the maximum deviation between a hypothetical distribution
///           function and an experimental distribution with
///     - \f$ n \f$  events
///
/// NOTE: To compare two experimental distributions with m and n events,
///       use \f$ z = \sqrt{m n/(m+n)) dn} \f$
///
/// Accuracy: The function is far too accurate for any imaginable application.
///           Probabilities less than \f$ 10^{-15} \f$ are returned as zero.
///           However, remember that the formula is only valid for "large" n.
///
/// Theta function inversion formula is used for z <= 1
///
/// This function was translated by Rene Brun from PROBKL in CERNLIB.
::ROOT::Double_v KolmogorovProb_Split3_Less(::ROOT::Double_v &u)
{
   ::ROOT::Double_v uu = u * u;

   ::ROOT::Double_v maxj = vecCore::math::Max(::ROOT::Double_v(1.0), vecCore::math::Round(3./u));
   ::ROOT::Double_v r0 = vecCore::math::Exp(-2. * uu);
   ::ROOT::Double_v r1, r2, r3, result;

   vecCore::MaskedAssign<::ROOT::Double_v>(r1, maxj >= ::ROOT::Double_v(1), vecCore::math::Exp(-8. * uu));
   vecCore::MaskedAssign<::ROOT::Double_v>(r2, maxj >= ::ROOT::Double_v(2), vecCore::math::Exp(-18. * uu));
   vecCore::MaskedAssign<::ROOT::Double_v>(r3, maxj >= ::ROOT::Double_v(3), vecCore::math::Exp(-32. * uu));

   return 2.*(r0 - r1 + r2 - r3);
}

::ROOT::Double_v KolmogorovProb_Split2_More(::ROOT::Double_v &u)
{
   return vecCore::Blend<::ROOT::Double_v>(u < 6.8116, KolmogorovProb_Split3_Less(u), 0);
}

::ROOT::Double_v KolmogorovProb_Split2_Less(::ROOT::Double_v &u)
{
   ::ROOT::Double_v u_inv = 1./u;
   ::ROOT::Double_v v = u_inv * u_inv;
   return ::ROOT::Double_v(1.0) - 2.50662827*u_inv*(vecCore::math::Exp(-1.2337005501361697*v) +
                  vecCore::math::Exp(-11.103304951225528*v) + vecCore::math::Exp(-30.842513753404244*v));
}

::ROOT::Double_v KolmogorovProb_Split1_More(::ROOT::Double_v &u)
{
   return vecCore::Blend<::ROOT::Double_v>(u < 0.755,
                        KolmogorovProb_Split2_Less(u), KolmogorovProb_Split2_More(u));
}

::ROOT::Double_v KolmogorovProb(::ROOT::Double_v &z)
{
   ::ROOT::Double_v u = vecCore::math::Abs(z);
   return vecCore::Blend<::ROOT::Double_v>(u < 0.2, 1, KolmogorovProb_Split1_More(u));
}

////////////////////////////////////////////////////////////////////////////////
/// The DiLogarithm function
inline ::ROOT::Double_v DiLog_Iterations(::ROOT::Double_v &y,
                                    ::ROOT::Double_v &s, ::ROOT::Double_v &a)
{
    Double_t c[20] = {0.42996693560813697, 0.40975987533077105,
   -0.01858843665014592, 0.00145751084062268,-0.00014304184442340,
   0.00001588415541880,-0.00000190784959387, 0.00000024195180854,
   -0.00000003193341274, 0.00000000434545063,-0.00000000060578480,
   0.00000000008612098,-0.00000000001244332, 0.00000000000182256,
   -0.00000000000027007, 0.00000000000004042,-0.00000000000000610,
   0.00000000000000093,-0.00000000000000014, 0.00000000000000002};
   
   ::ROOT::Double_v h = y+y-1;
   ::ROOT::Double_v alfa = h+h;
   ::ROOT::Double_v b0 = ::ROOT::Double_v(0);
   ::ROOT::Double_v b1 = ::ROOT::Double_v(0);
   ::ROOT::Double_v b2 = ::ROOT::Double_v(0);
   for (Int_t i=19;i>=0;i--){
      b0 = c[i] + alfa*b1-b2;
      b2 = b1;
      b1 = b0;
   }
   return -(s*(b0-h*b2)+a);
}

inline ::ROOT::Double_v DiLog_Split7_Less(::ROOT::Double_v &t)
{
   ::ROOT::Double_v ones = ::ROOT::Double_v(1);
   ::ROOT::Double_v zeros = ::ROOT::Double_v(0);
   return DiLog_Iterations(t, ones, zeros);
}

inline ::ROOT::Double_v DiLog_Split7_More(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   ::ROOT::Double_v y = 1/t;
   ::ROOT::Double_v s = ::ROOT::Double_v(-1);
   ::ROOT::Double_v b1= vecCore::math::Log(t);
   ::ROOT::Double_v a = pi6+0.5f*b1*b1;

   return DiLog_Iterations(y, s, a);
}

inline ::ROOT::Double_v DiLog_Split6_More(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   return vecCore::Blend<::ROOT::Double_v>(t <= 1,
                        DiLog_Split7_Less(t), DiLog_Split7_More(t, pi6));
}

inline ::ROOT::Double_v DiLog_Split6_Less(::ROOT::Double_v &t)
{
   ::ROOT::Double_v y = -t/(1+t);
   ::ROOT::Double_v s = ::ROOT::Double_v(-1);
   ::ROOT::Double_v b1= vecCore::math::Log(1+t);
   ::ROOT::Double_v a = 0.5*b1*b1;

   return DiLog_Iterations(y, s, a);
}

inline ::ROOT::Double_v DiLog_Split5_More(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   return vecCore::Blend<::ROOT::Double_v>(t < 0,
                        DiLog_Split6_Less(t), DiLog_Split6_More(t, pi6));
}

inline ::ROOT::Double_v DiLog_Split5_Less(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   ::ROOT::Double_v y = -(1+t)/t;
   ::ROOT::Double_v s = ::ROOT::Double_v(1);
   ::ROOT::Double_v a = vecCore::math::Log(-t);
   a = -pi6+a*(-0.5f*a+vecCore::math::Log(1+t));

   return DiLog_Iterations(y, s, a);
}

inline ::ROOT::Double_v DiLog_Split4_More(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   return vecCore::Blend<::ROOT::Double_v>(t <= -0.5,
                        DiLog_Split5_Less(t, pi6), DiLog_Split5_More(t, pi6));
}

inline ::ROOT::Double_v DiLog_Split4_Less(::ROOT::Double_v &t, ::ROOT::Double_v &pi6)
{
   ::ROOT::Double_v y = -t-1;
   ::ROOT::Double_v s = ::ROOT::Double_v(-1);
   ::ROOT::Double_v a = vecCore::math::Log(-t);
   a = -pi6 + a*(a+vecCore::math::Log(1+1/t));

   return DiLog_Iterations(y, s, a);
}

inline ::ROOT::Double_v DiLog_Split3_More(::ROOT::Double_v &t, ::ROOT::Double_v &pi3)
{
   ::ROOT::Double_v pi6 = ::ROOT::Double_v(pi3/2);
   return vecCore::Blend<::ROOT::Double_v>(t < -1,
                        DiLog_Split4_Less(t, pi6), DiLog_Split4_More(t, pi6));
}

inline ::ROOT::Double_v DiLog_Split3_Less(::ROOT::Double_v &t, ::ROOT::Double_v &pi3)
{
   ::ROOT::Double_v y = -1/(1 + t);
   ::ROOT::Double_v s = ::ROOT::Double_v(1);
   ::ROOT::Double_v b1 = vecCore::math::Log(-t);
   ::ROOT::Double_v b2 = vecCore::math::Log(1+1/t);
   ::ROOT::Double_v a = -pi3 + 0.5f*(b1*b1 - b2*b2);

   return DiLog_Iterations(y, s, a);
}

inline ::ROOT::Double_v DiLog_Split2(::ROOT::Double_v &x, ::ROOT::Double_v &pi2)
{
   ::ROOT::Double_v t = -x;
   ::ROOT::Double_v pi3 = pi2/3;
   return vecCore::Blend<::ROOT::Double_v>(t <= -2,
                        DiLog_Split3_Less(t, pi3), DiLog_Split3_More(t, pi2));
}

inline ::ROOT::Double_v DiLog_Split1(::ROOT::Double_v &x, ::ROOT::Double_v &pi2)
{
   ::ROOT::Double_v pi12 = ::ROOT::Double_v(pi2/12);
   return vecCore::Blend<::ROOT::Double_v>(x == -1, -pi12, DiLog_Split2(x, pi2));
}

inline ::ROOT::Double_v DiLog(::ROOT::Double_v &x)
{
    Double_t pi  = TMath::Pi();
    ::ROOT::Double_v pi2 = ::ROOT::Double_v(pi*pi);

   return vecCore::Blend<::ROOT::Double_v>(x == 1, pi2/6, DiLog_Split1(x, pi2));
}

////////////////////////////////////////////////////////////////////////////////
/// Vectorized implementation of Bessel function I_0(x) for a vector x.
inline ::ROOT::Double_v BesselI0_Split_More(::ROOT::Double_v &ax)
{
      ::ROOT::Double_v y = 3.75/ax;
      return (vecCore::math::Exp(ax)/vecCore::math::Sqrt(ax))*
            (0.39894228+y*(1.328592e-2+y*(2.25319e-3+
            y*(-1.57565e-3+y*(9.16281e-3+y*(-2.057706e-2+
            y*(2.635537e-2+y*(-1.647633e-2+y*3.92377e-3))))))));
}

inline ::ROOT::Double_v BesselI0_Split_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v xx = x/3.75;
   ::ROOT::Double_v y = xx * xx;

   return 1.0+y*(3.5156229+y*(3.0899424+
      y*(1.2067492+y*(0.2659732+
      y*(3.60768e-2+y*4.5813e-3)))));
}

inline ::ROOT::Double_v BesselI0(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);

   return vecCore::Blend<::ROOT::Double_v>(ax <= 3.75, BesselI0_Split_Less(x), BesselI0_Split_More(ax));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of modified Bessel function K_0(x) for a vector x.
inline ::ROOT::Double_v BesselK0_Split2_More(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = 2/x;
   return (vecCore::math::Exp(-x)/vecCore::math::Sqrt(x))*
            (1.25331414+y*(-7.832358e-2+y*(2.189568e-2+
            y*(-1.062446e-2+y*(5.87872e-3+y*(-2.51540e-3+y*5.3208e-4))))));
}

inline ::ROOT::Double_v BesselK0_Split2_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x*x/4;
   return (-vecCore::math::Log(x/2.)*BesselI0(x))+(-0.57721566+y*(0.42278420+
                        y*(0.23069756+y*(3.488590e-2+
                        y*(2.62698e-3+y*(1.0750e-4+y*7.4e-6))))));
}

inline ::ROOT::Double_v BesselK0_Split1_More(::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(x <= 2, BesselK0_Split2_Less(x), BesselK0_Split2_More(x));
}

inline ::ROOT::Double_v BesselK0(::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(x <= 0, ::ROOT::Double_v(0), BesselK0_Split1_More(x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of modified Bessel function I_1(x) for a vector x.
inline ::ROOT::Double_v BesselI1_Split_More(::ROOT::Double_v &ax, ::ROOT::Double_v &x)
{
      ::ROOT::Double_v y = 3.75/ax;
      ::ROOT::Double_v result = (vecCore::math::Exp(ax)/vecCore::math::Sqrt(ax))*
            (0.39894228+y*(-3.988024e-2+y*(-3.62018e-3+
            y*(1.63801e-3+y*(-1.031555e-2+y*(2.282967e-2+
            y*(-2.895312e-2+y*(1.787654e-2+y*-4.20059e-3))))))));
      return vecCore::Blend<::ROOT::Double_v>(x < 0, -result, result);
}

inline ::ROOT::Double_v BesselI1_Split_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v xx = x/3.75;
   ::ROOT::Double_v y = xx * xx;

   return x*(0.5+y*(0.87890594+y*(0.51498869+
      y*(0.15084934+y*(2.658733e-2+y*(3.01532e-3+y*3.2411e-4))))));
}

inline ::ROOT::Double_v BesselI1(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);

   return vecCore::Blend<::ROOT::Double_v>(ax <= 3.75,
            BesselI1_Split_Less(x), BesselI1_Split_More(ax, x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of modified Bessel function K_1(x) for a vector x.
inline ::ROOT::Double_v BesselK1_Split2_More(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = 2/x;
   return (vecCore::math::Exp(-x)/vecCore::math::Sqrt(x))*
            (1.25331414+y*(0.23498619+y*(-3.655620e-2+
            y*(1.504268e-2+y*(-7.80353e-3+y*(3.25614e-3+y*-6.8245e-4))))));
}

inline ::ROOT::Double_v BesselK1_Split2_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x*x/4;
   return (-vecCore::math::Log(x/2.0f)*BesselI1(x))+(1.0+y*(0.15443144+
                        y*(-0.67278579+y*(-0.18156897+
                        y*(-1.919402e-2+y*(-1.10404e-3+y*-4.686e-5))))));
}

inline ::ROOT::Double_v BesselK1_Split1_More(::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(x <= 2, BesselK1_Split2_Less(x), BesselK1_Split2_More(x));
}

inline ::ROOT::Double_v BesselK1(::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(x <= 0, ::ROOT::Double_v(0), BesselK1_Split1_More(x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Integer Order modified Bessel function K_n(x) for a vector x.

inline ::ROOT::Double_v BesselK_Split1_More(Int_t n, ::ROOT::Double_v &x)
{
   if (n==0) return BesselK0(x);
   if (n==1) return BesselK1(x);

   // Perform upward recurrence for all x
   ::ROOT::Double_v tox = 2/x;
   ::ROOT::Double_v bkm = TMath::BesselK0(x);
   ::ROOT::Double_v bk  = TMath::BesselK1(x);
   ::ROOT::Double_v bkp = ::ROOT::Double_v(0);
   for (Int_t j=1; j<n; j++) {
      bkp = bkm+::ROOT::Double_v(j)*tox*bk;
      bkm = bk;
      bk  = bkp;
   }
   return bk;
}

inline ::ROOT::Double_v BesselK(Int_t n, ::ROOT::Double_v &x)
{
   if(n < 0)
      return ::ROOT::Double_v(0);
   return vecCore::Blend<::ROOT::Double_v>(x <= 0, ::ROOT::Double_v(0), BesselK_Split1_More(n, x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Integer Order modified Bessel function I_n(x) for a vector x.
inline ::ROOT::Double_v BesselI_Split2_Less(Int_t n, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v tox = 2/vecCore::math::Abs(x);
   ::ROOT::Double_v bip = ::ROOT::Double_v(0);
   ::ROOT::Double_v bim = ::ROOT::Double_v(0);
   ::ROOT::Double_v bi  = ::ROOT::Double_v(1);
   ::ROOT::Double_v result = ::ROOT::Double_v(0);

   Int_t iacc = 40; // Increase to enhance accuracy
    ::ROOT::Double_v kBigPositive = ::ROOT::Double_v(1.e10);
    ::ROOT::Double_v kBigNegative = ::ROOT::Double_v(1.e-10);

   Int_t m = 2*((n+Int_t(sqrt(Float_t(iacc*n)))));
   for (Int_t j=m; j>=1; j--) {
      bim = bip+::ROOT::Double_v(j)*tox*bi;
      bip = bi;
      bi  = bim;
      // Renormalise to prevent overflows
      vecCore::Mask<::ROOT::Double_v> mask = vecCore::math::Abs(bi) > kBigPositive;
      vecCore::MaskedAssign<::ROOT::Double_v>(result, mask, result * kBigNegative);
      vecCore::MaskedAssign<::ROOT::Double_v>(bi, mask, bi * kBigNegative);
      vecCore::MaskedAssign<::ROOT::Double_v>(bip, mask, bip * kBigNegative);

      if (j==n) result=bip;
   }

   result *= BesselI0(x)/bi; // Normalise with BesselI0(x)
   if(n%2 == 1)
   {
      vecCore::Mask<::ROOT::Double_v> mask_negative = x < 0;
      vecCore::MaskedAssign<::ROOT::Double_v>(result, mask_negative, -result);
   }
   return result;
}

inline ::ROOT::Double_v BesselI_Split1_More(Int_t n, ::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(vecCore::math::Abs(x) > ::ROOT::Double_v(1.e10),
                              ::ROOT::Double_v(0), BesselI_Split2_Less(n, x));
}

::ROOT::Double_v BesselI(Int_t n, ::ROOT::Double_v &x)
{
   if (n < 0) return ::ROOT::Double_v(0);

   if (n==0) return BesselI0(x);
   if (n==1) return BesselI1(x);

   return vecCore::Blend<::ROOT::Double_v>(x == 0, ::ROOT::Double_v(0), BesselI_Split1_More(n, x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function J0(x) for a vector x.
::ROOT::Double_v BesselJ0_Split1_More(::ROOT::Double_v &ax)
{
   ::ROOT::Double_v z  = 8/ax;
   ::ROOT::Double_v y  = z*z;
   ::ROOT::Double_v xx = ax-0.785398164;
   ::ROOT::Double_v result1 = 1  + y*(-0.1098628627e-2 + y*(0.2734510407e-4 +
                              y*(-0.2073370639e-5 + y*0.2093887211e-6)));
   ::ROOT::Double_v result2 = -0.1562499995e-1 + y*(0.1430488765e-3 +
                  y*(-0.6911147651e-5 + y*(0.7621095161e-6 - y*0.934935152e-7)));
   return vecCore::math::Sqrt(0.636619772/ax)*(vecCore::math::Cos(xx)*result1-z*vecCore::math::Sin(xx)*result2);
}

::ROOT::Double_v BesselJ0_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y=x*x;
   return (57568490574.0 + y*(-13362590354.0 + y*(651619640.7 + y*(-11214424.18  +
            y*(77392.33017  + y*-184.9052456))))) /
            (57568490411.0 + y*(1029532985.0 + y*(9494680.718 +
            y*(59272.64853 + y*(267.8532712 + y)))));
}

::ROOT::Double_v BesselJ0(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);
   return vecCore::Blend<::ROOT::Double_v>(ax < 8, BesselJ0_Split1_Less(x),
                  BesselJ0_Split1_More(ax));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function J1(x) for a vector x.
::ROOT::Double_v BesselJ1_Split1_More(::ROOT::Double_v &ax, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v z  = 8/ax;
   ::ROOT::Double_v y  = z*z;
   ::ROOT::Double_v xx = ax-2.356194491;
   ::ROOT::Double_v result1 = 1  + y*(0.183105e-2 + y*(-0.3516396496e-4 +
                              y*(0.2457520174e-5 + y*-0.240337019e-6)));
   ::ROOT::Double_v result2 = 0.04687499995 + y*(-0.2002690873e-3 +
                  y*(0.8449199096e-5 + y*(-0.88228987e-6 - y*0.105787412e-6)));
   ::ROOT::Double_v result = vecCore::math::Sqrt(0.636619772/ax)*(vecCore::math::Cos(xx)*result1-z*vecCore::math::Sin(xx)*result2);
   vecCore::MaskedAssign<::ROOT::Double_v>(result, x < 0, -result);
   return result;
}

::ROOT::Double_v BesselJ1_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y=x*x;
   return x*(72362614232.0 + y*(-7895059235.0 + y*(242396853.1 + y*(-2972611.439 +
            y*(15704.48260  + y*-30.16036606))))) /
            (144725228442.0 + y*(2300535178.0 + y*(18583304.74 +
            y*(99447.43394 + y*(376.9991397 + y)))));
}

::ROOT::Double_v BesselJ1(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);
   return vecCore::Blend<::ROOT::Double_v>(ax < 8, BesselJ1_Split1_Less(x),
                  BesselJ1_Split1_More(ax, x));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function Y0(x) for a vector x.
::ROOT::Double_v BesselY0_Split1_More(::ROOT::Double_v &x)
{
   ::ROOT::Double_v z  = 8/x;
   ::ROOT::Double_v y  = z*z;
   ::ROOT::Double_v xx = x-0.785398164;
   ::ROOT::Double_v result1 = 1  + y*(-0.1098628627e-2 + y*(0.2734510407e-4 +
                        y*(-0.2073370639e-5 + y*0.2093887211e-6)));
   ::ROOT::Double_v result2 = -0.1562499995e-1 + y*(0.1430488765e-3 +
                        y*(-0.6911147651e-5 + y*(0.7621095161e-6 + y*-0.934945152e-7)));
   ::ROOT::Double_v result  = vecCore::math::Sqrt(0.636619772/x)*(vecCore::math::Sin(xx)*result1+z*vecCore::math::Cos(xx)*result2);
   return result;
}

::ROOT::Double_v BesselY0_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x*x;
   ::ROOT::Double_v result1 = -2957821389.0 + y*(7062834065.0 +
                  y*(-512359803.6 + y*(10879881.29  + y*(-86327.92757  + y*228.4622733))));
   ::ROOT::Double_v result2 = 40076544269.0 + y*(745249964.8 +
                  y*(7189466.438 + y*(47447.26470 + y*(226.1030244 + y))));
   return (result1/result2) + 0.636619772*BesselJ0(x)*vecCore::math::Log(x);
}

::ROOT::Double_v BesselY0(::ROOT::Double_v &x)
{
   ::ROOT::Double_v ax = vecCore::math::Abs(x);
   return vecCore::Blend<::ROOT::Double_v>(x < 8, BesselY0_Split1_Less(x),
                  BesselY0_Split1_More(ax));
}

////////////////////////////////////////////////////////////////////////////////
///  Vectorized implementation of Bessel function Y1(x) for a vector x.
inline ::ROOT::Double_v BesselY1_Split1_More(::ROOT::Double_v &x)
{
   ::ROOT::Double_v z  = 8/x;
   ::ROOT::Double_v y  = z*z;
   ::ROOT::Double_v xx = x-2.356194491;
   ::ROOT::Double_v result1 = 1  + y*(0.183105e-2 + y*(-0.3516396496e-4 +
                              y*(0.2457520174e-5 + y*-0.240337019e-6)));
   ::ROOT::Double_v result2 = 0.04687499995 + y*(-0.2002690873e-3 +
                              y*(0.8449199096e-5 + y*(-0.88228987e-6 + y*0.105787412e-6)));
   return vecCore::math::Sqrt(0.636619772/x)*(vecCore::math::Sin(xx)*result1+z*vecCore::math::Cos(xx)*result2);
}

inline ::ROOT::Double_v BesselY1_Split1_Less(::ROOT::Double_v &x)
{
   ::ROOT::Double_v y = x*x;
   ::ROOT::Double_v result1 = x*(-0.4900604943e13 + y*(0.1275274390e13 + y*(-0.5153438139e11 +
            y*(0.7349264551e9 + y*(-0.4237922726e7 + y*0.8511937935e4)))));
   ::ROOT::Double_v result2 = 0.2499580570e14 + y*(0.4244419664e12 + y*(0.3733650367e10 +
            y*(0.2245904002e8 + y*(0.1020426050e6  + y*(0.3549632885e3+y)))));
   return (result1/result2) + 0.636619772*(BesselJ1(x)*vecCore::math::Log(x)-1/x);
}

inline ::ROOT::Double_v BesselY1(::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(x < 8, BesselY1_Split1_Less(x),
                  BesselY1_Split1_More(x));
}

inline ::ROOT::Double_v Gamma(::ROOT::Double_v &x)
{
   return vecCore::math::TGamma(x);
}

inline ::ROOT::Double_v LnGamma(::ROOT::Double_v &x)
{
   return vecCore::math::Log(vecCore::math::TGamma(x));
}

////////////////////////////////////////////////////////////////////////////////
/// Vectorized implementation of the incomplete gamma function P(a,x)
/// via its continued fraction representation.
inline ::ROOT::Double_v GamCf_iterations(::ROOT::Double_v &a, ::ROOT::Double_v &b,
                                         ::ROOT::Double_v &c, ::ROOT::Double_v &d,
                                         ::ROOT::Double_v &h, ::ROOT::Double_v &x,
                                         Int_t iter_start)
{
   ::ROOT::Double_v eps = ::ROOT::Double_v(3.e-14);
   ::ROOT::Double_v fpmin = ::ROOT::Double_v(1.e-30);
   ::ROOT::Double_v gln = LnGamma(a);
   ::ROOT::Double_v an, del, v;
   if(iter_start <= 100) {
      an = ::ROOT::Double_v(-iter_start)*(::ROOT::Double_v(iter_start) - ::ROOT::Double_v(a));
      b = b + ::ROOT::Double_v(2.);

      d = an*d + b;
      vecCore::MaskedAssign<::ROOT::Double_v>(d, vecCore::math::Abs(d) < fpmin, fpmin);

      c = b + an/c;
      vecCore::MaskedAssign<::ROOT::Double_v>(c, vecCore::math::Abs(c) < fpmin, fpmin);

      d = 1./d;
      del = d * c;

      h = h * del;
      return vecCore::Blend<::ROOT::Double_v>(vecCore::math::Abs(del - 1.) < eps,
                                 vecCore::math::Exp(-x + a*vecCore::math::Log(x) - gln)*h ,
                                 GamCf_iterations(a, b, c, d, h, x, iter_start + 1));
   }
   return vecCore::math::Exp(-x + a*vecCore::math::Log(x) - gln)*h;
}

inline ::ROOT::Double_v GamCf_nonzero(::ROOT::Double_v &a, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v b = x + 1. - a;
   ::ROOT::Double_v c = 1.e30;
   ::ROOT::Double_v d = 1./b;
   ::ROOT::Double_v h = d;

   return 1 - GamCf_iterations(a, b, c, d, h, x, 1);

}

inline ::ROOT::Double_v GamCf(::ROOT::Double_v &a, ::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(a <= 0 || x <= 0,
                              0, GamCf_nonzero(a, x));
}

////////////////////////////////////////////////////////////////////////////////
/// Vectorized implementation of the incomplete gamma function P(a,x)
/// via its series representation.
inline ::ROOT::Double_v GamSer_iterations(::ROOT::Double_v &a, ::ROOT::Double_v &x,
                                          ::ROOT::Double_v &ap, ::ROOT::Double_v &del,
                                          ::ROOT::Double_v &sum, Int_t iter_start)
{
   ::ROOT::Double_v eps = ::ROOT::Double_v(3.e-14);
   ::ROOT::Double_v gln = LnGamma(a);

   if(iter_start <= 100) {
      ap += ::ROOT::Double_v(1.);
      del = del*x/ap;
      sum += del;

      return vecCore::Blend<::ROOT::Double_v>(vecCore::math::Abs(del) < vecCore::math::Abs(sum)*eps,
                                 sum*vecCore::math::Exp(-x + a*vecCore::math::Log(x) - gln),
                                 GamSer_iterations(a, x, ap, del, sum, iter_start + 1));
   }
   return sum*vecCore::math::Exp(-x + a*vecCore::math::Log(x) - gln);
}

inline ::ROOT::Double_v GamSer_nonzero(::ROOT::Double_v &a, ::ROOT::Double_v &x)
{
   ::ROOT::Double_v ap = a;
   ::ROOT::Double_v sum = 1./a;
   ::ROOT::Double_v del = sum;

   return GamSer_iterations(a, x, ap, del, sum, 1);
}

inline ::ROOT::Double_v GamSer(::ROOT::Double_v &a, ::ROOT::Double_v &x)
{
   return vecCore::Blend<::ROOT::Double_v>(a <= 0 || x <= 0,
                              0, GamSer_nonzero(a, x));
}

inline ::ROOT::Double_v Poisson_nonzero(::ROOT::Double_v &x, ::ROOT::Double_v &par)
{
   ::ROOT::Double_v x_plus_one = x + 1.;
   return vecCore::Blend<::ROOT::Double_v>(x == 0.0,
                              1./vecCore::math::Exp(par),
            vecCore::math::Exp(x*vecCore::math::Log(par) - par - LnGamma(x_plus_one)));
}

inline ::ROOT::Double_v Poisson(::ROOT::Double_v &x, ::ROOT::Double_v &par)
{
   return vecCore::Blend<::ROOT::Double_v>(x < 0,
                              0, Poisson_nonzero(x, par));
}

} //namespace TMath
#endif //VECCORE and VC exist check

#endif //file defined check
