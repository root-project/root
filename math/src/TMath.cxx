// @(#)root/base:$Name:  $:$Id: TMath.cxx,v 1.19 2002/02/18 18:09:30 brun Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath                                                                //
//                                                                      //
// Encapsulate math routines (i.e. provide a kind of namespace).        //
// For the time being avoid templates.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMath.h"
#include <math.h>
#include <string.h>

//const Double_t
//   TMath::Pi = 3.14159265358979323846,
//   TMath::E  = 2.7182818284590452354;


ClassImp(TMath)

//______________________________________________________________________________
#if defined(R__MAC) || defined(R__KCC)
Double_t hypot(Double_t x, Double_t y) {
   return sqrt(x*x+y*y);
}
#endif

//______________________________________________________________________________
Long_t TMath::Sqrt(Long_t x)
{
   return (Long_t) (sqrt((Double_t)x) + 0.5);
}

//______________________________________________________________________________
Long_t TMath::Hypot(Long_t x, Long_t y)
{
   return (Long_t) (hypot((Double_t)x, (Double_t)y) + 0.5);
}

//______________________________________________________________________________
Double_t TMath::Hypot(Double_t x, Double_t y)
{
   return hypot(x, y);
}

//______________________________________________________________________________
Double_t TMath::ASinH(Double_t x)
{
#if defined(WIN32) || defined(R__KCC)
   return log(x+sqrt(x*x+1));
#else
   return asinh(x);
#endif
}

//______________________________________________________________________________
Double_t TMath::ACosH(Double_t x)
{
#if defined(WIN32) || defined(R__KCC)
   return log(x+sqrt(x*x-1));
#else
   return acosh(x);
#endif
}

//______________________________________________________________________________
Double_t TMath::ATanH(Double_t x)
{
#if defined(WIN32) || defined(R__KCC)
   return log((1+x)/(1-x))/2;
#else
   return atanh(x);
#endif
}

//______________________________________________________________________________
Double_t TMath::Ceil(Double_t x)
{
   return ceil(x);
}

//______________________________________________________________________________
Double_t TMath::Floor(Double_t x)
{
   return floor(x);
}

//______________________________________________________________________________
Double_t TMath::Log2(Double_t x)
{
   return log(x)/log(2.0);
}

//______________________________________________________________________________
Long_t TMath::NextPrime(Long_t x)
{
   // Return next prime number after x, unless x is a prime in which case
   // x is returned.

   if (x < 2)
      return 2;
   if (x == 3)
      return 3;

   if (x % 2 == 0)
      x++;

   Long_t sqr = (Long_t) sqrt((Double_t)x) + 1;

   for (;;) {
      Long_t n;
      for (n = 3; (n <= sqr) && ((x % n) != 0); n += 2)
         ;
      if (n > sqr)
         return x;
      x += 2;
   }
}

//______________________________________________________________________________
Int_t TMath::Nint(Float_t x)
{
   // Round to nearest integer. Rounds half integers to the nearest
   // even integer.

   int i;
   if (x >= 0) {
      i = int(x + 0.5);
      if (x + 0.5 == Float_t(i) && i & 1) i--;
   } else {
      i = int(x - 0.5);
      if (x - 0.5 == Float_t(i) && i & 1) i++;

   }
   return i;
}

//______________________________________________________________________________
Int_t TMath::Nint(Double_t x)
{
   // Round to nearest integer. Rounds half integers to the nearest
   // even integer.

   int i;
   if (x >= 0) {
      i = int(x + 0.5);
      if (x + 0.5 == Double_t(i) && i & 1) i--;
   } else {
      i = int(x - 0.5);
      if (x - 0.5 == Double_t(i) && i & 1) i++;

   }
   return i;
}

//______________________________________________________________________________
Float_t *TMath::Cross(Float_t v1[3],Float_t v2[3],Float_t out[3])
{
   // Calculate the Cross Product of two vectors
   //         out = [v1 x v2]

    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];

    return out;
}

//______________________________________________________________________________
Double_t *TMath::Cross(Double_t v1[3],Double_t v2[3],Double_t out[3])
{
   // Calculate the Cross Product of two vectors
   //         out = [v1 x v2]

    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return out;
}

//______________________________________________________________________________
Double_t TMath::Erf(Double_t x)
{
   // Computation of the error function erf(x).
   // Erf(x) = (2/sqrt(pi)) Integral(exp(-t^2))dt between 0 and x

   //--- NvE 14-nov-1998 UU-SAP Utrecht

   return (1-Erfc(x));
}

//______________________________________________________________________________
Double_t TMath::Erfc(Double_t x)
{
   // Computation of the complementary error function erfc(x).
   // Erfc(x) = (2/sqrt(pi)) Integral(exp(-t^2))dt between x and infinity
   //
   // The algorithm is based on a Chebyshev fit as denoted in
   // Numerical Recipes 2nd ed. on p. 214 (W.H.Press et al.).
   //
   // The fractional error is always less than 1.2e-7.
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   // The parameters of the Chebyshev fit
   const Double_t a1 = -1.26551223,   a2 = 1.00002368,
                  a3 =  0.37409196,   a4 = 0.09678418,
                  a5 = -0.18628806,   a6 = 0.27886807,
                  a7 = -1.13520398,   a8 = 1.48851587,
                  a9 = -0.82215223,  a10 = 0.17087277;

   Double_t v = 1; // The return value
   Double_t z = Abs(x);

   if (z <= 0) return v; // erfc(0)=1

   Double_t t = 1/(1+0.5*z);

   v = t*Exp((-z*z) +a1+t*(a2+t*(a3+t*(a4+t*(a5+t*(a6+t*(a7+t*(a8+t*(a9+t*a10)))))))));

   if (x < 0) v = 2-v; // erfc(-x)=2-erfc(x)

   return v;
}

//______________________________________________________________________________
Double_t TMath::Freq(Double_t x)
{
   // Computation of the normal frequency function freq(x).
   // Freq(x) = (1/sqrt(2pi)) Integral(exp(-t^2/2))dt between -infinity and x
   //
   // translated from CERNLIB C300 by Rene Brun

   const Double_t C1 = 0.56418958354775629;
   const Double_t W2 = 1.41421356237309505;

   const Double_t p10 = 2.4266795523053175e+2,  q10 = 2.1505887586986120e+2,
                  p11 = 2.1979261618294152e+1,  q11 = 9.1164905404514901e+1,
                  p12 = 6.9963834886191355e+0,  q12 = 1.5082797630407787e+1,
                  p13 =-3.5609843701815385e-2,  q13 = 1;

   const Double_t p20 = 3.00459261020161601e+2, q20 = 3.00459260956983293e+2,
                  p21 = 4.51918953711872942e+2, q21 = 7.90950925327898027e+2,
                  p22 = 3.39320816734343687e+2, q22 = 9.31354094850609621e+2,
                  p23 = 1.52989285046940404e+2, q23 = 6.38980264465631167e+2,
                  p24 = 4.31622272220567353e+1, q24 = 2.77585444743987643e+2,
                  p25 = 7.21175825088309366e+0, q25 = 7.70001529352294730e+1,
                  p26 = 5.64195517478973971e-1, q26 = 1.27827273196294235e+1,
                  p27 =-1.36864857382716707e-7, q27 = 1;

   const Double_t p30 =-2.99610707703542174e-3, q30 = 1.06209230528467918e-2,
                  p31 =-4.94730910623250734e-2, q31 = 1.91308926107829841e-1,
                  p32 =-2.26956593539686930e-1, q32 = 1.05167510706793207e+0,
                  p33 =-2.78661308609647788e-1, q33 = 1.98733201817135256e+0,
                  p34 =-2.23192459734184686e-2, q34 = 1;

   Double_t v  = TMath::Abs(x)/W2;
   Double_t vv = v*v;
   Double_t ap, aq, h, hc, y;
   if (v < 0.5) {
      y=vv;
      ap=p13;
      aq=q13;
      ap = p12 +y*ap;
      ap = p11 +y*ap;
      ap = p10 +y*ap;
      aq = q12 +y*aq;
      aq = q11 +y*aq;
      aq = q10 +y*aq;
      h  = v*ap/aq;
      hc = 1-h;
   } else if (v < 4) {
      ap = p27;
      aq = q27;
      ap = p26 +v*ap;
      ap = p25 +v*ap;
      ap = p24 +v*ap;
      ap = p23 +v*ap;
      ap = p22 +v*ap;
      ap = p21 +v*ap;
      ap = p20 +v*ap;
      aq = q26 +v*aq;
      aq = q25 +v*aq;
      aq = q24 +v*aq;
      aq = q23 +v*aq;
      aq = q22 +v*aq;
      aq = q21 +v*aq;
      aq = q20 +v*aq;
      hc = TMath::Exp(-vv)*ap/aq;
      h  = 1-hc;
   } else {
      y  = 1/vv;
      ap = p34;
      aq = q34;
      ap = p33 +y*ap;
      ap = p32 +y*ap;
      ap = p31 +y*ap;
      ap = p30 +y*ap;
      aq = q33 +y*aq;
      aq = q32 +y*aq;
      aq = q31 +y*aq;
      aq = q30 +y*aq;
      hc = TMath::Exp(-vv)*(C1+y*ap/aq)/v;
      h  = 1-hc;
   }
   if (x > 0) return 0.5 +0.5*h;
   else return 0.5*hc;
}

//______________________________________________________________________________
Double_t TMath::Gamma(Double_t z)
{
   // Computation of gamma(z) for all z>0.
   //
   // The algorithm is based on the article by C.Lanczos [1] as denoted in
   // Numerical Recipes 2nd ed. on p. 207 (W.H.Press et al.).
   //
   // [1] C.Lanczos, SIAM Journal of Numerical Analysis B1 (1964), 86.
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   if (z<=0) return 0;

   Double_t v = LnGamma(z);
   return Exp(v);
}

//______________________________________________________________________________
Double_t TMath::Gamma(Double_t a,Double_t x)
{
   // Computation of the incomplete gamma function P(a,x)
   //
   // The algorithm is based on the formulas and code as denoted in
   // Numerical Recipes 2nd ed. on p. 210-212 (W.H.Press et al.).
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   if (a <= 0 || x <= 0) return 0;

   if (x < (a+1)) return GamSer(a,x);
   else           return GamCf(a,x);
}

//______________________________________________________________________________
Double_t TMath::GamCf(Double_t a,Double_t x)
{
   // Computation of the incomplete gamma function P(a,x)
   // via its continued fraction representation.
   //
   // The algorithm is based on the formulas and code as denoted in
   // Numerical Recipes 2nd ed. on p. 210-212 (W.H.Press et al.).
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   Int_t itmax    = 100;      // Maximum number of iterations
   Double_t eps   = 3.e-7;    // Relative accuracy
   Double_t fpmin = 1.e-30;   // Smallest Double_t value allowed here

   if (a <= 0 || x <= 0) return 0;

   Double_t gln = LnGamma(a);
   Double_t b   = x+1-a;
   Double_t c   = 1/fpmin;
   Double_t d   = 1/b;
   Double_t h   = d;
   Double_t an,del;
   for (Int_t i=1; i<=itmax; i++) {
      an = Double_t(-i)*(Double_t(i)-a);
      b += 2;
      d  = an*d+b;
      if (Abs(d) < fpmin) d = fpmin;
      c = b+an/c;
      if (Abs(c) < fpmin) c = fpmin;
      d   = 1/d;
      del = d*c;
      h   = h*del;
      if (Abs(del-1) < eps) break;
      //if (i==itmax) cout << "*GamCf(a,x)* a too large or itmax too small" << endl;
   }
   Double_t v = Exp(-x+a*Log(x)-gln)*h;
   return (1-v);
}

//______________________________________________________________________________
Double_t TMath::GamSer(Double_t a,Double_t x)
{
   // Computation of the incomplete gamma function P(a,x)
   // via its series representation.
   //
   // The algorithm is based on the formulas and code as denoted in
   // Numerical Recipes 2nd ed. on p. 210-212 (W.H.Press et al.).
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   Int_t itmax  = 100;   // Maximum number of iterations
   Double_t eps = 3.e-7; // Relative accuracy

   if (a <= 0 || x <= 0) return 0;

   Double_t gln = LnGamma(a);
   Double_t ap  = a;
   Double_t sum = 1/a;
   Double_t del = sum;
   for (Int_t n=1; n<=itmax; n++) {
      ap  += 1;
      del  = del*x/ap;
      sum += del;
      if (TMath::Abs(del) < Abs(sum*eps)) break;
      //if (n==itmax) cout << "*GamSer(a,x)* a too large or itmax too small" << endl;
   }
   Double_t v = sum*Exp(-x+a*Log(x)-gln);
   return v;
}

//______________________________________________________________________________
Double_t TMath::BreitWigner(Double_t x, Double_t mean, Double_t gamma)
{
   // Calculate a Breit Wigner function with mean and gamma

   Double_t bw = gamma/((x-mean)*(x-mean) + gamma*gamma/4);
   return bw/(2*Pi());
}

//______________________________________________________________________________
Double_t TMath::Gaus(Double_t x, Double_t mean, Double_t sigma)
{
   // Calculate a gaussian function with mean and sigma

    if (sigma == 0) return 1.e30;
    Double_t arg = (x-mean)/sigma;
    return TMath::Exp(-0.5*arg*arg);
}

//______________________________________________________________________________
Double_t TMath::Landau(Double_t x, Double_t mpv, Double_t sigma)
{
   // The LANDAU function with mpv(most probable value) and sigma.
   // This function has been adapted from the CERNLIB routine G110 denlan.

   Double_t p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
   Double_t q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

   Double_t p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
   Double_t q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

   Double_t p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
   Double_t q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

   Double_t p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
   Double_t q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

   Double_t p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
   Double_t q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

   Double_t p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
   Double_t q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

   Double_t a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};

   Double_t a2[2] = {-1.845568670,-4.284640743};

   if (sigma <= 0) return 0;
   Double_t v = (x-mpv)/sigma;
   Double_t u, ue, us, den;
   if (v < -5.5) {
      u   = TMath::Exp(v+1.0);
      ue  = TMath::Exp(-1/u);
      us  = TMath::Sqrt(u);
      den = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
   } else if(v < -1) {
       u   = TMath::Exp(-v-1);
       den = TMath::Exp(-u)*TMath::Sqrt(u)*
             (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
             (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
   } else if(v < 1) {
       den = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
             (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
   } else if(v < 5) {
       den = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
             (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
   } else if(v < 12) {
       u   = 1/v;
       den = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
                 (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
   } else if(v < 50) {
       u   = 1/v;
       den = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
                 (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
   } else if(v < 300) {
       u   = 1/v;
       den = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
                 (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
   } else {
       u   = 1/(v-v*TMath::Log(v)/(v+1));
       den = u*u*(1+(a2[0]+a2[1]*u)*u);
   }
   return den;
}

//______________________________________________________________________________
Double_t TMath::LnGamma(Double_t z)
{
   // Computation of ln[gamma(z)] for all z>0.
   //
   // The algorithm is based on the article by C.Lanczos [1] as denoted in
   // Numerical Recipes 2nd ed. on p. 207 (W.H.Press et al.).
   //
   // [1] C.Lanczos, SIAM Journal of Numerical Analysis B1 (1964), 86.
   //
   // The accuracy of the result is better than 2e-10.
   //
   //--- Nve 14-nov-1998 UU-SAP Utrecht

   if (z<=0) return 0;

   // Coefficients for the series expansion
   Double_t c[7] = { 2.5066282746310005, 76.18009172947146, -86.50532032941677
                   ,24.01409824083091,  -1.231739572450155, 0.1208650973866179e-2
                   ,-0.5395239384953e-5};

   Double_t x   = z;
   Double_t y   = x;
   Double_t tmp = x+5.5;
   tmp = (x+0.5)*Log(tmp)-tmp;
   Double_t ser = 1.000000000190015;
   for (Int_t i=1; i<7; i++) {
      y   += 1;
      ser += c[i]/y;
   }
   Double_t v = tmp+Log(c[0]*ser/x);
   return v;
}

//______________________________________________________________________________
Float_t TMath::Normalize(Float_t v[3])
{
   // Normalize a vector v in place
   // Return:
   //    The norm of the original vector

    Float_t d = Sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (d != 0)
    {
      v[0] /= d;
      v[1] /= d;
      v[2] /= d;
    }
    return d;
}

//______________________________________________________________________________
Double_t TMath::Normalize(Double_t v[3])
{
   // Normalize a vector v in place
   //  Return:
   //    The norm of the original vector

    Double_t d = Sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (d != 0)
    {
      v[0] /= d;
      v[1] /= d;
      v[2] /= d;
    }
    return d;
}

//______________________________________________________________________________
Float_t *TMath::Normal2Plane(Float_t p1[3],Float_t p2[3],Float_t p3[3], Float_t normal[3])
{
   // Calculate a normal vector of a plane
   //
   //  Input:
   //     Float_t *p1,*p2,*p3  -  3 3D points belonged the plane to define it.
   //
   //  Return:
   //     Pointer to 3D normal vector (normalized)
   //

   Float_t v1[3], v2[3];

   v1[0] = p2[0] - p1[0];
   v1[1] = p2[1] - p1[1];
   v1[2] = p2[2] - p1[2];

   v2[0] = p3[0] - p1[0];
   v2[1] = p3[1] - p1[1];
   v2[2] = p3[2] - p1[2];

   NormCross(v1,v2,normal);
   return normal;
}

//______________________________________________________________________________
Double_t *TMath::Normal2Plane(Double_t p1[3],Double_t p2[3],Double_t p3[3], Double_t normal[3])
{
   // Calculate a normal vector of a plane
   //
   //  Input:
   //     Float_t *p1,*p2,*p3  -  3 3D points belonged the plane to define it.
   //
   //  Return:
   //     Pointer to 3D normal vector (normalized)
   //

   Double_t v1[3], v2[3];

   v1[0] = p2[0] - p1[0];
   v1[1] = p2[1] - p1[1];
   v1[2] = p2[2] - p1[2];

   v2[0] = p3[0] - p1[0];
   v2[1] = p3[1] - p1[1];
   v2[2] = p3[2] - p1[2];

   NormCross(v1,v2,normal);
   return normal;
}

//______________________________________________________________________________
Double_t TMath::Prob(Double_t chi2,Int_t ndf)
{
   // Computation of the probability for a certain Chi-squared (chi2)
   // and number of degrees of freedom (ndf).
   //
   // Calculations are based on the incomplete gamma function P(a,x),
   // where a=ndf/2 and x=chi2/2.
   //
   // P(a,x) represents the probability that the observed Chi-squared
   // for a correct model should be less than the value chi2.
   //
   // The returned probability corresponds to 1-P(a,x),
   // which denotes the probability that an observed Chi-squared exceeds
   // the value chi2 by chance, even for a correct model.
   //
   //--- NvE 14-nov-1998 UU-SAP Utrecht

   if (ndf <= 0) return 0; // Set CL to zero in case ndf<=0

   if (chi2 <= 0) {
      if (chi2 < 0) return 0;
      else          return 1;
   }

   if (ndf==1) {
      Double_t v = 1.-Erf(Sqrt(chi2)/Sqrt(2.));
      return v;
   }

   // Gaussian approximation for large ndf
   Double_t q = Sqrt(2*chi2)-Sqrt(Double_t(2*ndf-1));
   if (ndf > 30 && q > 5) {
      Double_t v = 0.5*(1-Erf(q/Sqrt(2.)));
      return v;
   }

   // Evaluate the incomplete gamma function
   return (1-Gamma(0.5*ndf,0.5*chi2));
}

//______________________________________________________________________________
Double_t TMath::KolmogorovProb(Double_t z)
{
   // Calculates the Kolmogorov distribution function,
   //Begin_Html
   /*
   <img src="gif/kolmogorov.gif">
   */
   //End_Html
   // which gives the probability that Kolmogorov's test statistic will exceed
   // the value z assuming the null hypothesis. This gives a very powerful
   // test for comparing two one-dimensional distributions.
   // see, for example, Eadie et al, "statistocal Methods in Experimental
   // Physics', pp 269-270).
   //
   // This function returns the confidence level for the null hypothesis, where:
   //   z = dn*sqrt(n), and
   //   dn  is the maximum deviation between a hypothetical distribution
   //       function and an experimental distribution with
   //   n    events
   //
   // NOTE: To compare two experimental distributions with m and n events,
   //       use z = sqrt(m*n/(m+n))*dn
   //
   // Accuracy: The function is far too accurate for any imaginable application.
   //           Probabilities less than 10^-15 are returned as zero.
   //           However, remember that the formula is only valid for "large" n.
   // Theta function inversion formula is used for z <= 1
   //
   // This function was translated by Rene Brun from PROBKL in CERNLIB

   // cons[j] = -0.5*(PI*2*j-1)/2(**2
   const Double_t cons[3] = { -1.233700550136 , -11.10330496 , -30.84251376};
   // jf2[j] = -2* j**2
   const Double_t fj2[5]  = {-2. , -8. , -18. , -32. , -50.};
   const Double_t sqr2pi = 2.50662827463;

   Double_t p = 0;
   Int_t j;
   if (z < 0.2) return 1;
   if (z > 1) {  // use series in exp(z**2)
      Double_t c;
      Double_t sig2 = -2;
      Double_t z2 = z*z;
      for (j=0;j<5;j++) {
         sig2 = -sig2;
         c = fj2[j] *z2;
         if (c < -100) return p;
         p += sig2*TMath::Exp(c);
      }
      return p;
   }
   // z< 1  use series in exp(1/z**2)
   Double_t zinv = 1/z;
   Double_t a = sqr2pi*zinv;
   Double_t zinv2 = zinv*zinv;
   Double_t arg;
   for (j=0;j<3;j++) {
      arg = cons[j]*zinv2;
      if (arg < -30) continue;
      p += TMath::Exp(arg);
   }
   p = 1 - a*p;
   return p;
}

//______________________________________________________________________________
Int_t TMath::LocMin(Int_t n, const Short_t *a)
{
   // Return index of array with the minimum element
   // If more than one element is minimum returns first found

  if  (n <= 0) return -1;
  Short_t xmin = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMin(Int_t n, const Int_t *a)
{
   // Return index of array with the minimum element
   // If more than one element is minimum returns first found

  if  (n <= 0) return -1;
  Int_t xmin = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMin(Int_t n, const Float_t *a)
{
   // Return index of array with the minimum element
   // If more than one element is minimum returns first found

  if  (n <= 0) return -1;
  Float_t xmin = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMin(Int_t n, const Double_t *a)
{
   // Return index of array with the minimum element
   // If more than one element is minimum returns first found

  if  (n <= 0) return -1;
  Double_t xmin = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMin(Int_t n, const Long_t *a)
{
   // Return index of array with the minimum element
   // If more than one element is minimum returns first found

  if  (n <= 0) return -1;
  Long_t xmin = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmin > a[i])  {
         xmin = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMax(Int_t n, const Short_t *a)
{
   // Return index of array with the maximum element
   // If more than one element is maximum returns first found

  if  (n <= 0) return -1;
  Short_t xmax = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMax(Int_t n, const Int_t *a)
{
   // Return index of array with the maximum element
   // If more than one element is maximum returns first found

  if  (n <= 0) return -1;
  Int_t xmax = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMax(Int_t n, const Float_t *a)
{
   // Return index of array with the maximum element
   // If more than one element is maximum returns first found

  if  (n <= 0) return -1;
  Float_t xmax = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMax(Int_t n, const Double_t *a)
{
   // Return index of array with the maximum element
   // If more than one element is maximum returns first found

  if  (n <= 0) return -1;
  Double_t xmax = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::LocMax(Int_t n, const Long_t *a)
{
   // Return index of array with the maximum element
   // If more than one element is maximum returns first found

  if  (n <= 0) return -1;
  Long_t xmax = a[0];
  Int_t loc = 0;
  for  (Int_t i = 0; i < n; i++) {
     if (xmax < a[i])  {
         xmax = a[i];
         loc = i;
     }
  }
  return loc;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Short_t *array, Short_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Short_t **array, Short_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Int_t *array, Int_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Int_t **array, Int_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Float_t *array, Float_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Float_t **array, Float_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Double_t *array, Double_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Double_t **array, Double_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Long_t *array, Long_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

//______________________________________________________________________________
Int_t TMath::BinarySearch(Int_t n, const Long_t **array, Long_t value)
{
   // Binary search in an array of n values to locate value
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

   Int_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
}

//_____________________________________________________________________________
void TMath::Sort(Int_t n1, const Short_t *a, Int_t *index, Bool_t down)
{
   //  Sort the n1 elements of the Short_t array a.
   //  In output the array index contains the indices of the sorted array.
   //  If down is false sort in increasing order (default is decreasing order).
   //  This is a translation of the CERNLIB routine sortzv (M101)
   //  based on the quicksort algorithm.
   //  NOTE that the array index must be created with a length >= n1
   //  before calling this function.

   Int_t i,i1,n,i2,i3,i33,i222,iswap,n2;
   Int_t i22 = 0;
   Short_t ai;
   n = n1;
   if (n <= 0) return;
   if (n == 1) {index[0] = 0; return;}
   for (i=0;i<n;i++) index[i] = i+1;
   for (i1=2;i1<=n;i1++) {
      i3 = i1;
      i33 = index[i3-1];
      ai  = a[i33-1];
      while(1) {
         i2 = i3/2;
         if (i2 <= 0) break;
         i22 = index[i2-1];
         if (ai <= a[i22-1]) break;
         index[i3-1] = i22;
         i3 = i2;
      }
      index[i3-1] = i33;
   }

   while(1) {
      i3 = index[n-1];
      index[n-1] = index[0];
      ai = a[i3-1];
      n--;
      if(n-1 < 0) {index[0] = i3; break;}
      i1 = 1;
      while(2) {
         i2 = i1+i1;
         if (i2 <= n) i22 = index[i2-1];
         if (i2-n > 0) {index[i1-1] = i3; break;}
         if (i2-n < 0) {
            i222 = index[i2];
            if (a[i22-1] - a[i222-1] < 0) {
                i2++;
                i22 = i222;
            }
         }
         if (ai - a[i22-1] > 0) {index[i1-1] = i3; break;}
         index[i1-1] = i22;
         i1 = i2;
      }
   }
   for (i=0;i<n1;i++) index[i]--;
   if (!down) return;
   n2 = n1/2;
   for (i=0;i<n2;i++) {
      iswap         = index[i];
      index[i]      = index[n1-i-1];
      index[n1-i-1] = iswap;
   }
}

//_____________________________________________________________________________
void TMath::Sort(Int_t n1, const Int_t *a, Int_t *index, Bool_t down)
{
   //  Sort the n1 elements of the Int_t array a.
   //  In output the array index contains the indices of the sorted array.
   //  If down is false sort in increasing order (default is decreasing order).
   //  This is a translation of the CERNLIB routine sortzv (M101)
   //  based on the quicksort algorithm.
   //  NOTE that the array index must be created with a length >= n1
   //  before calling this function.

   Int_t i,i1,n,i2,i3,i33,i222,iswap,n2;
   Int_t i22 = 0;
   Int_t ai;
   n = n1;
   if (n <= 0) return;
   if (n == 1) {index[0] = 0; return;}
   for (i=0;i<n;i++) index[i] = i+1;
   for (i1=2;i1<=n;i1++) {
      i3 = i1;
      i33 = index[i3-1];
      ai  = a[i33-1];
      while(1) {
         i2 = i3/2;
         if (i2 <= 0) break;
         i22 = index[i2-1];
         if (ai <= a[i22-1]) break;
         index[i3-1] = i22;
         i3 = i2;
      }
      index[i3-1] = i33;
   }

   while(1) {
      i3 = index[n-1];
      index[n-1] = index[0];
      ai = a[i3-1];
      n--;
      if(n-1 < 0) {index[0] = i3; break;}
      i1 = 1;
      while(2) {
         i2 = i1+i1;
         if (i2 <= n) i22 = index[i2-1];
         if (i2-n > 0) {index[i1-1] = i3; break;}
         if (i2-n < 0) {
            i222 = index[i2];
            if (a[i22-1] - a[i222-1] < 0) {
                i2++;
                i22 = i222;
            }
         }
         if (ai - a[i22-1] > 0) {index[i1-1] = i3; break;}
         index[i1-1] = i22;
         i1 = i2;
      }
   }
   for (i=0;i<n1;i++) index[i]--;
   if (!down) return;
   n2 = n1/2;
   for (i=0;i<n2;i++) {
      iswap         = index[i];
      index[i]      = index[n1-i-1];
      index[n1-i-1] = iswap;
   }
}

//_____________________________________________________________________________
void TMath::Sort(Int_t n1, const Float_t *a, Int_t *index, Bool_t down)
{
   //  Sort the n1 elements of the Float_t array a.
   //  In output the array index contains the indices of the sorted array.
   //  If down is false sort in increasing order (default is decreasing order).
   //  This is a translation of the CERNLIB routine sortzv (M101)
   //  based on the quicksort algorithm.
   //  NOTE that the array index must be created with a length >= n1
   //  before calling this function.

   Int_t i,i1,n,i2,i3,i33,i222,iswap,n2;
   Int_t i22 = 0;
   Float_t ai;
   n = n1;
   if (n <= 0) return;
   if (n == 1) {index[0] = 0; return;}
   for (i=0;i<n;i++) index[i] = i+1;
   for (i1=2;i1<=n;i1++) {
      i3 = i1;
      i33 = index[i3-1];
      ai  = a[i33-1];
      while(1) {
         i2 = i3/2;
         if (i2 <= 0) break;
         i22 = index[i2-1];
         if (ai <= a[i22-1]) break;
         index[i3-1] = i22;
         i3 = i2;
      }
      index[i3-1] = i33;
   }

   while(1) {
      i3 = index[n-1];
      index[n-1] = index[0];
      ai = a[i3-1];
      n--;
      if(n-1 < 0) {index[0] = i3; break;}
      i1 = 1;
      while(2) {
         i2 = i1+i1;
         if (i2 <= n) i22 = index[i2-1];
         if (i2-n > 0) {index[i1-1] = i3; break;}
         if (i2-n < 0) {
            i222 = index[i2];
            if (a[i22-1] - a[i222-1] < 0) {
                i2++;
                i22 = i222;
            }
         }
         if (ai - a[i22-1] > 0) {index[i1-1] = i3; break;}
         index[i1-1] = i22;
         i1 = i2;
      }
   }
   for (i=0;i<n1;i++) index[i]--;
   if (!down) return;
   n2 = n1/2;
   for (i=0;i<n2;i++) {
      iswap         = index[i];
      index[i]      = index[n1-i-1];
      index[n1-i-1] = iswap;
   }
}

//_____________________________________________________________________________
void TMath::Sort(Int_t n1, const Double_t *a, Int_t *index, Bool_t down)
{
   //  Sort the n1 elements of the Double_t array a.
   //  In output the array index contains the indices of the sorted array.
   //  If down is false sort in increasing order (default is decreasing order).
   //  This is a translation of the CERNLIB routine sortzv (M101)
   //  based on the quicksort algorithm.
   //  NOTE that the array index must be created with a length >= n1
   //  before calling this function.

   Int_t i,i1,n,i2,i3,i33,i222,iswap,n2;
   Int_t i22 = 0;
   Double_t ai;
   n = n1;
   if (n <= 0) return;
   if (n == 1) {index[0] = 0; return;}
   for (i=0;i<n;i++) index[i] = i+1;
   for (i1=2;i1<=n;i1++) {
      i3 = i1;
      i33 = index[i3-1];
      ai  = a[i33-1];
      while(1) {
         i2 = i3/2;
         if (i2 <= 0) break;
         i22 = index[i2-1];
         if (ai <= a[i22-1]) break;
         index[i3-1] = i22;
         i3 = i2;
      }
      index[i3-1] = i33;
   }

   while(1) {
      i3 = index[n-1];
      index[n-1] = index[0];
      ai = a[i3-1];
      n--;
      if(n-1 < 0) {index[0] = i3; break;}
      i1 = 1;
      while(2) {
         i2 = i1+i1;
         if (i2 <= n) i22 = index[i2-1];
         if (i2-n > 0) {index[i1-1] = i3; break;}
         if (i2-n < 0) {
            i222 = index[i2];
            if (a[i22-1] - a[i222-1] < 0) {
                i2++;
                i22 = i222;
            }
         }
         if (ai - a[i22-1] > 0) {index[i1-1] = i3; break;}
         index[i1-1] = i22;
         i1 = i2;
      }
   }
   for (i=0;i<n1;i++) index[i]--;
   if (!down) return;
   n2 = n1/2;
   for (i=0;i<n2;i++) {
      iswap         = index[i];
      index[i]      = index[n1-i-1];
      index[n1-i-1] = iswap;
   }
}

//_____________________________________________________________________________
void TMath::Sort(Int_t n1, const Long_t *a, Int_t *index, Bool_t down)
{
   //  Sort the n1 elements of the Long_t array a.
   //  In output the array index contains the indices of the sorted array.
   //  If down is false sort in increasing order (default is decreasing order).
   //  This is a translation of the CERNLIB routine sortzv (M101)
   //  based on the quicksort algorithm.
   //  NOTE that the array index must be created with a length >= n1
   //  before calling this function.

   Int_t i,i1,n,i2,i3,i33,i222,iswap,n2;
   Int_t i22 = 0;
   Long_t ai;
   n = n1;
   if (n <= 0) return;
   if (n == 1) {index[0] = 0; return;}
   for (i=0;i<n;i++) index[i] = i+1;
   for (i1=2;i1<=n;i1++) {
      i3 = i1;
      i33 = index[i3-1];
      ai  = a[i33-1];
      while(1) {
         i2 = i3/2;
         if (i2 <= 0) break;
         i22 = index[i2-1];
         if (ai <= a[i22-1]) break;
         index[i3-1] = i22;
         i3 = i2;
      }
      index[i3-1] = i33;
   }

   while(1) {
      i3 = index[n-1];
      index[n-1] = index[0];
      ai = a[i3-1];
      n--;
      if(n-1 < 0) {index[0] = i3; break;}
      i1 = 1;
      while(2) {
         i2 = i1+i1;
         if (i2 <= n) i22 = index[i2-1];
         if (i2-n > 0) {index[i1-1] = i3; break;}
         if (i2-n < 0) {
            i222 = index[i2];
            if (a[i22-1] - a[i222-1] < 0) {
                i2++;
                i22 = i222;
            }
         }
         if (ai - a[i22-1] > 0) {index[i1-1] = i3; break;}
         index[i1-1] = i22;
         i1 = i2;
      }
   }
   for (i=0;i<n1;i++) index[i]--;
   if (!down) return;
   n2 = n1/2;
   for (i=0;i<n2;i++) {
      iswap         = index[i];
      index[i]      = index[n1-i-1];
      index[n1-i-1] = iswap;
   }
}


//______________________________________________________________________________
void TMath::BubbleHigh(Int_t Narr, Double_t *arr1, Int_t *arr2)
{
// bubble sort variant to obtain the order of an array's elements into an index
// in order to do more useful things than the standard built in functions.
// *arr1 is unchanged; 
// *arr2 is the array of indicies corresponding to the decending value of arr1
//  with arr2[0] corresponding to the largest arr1 value and arr2[Narr] the
//  smallest.
//
//  Author:        Adrian Bevan                    bevan@slac.stanford.edu
//  Copyright:     Liverpool University, July 2001

  if (Narr <= 0) return;
  double *localArr1 = new double[Narr];
  int    *localArr2 = new int[Narr];
  int iEl;
  int iEl2;

  for(iEl = 0; iEl < Narr; iEl++) {
    localArr1[iEl] = arr1[iEl];
    localArr2[iEl] = iEl;
  }

  for(iEl = 0; iEl < Narr; iEl++) {
    for(iEl2 = Narr-1; iEl2 > iEl; --iEl2) {
      if(localArr1[iEl2-1] < localArr1[iEl2]) {
        double tmp        = localArr1[iEl2-1];
        localArr1[iEl2-1] = localArr1[iEl2];
        localArr1[iEl2]   = tmp;

        int    tmp2       = localArr2[iEl2-1];
        localArr2[iEl2-1] = localArr2[iEl2];
        localArr2[iEl2]   = tmp2;
      }
    }
  }

  for(iEl = 0; iEl < Narr; iEl++) {
    arr2[iEl] = localArr2[iEl];
  }
  delete [] localArr2;
  delete [] localArr1;
}


//______________________________________________________________________________
void TMath::BubbleLow(Int_t Narr, Double_t *arr1, Int_t *arr2)
{
// opposite ordering of the array arr2[] to that of BubbleHigh.
//
//  Author:        Adrian Bevan                    bevan@slac.stanford.edu
//  Copyright:     Liverpool University, July 2001
   
  if (Narr <= 0) return;
  double *localArr1 = new double[Narr];
  int    *localArr2 = new int[Narr];
  int iEl;
  int iEl2;

  for(iEl = 0; iEl < Narr; iEl++) {
    localArr1[iEl] = arr1[iEl];
    localArr2[iEl] = iEl;
  }

  for(iEl = 0; iEl < Narr; iEl++) {
    for(iEl2 = Narr-1; iEl2 > iEl; --iEl2) {
      if(localArr1[iEl2-1] > localArr1[iEl2]) {
        double tmp        = localArr1[iEl2-1];
        localArr1[iEl2-1] = localArr1[iEl2];
        localArr1[iEl2]   = tmp;

        int    tmp2       = localArr2[iEl2-1];
        localArr2[iEl2-1] = localArr2[iEl2];
        localArr2[iEl2]   = tmp2;
      }
    }
  }

  for(iEl = 0; iEl < Narr; iEl++) {
    arr2[iEl] = localArr2[iEl];
  }
  delete [] localArr2;
  delete [] localArr1;
}

#ifdef OLD_HASH

//______________________________________________________________________________
ULong_t TMath::Hash(const void *txt, Int_t ntxt)
{
   // Calculates hash index from any char string.
   // Based on precalculated table of 256 specially selected random numbers.
   //
   //   For string:  i = TMath::Hash(string,nstring);
   //   For int:     i = TMath::Hash(&intword,sizeof(int));
   //   For pointer: i = TMath::Hash(&pointer,sizeof(void*));
   //
   //   Limitation: for ntxt>256 calculates hash only from first 256 bytes
   //
   //              V.Perev

   const UChar_t *uc = (const UChar_t*) txt;
   ULong_t  u = 0, uu = 0;

   static ULong_t utab[256] =
      {0xb93f6fc0,0x553dfc51,0xb22c1e8c,0x638462c0,0x13e81418,0x2836e171,0x7c4abb90,0xda1a4f39
      ,0x38f211d1,0x8c804829,0x95a6602d,0x4c590993,0x1810580a,0x721057d4,0x0f587215,0x9f49ce2a
      ,0xcd5ab255,0xab923a99,0x80890f39,0xbcfa2290,0x16587b52,0x6b6d3f0d,0xea8ff307,0x51542d5c
      ,0x189bf223,0x39643037,0x0e4a326a,0x214eca01,0x47645a9b,0x0f364260,0x8e9b2da4,0x5563ebd9
      ,0x57a31c1c,0xab365854,0xdd63ab1f,0x0b89acbd,0x23d57d33,0x1800a0fd,0x225ac60a,0xd0e51943
      ,0x6c65f669,0xcb966ea0,0xcbafda95,0x2e5c0c5f,0x2988e87e,0xc781cbab,0x3add3dc7,0x693a2c30
      ,0x42d6c23c,0xebf85f26,0x2544987e,0x2e315e3f,0xac88b5b5,0x7ebd2bbb,0xda07c87b,0x20d460f1
      ,0xc61c3f40,0x182046e7,0x3b6c3b66,0x2fc10d4a,0x0780dfbb,0xc437280c,0x0988dd07,0xe1498606
      ,0x8e61d728,0x4f1f3909,0x040a9682,0x49411b29,0x391b0e1c,0xd7905241,0xdd77d95b,0x88426c13
      ,0x33033e58,0xe158e30e,0x7e342647,0x1e09544b,0x4637353d,0x18ea0924,0x39212b08,0x12580ae8
      ,0x269a6f06,0x3e10b73b,0x123db33b,0x085412da,0x3bb5f464,0xd9b2d442,0x103d26bb,0xd0038bab
      ,0x45b6177f,0xfb48f1fe,0x99074c55,0xb545e82e,0x5f79fd0d,0x570f3ae4,0x57e43255,0x037a12ae
      ,0x4357bdb2,0x337c2c4d,0x2982499d,0x2ab72793,0x3900e7d1,0x57a6bb81,0x7503609b,0x3f39c0d0
      ,0x717b389d,0x5748034f,0x4698162b,0x5801b97c,0x1dfd5d7e,0xc1386d1c,0xa387a72a,0x084547e4
      ,0x2e54d8e9,0x2e2f384c,0xe09ccc20,0x8904b71e,0x3e24edc5,0x06a22e16,0x8a2be1df,0x9e5058b2
      ,0xe01a2f16,0x03325eed,0x587ecfe6,0x584d9cd3,0x32926930,0xe943d68c,0xa9442da8,0xf9650560
      ,0xf003871e,0x1109c663,0x7a2f2f89,0x1c2210bb,0x37335787,0xb92b382f,0xea605cb5,0x336bbe38
      ,0x08126bd3,0x1f8c2bd6,0xba6c46f2,0x1a4d1b83,0xc988180d,0xe2582505,0xa8a1b375,0x59a08c49
      ,0x3db54b48,0x44400f35,0x272d4e7f,0x5579f733,0x98eb590e,0x8ee09813,0x12cc9301,0xc85c402d
      ,0x135c1039,0x22318128,0x4063c705,0x87a8a3fa,0xfc14431f,0x6e27bf47,0x2d080a19,0x01dba174
      ,0xe343530b,0xaa1bfced,0x283bb2c8,0x5df250c8,0x4ff9140b,0x045039c1,0xa377780d,0x750f2661
      ,0x2b108918,0x0b152120,0x3cbc251f,0x5e87b350,0x060625bb,0xe068ba3b,0xdb73ebd7,0x66014ff3
      ,0xdb003000,0x161a3a0b,0xdc24e142,0x97ea5575,0x635a3cab,0xa719100a,0x256084db,0xc1f4a1e7
      ,0xe13388f2,0xb8199fc9,0x50c70dc9,0x08154211,0xd60e5220,0xe52c6592,0x584c5fe1,0xfe5e0875
      ,0x21072b30,0x3370d773,0x92608fe2,0x2d013d93,0x53414b3c,0x2c066142,0x64676644,0x0420887c
      ,0x35c01187,0x6822119b,0xf9bfe6df,0x273f4ee4,0x87973149,0x7b41282d,0x635d0d1f,0x5f7ecc1e
      ,0x14c3608a,0x462dfdab,0xc33d8808,0x1dcd995e,0x0fcb11ba,0x11755914,0x5a62044b,0x37f76755
      ,0x345bd058,0x8831c2b5,0x204a8468,0x3b0b1cd2,0x444e56f4,0x97a93e2c,0xd5f15067,0x266a95fa
      ,0xff4f8036,0x6160060d,0x930c472f,0xed922184,0x37120251,0xc0add74f,0x1c0bc89d,0x018d47f2
      ,0xff59ef66,0xd1901a17,0x91f6701b,0x0960082f,0x86f6a8f3,0x1154fecd,0x9867d1de,0x0945482f
      ,0x790ffcac,0xe5610011,0x4765637e,0xa745dbff,0x841fdcb3,0x4f7372a0,0x3c05013d,0xf1ac4ab7
      ,0x3bc5b5cc,0x49a73349,0x356a7f67,0x1174f031,0x11d32634,0x4413d301,0x1dd285c4,0x3fae4800
   };

   if (ntxt > 255) ntxt = 255;

   for ( ; ntxt--; uc++) {
      uu = uu<<1 ^ utab[(*uc) ^ ntxt];
      u ^= uu;
   }
   return u;
}

#else

//______________________________________________________________________________
ULong_t TMath::Hash(const void *txt, Int_t ntxt)
{
   // Calculates hash index from any char string.
   // Based on precalculated table of 256 specially selected numbers.
   // These numbers are selected in such a way, that for string
   // length == 4 (integer number) the hash is unambigous, i.e.
   // from hash value we can recalculate input (no degeneration).
   //
   // The quality of hash method is good enough, that
   // "random" numbers made as R = Hash(1), Hash(2), ...Hash(N)
   // tested by <R>, <R*R>, <Ri*Ri+1> gives the same result
   // as for libc rand().
   //
   // For string:  i = TMath::Hash(string,nstring);
   // For int:     i = TMath::Hash(&intword,sizeof(int));
   // For pointer: i = TMath::Hash(&pointer,sizeof(void*));
   //
   //              V.Perev

   static const ULong_t utab[] = {
       0xdd367647,0x9caf993f,0x3f3cc5ff,0xfde25082,0x4c764b21,0x89affca7,0x5431965c,0xce22eeec
      ,0xc61ab4dc,0x59cc93bd,0xed3107e3,0x0b0a287a,0x4712475a,0xce4a4c71,0x352c8403,0x94cb3cee
      ,0xc3ac509b,0x09f827a2,0xce02e37e,0x7b20bbba,0x76adcedc,0x18c52663,0x19f74103,0x6f30e47b
      ,0x132ea5a1,0xfdd279e0,0xa3d57d00,0xcff9cb40,0x9617f384,0x6411acfa,0xff908678,0x5c796b2c
      ,0x4471b62d,0xd38e3275,0xdb57912d,0x26bf953f,0xfc41b2a5,0xe64bcebd,0x190b7839,0x7e8e6a56
      ,0x9ca22311,0xef28aa60,0xe6b9208e,0xd257fb65,0x45781c2c,0x9a558ac3,0x2743e74d,0x839417a8
      ,0x06b54d5d,0x1a82bcb4,0x06e97a66,0x70abdd03,0xd163f30d,0x222ed322,0x777bfeda,0xab7a2e83
      ,0x8494e0cf,0x2dca2d4f,0x78f94278,0x33f04a09,0x402b6452,0x0cd8b709,0xdb72a39e,0x170e00a2
      ,0x26354faa,0x80e57453,0xcfe8d4e1,0x19e45254,0x04c291c3,0xeb503738,0x425af3bc,0x67836f2a
      ,0xfac22add,0xfafc2b8c,0x59b8c2a0,0x03e806f9,0xcb4938b9,0xccc942af,0xcee3ae2e,0xfbe748fa
      ,0xb223a075,0x85c49b5d,0xe4576ac9,0x0fbd46e2,0xb49f9cf5,0xf3e1e86a,0x7d7927fb,0x711afe12
      ,0xbf61c346,0x157c9956,0x86b6b046,0x2e402146,0xb2a57d8a,0x0d064bb1,0x30ce390c,0x3a3e1eb1
      ,0xbe7f6f8f,0xd8e30f87,0x5be2813c,0x73a3a901,0xa3aaf967,0x59ff092c,0x1705c798,0xf610dd66
      ,0xb17da91e,0x8e59534e,0x2211ea5b,0xa804ba03,0xd890efbb,0xb8b48110,0xff390068,0xc8c325b4
      ,0xf7289c07,0x787e104f,0x3d0df3d0,0x3526796d,0x10548055,0x1d59a42b,0xed1cc5a3,0xdd45372a
      ,0x31c50d57,0x65757cb7,0x3cfb85be,0xa329910d,0x6ad8ce39,0xa2de44de,0x0dd32432,0xd4a5b617
      ,0x8f3107fc,0x96485175,0x7f94d4f3,0x35097634,0xdb3ca782,0x2c0290b8,0x2045300b,0xe0f5d15a
      ,0x0e8cbffa,0xaa1cc38a,0x84008d6f,0xe9a9e794,0x5c602c25,0xfa3658fa,0x98d9d82b,0x3f1497e7
      ,0x84b6f031,0xe381eff9,0xfc7ae252,0xb239e05d,0xe3723d1f,0xcc3bda82,0xe21b1ad3,0x9104f7c8
      ,0x4bb2dfcd,0x4d14a8bc,0x6ba7f28c,0x8f89886c,0xad44c97e,0xb30fd975,0x633cdab1,0xf6c2d514
      ,0x067a49d2,0xdc461ad9,0xebaf9f3f,0x8dc6cac3,0x7a060f16,0xbab063ad,0xf42e25e6,0x60724ca6
      ,0xc7245c2e,0x4e48ea3c,0x9f89a609,0xa1c49890,0x4bb7f116,0xd722865c,0xa8ee3995,0x0ee070b1
      ,0xd9bffcc2,0xe55b64f9,0x25507a5a,0xc7a3e2b5,0x5f395f7e,0xe7957652,0x7381ba6a,0xde3d21f1
      ,0xdf1708dd,0xad0c9d0c,0x00cbc9e5,0x1160e833,0x6779582c,0x29d5d393,0x3f11d7d7,0x826a6b9b
      ,0xe73ff12f,0x8bad3d86,0xee41d3e5,0x7f0c8917,0x8089ef24,0x90c5cb28,0x2f7f8e6b,0x6966418a
      ,0x345453fb,0x7a2f8a68,0xf198593d,0xc079a532,0xc1971e81,0x1ab74e26,0x329ef347,0x7423d3d0
      ,0x942c510b,0x7f6c6382,0x14ae6acc,0x64b59da7,0x2356fa47,0xb6749d9c,0x499de1bb,0x92ffd191
      ,0xe8f2fb75,0x848dc913,0x3e8727d3,0x1dcffe61,0xb6e45245,0x49055738,0x827a6b55,0xb4788887
      ,0x7e680125,0xd19ce7ed,0x6b4b8e30,0xa8cadea2,0x216035d8,0x1c63bc3c,0xe1299056,0x1ad3dff4
      ,0x0aefd13c,0x0e7b921c,0xca0173c6,0x9995782d,0xcccfd494,0xd4b0ac88,0x53d552b1,0x630dae8b
      ,0xa8332dad,0x7139d9a2,0x5d76f2c4,0x7a4f8f1e,0x8d1aef97,0xd1cf285d,0xc8239153,0xce2608a9
      ,0x7b562475,0xe4b4bc83,0xf3db0c3a,0x70a65e48,0x6016b302,0xdebd5046,0x707e786a,0x6f10200c
   };

   static const ULong_t msk[] = { 0x11111111, 0x33333333, 0x77777777, 0xffffffff };

   const UChar_t *uc = (const UChar_t *) txt;
   ULong_t u = 0, uu = 0;
   UShort_t *s = (UShort_t *) &u;
   Int_t i, idx;

   for (i = 0; i < ntxt; i++) {
      idx = (uc[i] ^ i) & 255;
      uu  = (uu << 1) ^ (utab[idx] & msk[i & 3]);
      if (i & 3 == 3) u ^= uu;
   }
   if (i & 3) u ^= uu;

   u *= 1879048201;      // prime number
   s[0] += s[1];
   u *= 1979048191;      // prime number
   s[1] ^= s[0];
   u *= 2079048197;      // prime number

   return u;
}

#endif

//______________________________________________________________________________
ULong_t TMath::Hash(const char *txt)
{
   return Hash(txt, Int_t(strlen(txt)));
}

//______________________________________________________________________________
Double_t TMath::BesselI0(Double_t x)
{
// Computation of the modified Bessel function I_0(x) for any real x.
//
// The algorithm is based on the article by Abramowitz and Stegun [1]
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).
//
// [1] M.Abramowitz and I.A.Stegun, Handbook of Mathematical Functions,
//     Applied Mathematics Series vol. 55 (1964), Washington.
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 // Parameters of the polynomial approximation
 const Double_t p1=1.0,          p2=3.5156229,    p3=3.0899424,
                p4=1.2067492,    p5=0.2659732,    p6=3.60768e-2,  p7=4.5813e-3;

 const Double_t q1= 0.39894228,  q2= 1.328592e-2, q3= 2.25319e-3,
                q4=-1.57565e-3,  q5= 9.16281e-3,  q6=-2.057706e-2,
                q7= 2.635537e-2, q8=-1.647633e-2, q9= 3.92377e-3;

 Double_t ax = TMath::Abs(x);

 Double_t y=0, result=0;

 if (ax < 3.75) {
    y = pow(x/3.75,2);
    result = p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))));
 } else {
    y = 3.75/ax;
    result = (TMath::Exp(ax)/TMath::Sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))));
 }
 return result;
}

//______________________________________________________________________________
Double_t TMath::BesselK0(Double_t x)
{
// Computation of the modified Bessel function K_0(x) for positive real x.
//
// The algorithm is based on the article by Abramowitz and Stegun [1]
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).
//
// [1] M.Abramowitz and I.A.Stegun, Handbook of Mathematical Functions,
//     Applied Mathematics Series vol. 55 (1964), Washington.
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 // Parameters of the polynomial approximation
 const Double_t p1=-0.57721566,  p2=0.42278420,   p3=0.23069756,
                p4= 3.488590e-2, p5=2.62698e-3,   p6=1.0750e-4,    p7=7.4e-5;

 const Double_t q1= 1.25331414,  q2=-7.832358e-2, q3= 2.189568e-2,
                q4=-1.062446e-2, q5= 5.87872e-3,  q6=-2.51540e-3,  q7=5.3208e-4;

 if (x <= 0) {
    printf(" *K0* Invalid argument x = %g\n",x);
    return 0;
 }

 Double_t y=0,result=0;

 if (x <= 2) {
    y = x*x/4.;
    result = (-log(x/2.)*TMath::BesselI0(x))+(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
 } else {
    y = 2./x;
    result = (exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))));
 }
 return result;
}

//______________________________________________________________________________
Double_t TMath::BesselI1(Double_t x)
{
// Computation of the modified Bessel function I_1(x) for any real x.
//
// The algorithm is based on the article by Abramowitz and Stegun [1]
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).
//
// [1] M.Abramowitz and I.A.Stegun, Handbook of Mathematical Functions,
//     Applied Mathematics Series vol. 55 (1964), Washington.
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 // Parameters of the polynomial approximation
 const Double_t p1=0.5,          p2=0.87890594,   p3=0.51498869,
                p4=0.15084934,   p5=2.658733e-2,  p6=3.01532e-3,  p7=3.2411e-4;

 const Double_t q1= 0.39894228,  q2=-3.988024e-2, q3=-3.62018e-3,
                q4= 1.63801e-3,  q5=-1.031555e-2, q6= 2.282967e-2,
                q7=-2.895312e-2, q8= 1.787654e-2, q9=-4.20059e-3;

 Double_t ax = TMath::Abs(x);

 Double_t y=0, result=0;

 if (ax < 3.75) {
    y = pow(x/3.75,2);
    result = x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
 } else {
    y = 3.75/ax;
    result = (exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))));
    if (x < 0) result = -result;
 }
 return result;
}

//______________________________________________________________________________
Double_t TMath::BesselK1(Double_t x)
{
// Computation of the modified Bessel function K_1(x) for positive real x.
//
// The algorithm is based on the article by Abramowitz and Stegun [1]
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).
//
// [1] M.Abramowitz and I.A.Stegun, Handbook of Mathematical Functions,
//     Applied Mathematics Series vol. 55 (1964), Washington.
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 // Parameters of the polynomial approximation
 const Double_t p1= 1.,          p2= 0.15443144,  p3=-0.67278579,
                p4=-0.18156897,  p5=-1.919402e-2, p6=-1.10404e-3,  p7=-4.686e-5;

 const Double_t q1= 1.25331414,  q2= 0.23498619,  q3=-3.655620e-2,
                q4= 1.504268e-2, q5=-7.80353e-3,  q6= 3.25614e-3,  q7=-6.8245e-4;

 if (x <= 0) {
    printf(" *K1* Invalid argument x = %g\n",x);
    return 0;
 }

 Double_t y=0,result=0;

 if (x <= 2) {
    y = x*x/4.;
    result = (log(x/2.)*TMath::BesselI1(x))+(1./x)*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
 } else {
    y = 2./x;
    result = (exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))));
 }
 return result;
}

//______________________________________________________________________________
Double_t TMath::BesselK(Int_t n,Double_t x)
{
// Computation of the Integer Order Modified Bessel function K_n(x)
// for n=0,1,2,... and positive real x.
//
// The algorithm uses the recurrence relation
//
//               K_n+1(x) = (2n/x)*K_n(x) + K_n-1(x)
//
// as denoted in Numerical Recipes 2nd ed. on p. 232 (W.H.Press et al.).
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 if (x <= 0 || n < 0) {
    printf(" *K* Invalid argument(s) (n,x) = (%d, %g)\n",n,x);
    return 0;
 }

 if (n==0) return TMath::BesselK0(x);
 if (n==1) return TMath::BesselK1(x);

 // Perform upward recurrence for all x
 Double_t tox = 2./x;
 Double_t bkm = TMath::BesselK0(x);
 Double_t bk  = TMath::BesselK1(x);
 Double_t bkp = 0;
 for (Int_t j=1; j<n; j++) {
    bkp = bkm+Double_t(j)*tox*bk;
    bkm = bk;
    bk  = bkp;
 }
 return bk;
}

//______________________________________________________________________________
Double_t TMath::BesselI(Int_t n,Double_t x)
{
// Computation of the Integer Order Modified Bessel function I_n(x)
// for n=0,1,2,... and any real x.
//
// The algorithm uses the recurrence relation
//
//               I_n+1(x) = (-2n/x)*I_n(x) + I_n-1(x)
//
// as denoted in Numerical Recipes 2nd ed. on p. 232 (W.H.Press et al.).
//
//--- NvE 12-mar-2000 UU-SAP Utrecht

 Int_t iacc = 40; // Increase to enhance accuracy
 Double_t bigno = 1.e10, bigni = 1.e-10;

 if (n < 0) {
     printf(" *I* Invalid argument (n,x) = (%d, %g)\n",n,x);
     return 0;
 }

 if (n==0) return TMath::BesselI0(x);
 if (n==1) return TMath::BesselI1(x);

 if (TMath::Abs(x) < 1.e-10) return 0;

 Double_t tox = 2./TMath::Abs(x);
 Double_t bip = 0,bim = 0;
 Double_t bi  = 1;
 Double_t result = 0;
 Int_t m = 2*((n+Int_t(sqrt(Float_t(iacc*n))))); // Downward recurrence from even m
 for (Int_t j=m; j<=1; j--) {
    bim = bip+Double_t(j)*tox*bi;
    bip = bi;
    bi  = bim;
    // Renormalise to prevent overflows
    if (TMath::Abs(bi) > bigno)  {
       result *= bigni;
       bi     *= bigni;
       bip    *= bigni;
    }
    if (j==n) result=bip;
 }

 result *= TMath::BesselI0(x)/bi; // Normalise with BesselI0(x)
 if ((x < 0) && (n%2 == 1)) result = -result;

 return result;
}

//______________________________________________________________________________
Double_t TMath::BesselJ0(Double_t x) {
//Returns the Bessel function J0(x) for any real x.
//
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).

       Double_t ax,z;
       double xx,y,ans,ans1,ans2;

       if ((ax=fabs(x)) < 8.0) {
           y=x*x;
           ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
               +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
           ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
               +y*(59272.64853+y*(267.8532712+y*1.0))));
           ans=ans1/ans2;
       } else {
           z=8.0/ax;
           y=z*z;
           xx=ax-0.785398164;
           ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
               +y*(-0.2073370639e-5+y*0.2093887211e-6)));
           ans2 = -0.1562499995e-1+y*(0.1430488765e-3
               +y*(-0.6911147651e-5+y*(0.7621095161e-6
               -y*0.934935152e-7)));
           ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
       }return ans;
}

//______________________________________________________________________________
Double_t TMath::BesselJ1(Double_t x) {
//Returns the Bessel function J1(x) for any real x.
//
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).

     Double_t ax,z;
     Double_t xx,y,ans,ans1,ans2;

     if ((ax=fabs(x)) < 8.0) {
         y=x*x;
         ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
             +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
         ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
             +y*(99447.43394+y*(376.9991397+y*1.0))));
         ans=ans1/ans2;
     } else {
         z=8.0/ax;
         y=z*z;
         xx=ax-2.356194491;
         ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
             +y*(0.2457520174e-5+y*(-0.240337019e-6))));
         ans2=0.04687499995+y*(-0.2002690873e-3
             +y*(0.8449199096e-5+y*(-0.88228987e-6
             +y*0.105787412e-6)));
         ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
         if (x < 0.0) ans = -ans;
     }return ans;
}

//______________________________________________________________________________
Double_t TMath::BesselY0(Double_t x) {
//Returns the Bessel function Y0(x) for positive x.
//
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).

       Double_t z,xx,y,ans,ans1,ans2;

       if (x < 8.0) { 
           y=x*x;
           ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
               +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
           ans2=40076544269.0+y*(745249964.8+y*(7189466.438
               +y*(47447.26470+y*(226.1030244+y*1.0))));
           ans=(ans1/ans2)+0.636619772*TMath::BesselJ0(x)*log(x);
       } else {
         z=8.0/x;
         y=z*z;
         xx=x-0.785398164;
         ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
             +y*(-0.2073370639e-5+y*0.2093887211e-6)));
         ans2 = -0.1562499995e-1+y*(0.1430488765e-3
             +y*(-0.6911147651e-5+y*(0.7621095161e-6
             +y*(-0.934945152e-7)))); 
         ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
     }return ans;
}

//______________________________________________________________________________
Double_t TMath::BesselY1(Double_t x) {
//Returns the Bessel function Y1(x) for positive x.
//
// as denoted in Numerical Recipes 2nd ed. on p. 230 (W.H.Press et al.).

     Double_t z,xx,y,ans,ans1,ans2;

     if (x < 8.0) { 
         y=x*x;
         ans1=x*(-0.4900604943e13+y*(0.1275274390e13
             +y*(-0.5153438139e11+y*(0.7349264551e9
             +y*(-0.4237922726e7+y*0.8511937935e4)))));
         ans2=0.2499580570e14+y*(0.4244419664e12
             +y*(0.3733650367e10+y*(0.2245904002e8
             +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
           ans=(ans1/ans2)+0.636619772*(TMath::BesselJ1(x)*log(x)-1.0/x);
       } else {
           z=8.0/x;
           y=z*z;
           xx=x-2.356194491;
           ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
                 +y*(0.2457520174e-5+y*(-0.240337019e-6))));
           ans2=0.04687499995+y*(-0.2002690873e-3 
                 +y*(0.8449199096e-5+y*(-0.88228987e-6
                 +y*0.105787412e-6)));
           ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
       }return ans;
}

//______________________________________________________________________________   
Double_t TMath::Struve(Int_t n, Double_t x)
{
//   Struve Functions of Orders Zero and One 
//
//  n = 0; compute Struve function of order 0
//  n = 1; compute Struve function of order 1
//
//  converted from CERNLIB M342 by Rene Brun

    const Int_t n1 = 15;
    const Int_t n2 = 25;
    const Int_t n3 = 16;
    const Int_t n4 = 22;
    const Double_t c1[16] = { 1.00215845609911981, -1.63969292681309147,
                              1.50236939618292819, -.72485115302121872,
                               .18955327371093136, -.03067052022988,
                               .00337561447375194, -2.6965014312602e-4,
                              1.637461692612e-5,   -7.8244408508e-7,
                              3.021593188e-8,      -9.6326645e-10,
                              2.579337e-11,        -5.8854e-13,
                              1.158e-14,           -2e-16 };
    const Double_t c2[26] = {  .99283727576423943, -.00696891281138625,
                              1.8205103787037e-4,  -1.063258252844e-5,
                              9.8198294287e-7,     -1.2250645445e-7,
                              1.894083312e-8,      -3.44358226e-9,
                              7.1119102e-10,       -1.6288744e-10,
                              4.065681e-11,        -1.091505e-11,
                              3.12005e-12,         -9.4202e-13,
                              2.9848e-13,          -9.872e-14,
                              3.394e-14,           -1.208e-14,
                              4.44e-15,            -1.68e-15,
                              6.5e-16,             -2.6e-16,
                              1.1e-16,             -4e-17,
                              2e-17,               -1e-17 };
    const Double_t c3[17] = { .5578891446481605,   -.11188325726569816,
                             -.16337958125200939,   .32256932072405902,
                             -.14581632367244242,   .03292677399374035,
                             -.00460372142093573,  4.434706163314e-4,
                             -3.142099529341e-5,   1.7123719938e-6,
                             -7.416987005e-8,      2.61837671e-9,
                             -7.685839e-11,        1.9067e-12,
                             -4.052e-14,           7.5e-16,
                             -1e-17 };
    const Double_t c4[23] = { 1.00757647293865641,  .00750316051248257,
                             -7.043933264519e-5,   2.66205393382e-6,
                             -1.8841157753e-7,     1.949014958e-8,
                             -2.6126199e-9,        4.236269e-10,
                             -7.955156e-11,        1.679973e-11,
                             -3.9072e-12,          9.8543e-13,
                             -2.6636e-13,          7.645e-14,
                             -2.313e-14,           7.33e-15,
                             -2.42e-15,            8.3e-16,
                             -3e-16,               1.1e-16,
                             -4e-17,               2e-17,-1e-17 };

    const Double_t c0  = 2/TMath::Pi();
    const Double_t cc  = 2/(3*TMath::Pi());

    Int_t i, i1;
    Double_t alfa, h, r, y, b0, b1, b2;
    Double_t v = TMath::Abs(x);

    switch(n) {
//___________________________________________________________
       case 0: {
           v = TMath::Abs(x);
           if (v < 8) {
	      y = v/8;
	      h = 2*y*y -1;
	      alfa = h + h;
	      b1 = 0;
	      b2 = 0;
	      for (i = n1; i >= 0; --i) {
	         b0 = c1[i] + alfa*b1 - b2;
	         b2 = b1;
	         b1 = b0;
	      }
	      h = y*(b0 - h*b2);
           } else {
	      r = 1/v;
	      h = 128*r*r -1;
	      alfa = h + h;
	      b1 = 0;
	      b2 = 0;
	      for (i = n2; i >= 0; --i) {
	         b0 = c2[i] + alfa*b1 - b2;
	         b2 = b1;
	         b1 = b0;
	      }
	      h = TMath::BesselY0(v) + r*c0*(b0 - h*b2);
           }
           if (x < 0)  h = -h;
           return h;
        }
//___________________________________________________________     
        case 1: {
           if (v == 0) {
	      h = 0;
           } else if (v <= 0.3) {
	      y = v*v;
	      r = 1;
	      h = 1;
	      i1 = (Int_t)(-8. / TMath::Log10(v));
	      for (i = 1; i <= i1; ++i) {
	         h = -h*y / ((2*i+ 1)*(2*i + 3));
	         r += h;
	      }
	      h = cc*y*r;
           } else if (v < 8) {
	      h = v*v/32 -1;
	      alfa = h + h;
	      b1 = 0;
	      b2 = 0;
	      for (i = n3; i >= 0; --i) {
	         b0 = c3[i] + alfa*b1 - b2;
	         b2 = b1;
	         b1 = b0;
	      }
	      h = b0 - h*b2;
           } else {
	      h = 128/(v*v) -1;
	      alfa = h + h;
	      b1 = 0;
	      b2 = 0;
	      for (i = n4; i >= 0; --i) {
	         b0 = c4[i] + alfa*b1 - b2;
	         b2 = b1;
	         b1 = b0;
	      }
	      h = TMath::BesselY1(v) + c0*(b0 - h*b2);
           }
           return h;
       }
    }
    return 0;
}
