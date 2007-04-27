// @(#)root/math:$Name:  $:$Id: TMath.h,v 1.73 2007/02/09 10:15:39 rdm Exp $
// Authors: Rene Brun, Anna Kreshuk, Eddy Offermann, Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMath
#define ROOT_TMath


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath                                                                //
//                                                                      //
// Encapsulate most frequently used Math functions.                     //
// NB. The basic functions Min, Max, Abs, Sign and Range are defined    //
// in TMathBase.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TMathBase
#include "TMathBase.h"
#endif

namespace TMath {

   // Fundamental constants
   inline Double_t Pi()       { return 3.14159265358979323846; }
   inline Double_t TwoPi()    { return 2.0 * Pi(); }
   inline Double_t PiOver2()  { return Pi() / 2.0; }
   inline Double_t PiOver4()  { return Pi() / 4.0; }
   inline Double_t InvPi()    { return 1.0 / Pi(); }
   inline Double_t RadToDeg() { return 180.0 / Pi(); }
   inline Double_t DegToRad() { return Pi() / 180.0; }
   inline Double_t Sqrt2()    { return 1.4142135623730950488016887242097; }

   // e (base of natural log)
   inline Double_t E()        { return 2.71828182845904523536; }

   // natural log of 10 (to convert log to ln)
   inline Double_t Ln10()     { return 2.30258509299404568402; }

   // base-10 log of e  (to convert ln to log)
   inline Double_t LogE()     { return 0.43429448190325182765; }

   // velocity of light
   inline Double_t C()        { return 2.99792458e8; }        // m s^-1
   inline Double_t Ccgs()     { return 100.0 * C(); }         // cm s^-1
   inline Double_t CUncertainty() { return 0.0; }             // exact

   // gravitational constant
   inline Double_t G()        { return 6.673e-11; }           // m^3 kg^-1 s^-2
   inline Double_t Gcgs()     { return G() / 1000.0; }        // cm^3 g^-1 s^-2
   inline Double_t GUncertainty() { return 0.010e-11; }

   // G over h-bar C
   inline Double_t GhbarC()   { return 6.707e-39; }           // (GeV/c^2)^-2
   inline Double_t GhbarCUncertainty() { return 0.010e-39; }

   // standard acceleration of gravity
   inline Double_t Gn()       { return 9.80665; }             // m s^-2
   inline Double_t GnUncertainty() { return 0.0; }            // exact

   // Planck's constant
   inline Double_t H()        { return 6.62606876e-34; }      // J s
   inline Double_t Hcgs()     { return 1.0e7 * H(); }         // erg s
   inline Double_t HUncertainty() { return 0.00000052e-34; }

   // h-bar (h over 2 pi)
   inline Double_t Hbar()     { return 1.054571596e-34; }     // J s
   inline Double_t Hbarcgs()  { return 1.0e7 * Hbar(); }      // erg s
   inline Double_t HbarUncertainty() { return 0.000000082e-34; }

   // hc (h * c)
   inline Double_t HC()       { return H() * C(); }           // J m
   inline Double_t HCcgs()    { return Hcgs() * Ccgs(); }     // erg cm

   // Boltzmann's constant
   inline Double_t K()        { return 1.3806503e-23; }       // J K^-1
   inline Double_t Kcgs()     { return 1.0e7 * K(); }         // erg K^-1
   inline Double_t KUncertainty() { return 0.0000024e-23; }

   // Stefan-Boltzmann constant
   inline Double_t Sigma()    { return 5.6704e-8; }           // W m^-2 K^-4
   inline Double_t SigmaUncertainty() { return 0.000040e-8; }

   // Avogadro constant (Avogadro's Number)
   inline Double_t Na()       { return 6.02214199e+23; }      // mol^-1
   inline Double_t NaUncertainty() { return 0.00000047e+23; }

   // universal gas constant (Na * K)
   // http://scienceworld.wolfram.com/physics/UniversalGasConstant.html
   inline Double_t R()        { return K() * Na(); }          // J K^-1 mol^-1
   inline Double_t RUncertainty() { return R()*((KUncertainty()/K()) + (NaUncertainty()/Na())); }

   // Molecular weight of dry air
   // 1976 US Standard Atmosphere,
   // also see http://atmos.nmsu.edu/jsdap/encyclopediawork.html
   inline Double_t MWair()    { return 28.9644; }             // kg kmol^-1 (or gm mol^-1)

   // Dry Air Gas Constant (R / MWair)
   // http://atmos.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
   inline Double_t Rgair()    { return (1000.0 * R()) / MWair(); }  // J kg^-1 K^-1

   // Euler-Mascheroni Constant
   inline Double_t EulerGamma() { return 0.577215664901532860606512090082402431042; }

   // Elementary charge
   inline Double_t Qe()       { return 1.602176462e-19; }     // C
   inline Double_t QeUncertainty() { return 0.000000063e-19; }

   // Trigo
   inline Double_t Sin(Double_t);
   inline Double_t Cos(Double_t);
   inline Double_t Tan(Double_t);
   inline Double_t SinH(Double_t);
   inline Double_t CosH(Double_t);
   inline Double_t TanH(Double_t);
   inline Double_t ASin(Double_t);
   inline Double_t ACos(Double_t);
   inline Double_t ATan(Double_t);
   inline Double_t ATan2(Double_t, Double_t);
          Double_t ASinH(Double_t);
          Double_t ACosH(Double_t);
          Double_t ATanH(Double_t);
          Double_t Hypot(Double_t x, Double_t y);

   // Misc
   inline Double_t Sqrt(Double_t x);
   inline Double_t Ceil(Double_t x);
   inline Int_t    CeilNint(Double_t x);
   inline Double_t Floor(Double_t x);
   inline Int_t    FloorNint(Double_t x);
   inline Double_t Exp(Double_t x);
   inline Double_t Ldexp(Double_t x, Int_t exp);
          Double_t Factorial(Int_t i);
   inline Double_t Power(Double_t x, Double_t y);
   inline Double_t Log(Double_t x);
          Double_t Log2(Double_t x);
   inline Double_t Log10(Double_t x);
          Int_t    Nint(Float_t x);
          Int_t    Nint(Double_t x);
   inline Int_t    Finite(Double_t x);
   inline Int_t    IsNaN(Double_t x);

   // Some integer math
   Long_t   Hypot(Long_t x, Long_t y);     // sqrt(px*px + py*py)

   // Min, Max of an array
   Short_t   MinElement(Long64_t n, const Short_t *a);
   Int_t     MinElement(Long64_t n, const Int_t *a);
   Float_t   MinElement(Long64_t n, const Float_t *a);
   Double_t  MinElement(Long64_t n, const Double_t *a);
   Long_t    MinElement(Long64_t n, const Long_t *a);
   Long64_t  MinElement(Long64_t n, const Long64_t *a);
   Short_t   MaxElement(Long64_t n, const Short_t *a);
   Int_t     MaxElement(Long64_t n, const Int_t *a);
   Float_t   MaxElement(Long64_t n, const Float_t *a);
   Double_t  MaxElement(Long64_t n, const Double_t *a);
   Long_t    MaxElement(Long64_t n, const Long_t *a);
   Long64_t  MaxElement(Long64_t n, const Long64_t *a);

   // Locate Min, Max element number in an array
   Long64_t  LocMin(Long64_t n, const Short_t *a);
   Long64_t  LocMin(Long64_t n, const Int_t *a);
   Long64_t  LocMin(Long64_t n, const Float_t *a);
   Long64_t  LocMin(Long64_t n, const Double_t *a);
   Long64_t  LocMin(Long64_t n, const Long_t *a);
   Long64_t  LocMin(Long64_t n, const Long64_t *a);
   Long64_t  LocMax(Long64_t n, const Short_t *a);
   Long64_t  LocMax(Long64_t n, const Int_t *a);
   Long64_t  LocMax(Long64_t n, const Float_t *a);
   Long64_t  LocMax(Long64_t n, const Double_t *a);
   Long64_t  LocMax(Long64_t n, const Long_t *a);
   Long64_t  LocMax(Long64_t n, const Long64_t *a);

   //Mean, Geometric Mean, Median, RMS
   Double_t  Mean(Long64_t n, const Short_t *a, const Double_t *w=0);
   Double_t  Mean(Long64_t n, const Int_t *a,   const Double_t *w=0);
   Double_t  Mean(Long64_t n, const Float_t *a, const Double_t *w=0);
   Double_t  Mean(Long64_t n, const Double_t *a,const Double_t *w=0);
   Double_t  Mean(Long64_t n, const Long_t *a,  const Double_t *w=0);
   Double_t  Mean(Long64_t n, const Long64_t *a,const Double_t *w=0);
   Double_t  GeomMean(Long64_t n, const Short_t *a);
   Double_t  GeomMean(Long64_t n, const Int_t *a);
   Double_t  GeomMean(Long64_t n, const Float_t *a);
   Double_t  GeomMean(Long64_t n, const Double_t *a);
   Double_t  GeomMean(Long64_t n, const Long_t *a);
   Double_t  GeomMean(Long64_t n, const Long64_t *a);

   Double_t  RMS(Long64_t n, const Short_t *a);
   Double_t  RMS(Long64_t n, const Int_t *a);
   Double_t  RMS(Long64_t n, const Float_t *a);
   Double_t  RMS(Long64_t n, const Double_t *a);
   Double_t  RMS(Long64_t n, const Long_t *a);
   Double_t  RMS(Long64_t n, const Long64_t *a);

   template <class Element, class Index, class Size>  Double_t MedianImp(Size n, const Element *a, const Double_t *w=0, Index *work=0);
   Double_t  Median(Long64_t n, const Short_t *a,  const Double_t *w=0, Long64_t *work=0);
   Double_t  Median(Long64_t n, const Int_t *a,    const Double_t *w=0, Long64_t *work=0);
   Double_t  Median(Long64_t n, const Float_t *a,  const Double_t *w=0, Long64_t *work=0);
   Double_t  Median(Long64_t n, const Double_t *a, const Double_t *w=0, Long64_t *work=0);
   Double_t  Median(Long64_t n, const Long_t *a,   const Double_t *w=0, Long64_t *work=0);
   Double_t  Median(Long64_t n, const Long64_t *a, const Double_t *w=0, Long64_t *work=0);

   //k-th order statistic
   template <class Element, class Index, class Size>  Element KOrdStatImp(Size n, const Element *a, Size k, Index *work = 0);
   Short_t   KOrdStat(Long64_t n, const Short_t *a,  Long64_t k, Long64_t *work=0);
   Int_t     KOrdStat(Long64_t n, const Int_t *a,    Long64_t k, Long64_t *work=0);
   Float_t   KOrdStat(Long64_t n, const Float_t *a,  Long64_t k, Long64_t *work=0);
   Double_t  KOrdStat(Long64_t n, const Double_t *a, Long64_t k, Long64_t *work=0);
   Double_t  KOrdStat(Long64_t n, const Double_t *a, Long64_t k, Int_t *work);
   Long64_t  KOrdStat(Long64_t n, const Long_t *a,   Long64_t k, Long64_t *work=0);
   Long64_t  KOrdStat(Long64_t n, const Long64_t *a, Long64_t k, Long64_t *work=0);

   //Sample quantiles
   void      Quantiles(Int_t n, Int_t nprob, Double_t *x, Double_t *quantiles, Double_t *prob, Bool_t isSorted=kTRUE, Int_t *index = 0, Int_t type=7);

   // Range
   inline Short_t   Range(Short_t lb, Short_t ub, Short_t x);
   inline Int_t     Range(Int_t lb, Int_t ub, Int_t x);
   inline Long_t    Range(Long_t lb, Long_t ub, Long_t x);
   inline ULong_t   Range(ULong_t lb, ULong_t ub, ULong_t x);
   inline Double_t  Range(Double_t lb, Double_t ub, Double_t x);

   // Binary search
   Long64_t BinarySearch(Long64_t n, const Short_t *array,   Short_t value);
   Long64_t BinarySearch(Long64_t n, const Short_t **array,  Short_t value);
   Long64_t BinarySearch(Long64_t n, const Int_t *array,     Int_t value);
   Long64_t BinarySearch(Long64_t n, const Int_t **array,    Int_t value);
   Long64_t BinarySearch(Long64_t n, const Float_t *array,   Float_t value);
   Long64_t BinarySearch(Long64_t n, const Float_t **array,  Float_t value);
   Long64_t BinarySearch(Long64_t n, const Double_t *array,  Double_t value);
   Long64_t BinarySearch(Long64_t n, const Double_t **array, Double_t value);
   Long64_t BinarySearch(Long64_t n, const Long_t   *array,  Long_t value);
   Long64_t BinarySearch(Long64_t n, const Long_t   **array, Long_t value);
   Long64_t BinarySearch(Long64_t n, const Long64_t *array,  Long64_t value);
   Long64_t BinarySearch(Long64_t n, const Long64_t **array, Long64_t value);

   // Hashing
   ULong_t Hash(const void *txt, Int_t ntxt);
   ULong_t Hash(const char *str);

   // IsInside
   Bool_t IsInside(Int_t xp, Int_t yp, Int_t np, Int_t *x, Int_t *y);
   Bool_t IsInside(Float_t xp, Float_t yp, Int_t np, Float_t *x, Float_t *y);
   Bool_t IsInside(Double_t xp, Double_t yp, Int_t np, Double_t *x, Double_t *y);

   // Sorting
   template <class Element, class Index, class Size>  void SortImp(Size n, const Element*, Index* index, Bool_t down=kTRUE);
   void Sort(Int_t n,    const Short_t *a,  Int_t *index,    Bool_t down=kTRUE);
   void Sort(Int_t n,    const Int_t *a,    Int_t *index,    Bool_t down=kTRUE);
   void Sort(Int_t n,    const Float_t *a,  Int_t *index,    Bool_t down=kTRUE);
   void Sort(Int_t n,    const Double_t *a, Int_t *index,    Bool_t down=kTRUE);
   void Sort(Int_t n,    const Long_t *a,   Int_t *index,    Bool_t down=kTRUE);
   void Sort(Int_t n,    const Long64_t *a, Int_t *index,    Bool_t down=kTRUE);
   void Sort(Long64_t n, const Short_t *a,  Long64_t *index, Bool_t down=kTRUE);
   void Sort(Long64_t n, const Int_t *a,    Long64_t *index, Bool_t down=kTRUE);
   void Sort(Long64_t n, const Float_t *a,  Long64_t *index, Bool_t down=kTRUE);
   void Sort(Long64_t n, const Double_t *a, Long64_t *index, Bool_t down=kTRUE);
   void Sort(Long64_t n, const Long_t *a,   Long64_t *index, Bool_t down=kTRUE);
   void Sort(Long64_t n, const Long64_t *a, Long64_t *index, Bool_t down=kTRUE);
   void BubbleHigh(Int_t Narr, Double_t *arr1, Int_t *arr2);
   void BubbleLow (Int_t Narr, Double_t *arr1, Int_t *arr2);

   // Advanced
          Float_t  *Cross(const Float_t v1[3],const Float_t v2[3],Float_t out[3]);    // Calculate the Cross Product of two vectors
          Double_t *Cross(const Double_t v1[3],const Double_t v2[3],Double_t out[3]); // Calculate the Cross Product of two vectors
          Float_t   Normalize(Float_t v[3]);                              // Normalize a vector
          Double_t  Normalize(Double_t v[3]);                             // Normalize a vector
   inline Float_t   NormCross(const Float_t v1[3],const Float_t v2[3],Float_t out[3]);    // Calculate the Normalized Cross Product of two vectors
   inline Double_t  NormCross(const Double_t v1[3],const Double_t v2[3],Double_t out[3]); // Calculate the Normalized Cross Product of two vectors
          Float_t  *Normal2Plane(const Float_t v1[3],const Float_t v2[3],const Float_t v3[3], Float_t normal[3]);     // Calculate a normal vector of a plane
          Double_t *Normal2Plane(const Double_t v1[3],const Double_t v2[3],const Double_t v3[3], Double_t normal[3]); // Calculate a normal vector of a plane
          Bool_t    RootsCubic(const Double_t coef[4],Double_t &a, Double_t &b, Double_t &c);

          Double_t  BreitWigner(Double_t x, Double_t mean=0, Double_t gamma=1);
          Double_t  Gaus(Double_t x, Double_t mean=0, Double_t sigma=1, Bool_t norm=kFALSE);
          Double_t  Landau(Double_t x, Double_t mpv=0, Double_t sigma=1, Bool_t norm=kFALSE);
          Double_t  Voigt(Double_t x, Double_t sigma, Double_t lg, Int_t R = 4);

   // Bessel functions
          Double_t BesselI(Int_t n,Double_t x);  // integer order modified Bessel function I_n(x)
          Double_t BesselK(Int_t n,Double_t x);  // integer order modified Bessel function K_n(x)
          Double_t BesselI0(Double_t x);         // modified Bessel function I_0(x)
          Double_t BesselK0(Double_t x);         // modified Bessel function K_0(x)
          Double_t BesselI1(Double_t x);         // modified Bessel function I_1(x)
          Double_t BesselK1(Double_t x);         // modified Bessel function K_1(x)
          Double_t BesselJ0(Double_t x);         // Bessel function J0(x) for any real x
          Double_t BesselJ1(Double_t x);         // Bessel function J1(x) for any real x
          Double_t BesselY0(Double_t x);         // Bessel function Y0(x) for positive x
          Double_t BesselY1(Double_t x);         // Bessel function Y1(x) for positive x
          Double_t StruveH0(Double_t x);         // Struve functions of order 0
          Double_t StruveH1(Double_t x);         // Struve functions of order 1
          Double_t StruveL0(Double_t x);         // Modified Struve functions of order 0
          Double_t StruveL1(Double_t x);         // Modified Struve functions of order 1

   // Statistics
          Double_t Beta(Double_t p, Double_t q);
          Double_t BetaCf(Double_t x, Double_t a, Double_t b);
          Double_t BetaDist(Double_t x, Double_t p, Double_t q);
          Double_t BetaDistI(Double_t x, Double_t p, Double_t q);
          Double_t BetaIncomplete(Double_t x, Double_t a, Double_t b);
          Double_t Binomial(Int_t n,Int_t k);  // Calculate the binomial coefficient n over k
          Double_t BinomialI(Double_t p, Int_t n, Int_t k);
          Double_t CauchyDist(Double_t x, Double_t t=0, Double_t s=1);
          Double_t ChisquareQuantile(Double_t p, Double_t ndf);
          Double_t DiLog(Double_t x);
          Double_t Erf(Double_t x);
          Double_t ErfInverse(Double_t x);
          Double_t Erfc(Double_t x);
   inline Double_t ErfcInverse(Double_t x) {return TMath::ErfInverse(1-x);}
          Double_t FDist(Double_t F, Double_t N, Double_t M);
          Double_t FDistI(Double_t F, Double_t N, Double_t M);
          Double_t Freq(Double_t x);
          Double_t Gamma(Double_t z);
          Double_t Gamma(Double_t a,Double_t x);
          Double_t GammaDist(Double_t x, Double_t gamma, Double_t mu=0, Double_t beta=1);
          Double_t KolmogorovProb(Double_t z);
          Double_t KolmogorovTest(Int_t na, const Double_t *a, Int_t nb, const Double_t *b, Option_t *option);
          Double_t LandauI(Double_t x);
          Double_t LaplaceDist(Double_t x, Double_t alpha=0, Double_t beta=1);
          Double_t LaplaceDistI(Double_t x, Double_t alpha=0, Double_t beta=1);
          Double_t LnGamma(Double_t z);
          Double_t LogNormal(Double_t x, Double_t sigma, Double_t theta=0, Double_t m=1);
          Double_t NormQuantile(Double_t p);
          Bool_t   Permute(Int_t n, Int_t *a); // Find permutations
          Double_t Poisson(Double_t x, Double_t par);
          Double_t PoissonI(Double_t x, Double_t par);
          Double_t Prob(Double_t chi2,Int_t ndf);
          Double_t Student(Double_t T, Double_t ndf);
          Double_t StudentI(Double_t T, Double_t ndf);
          Double_t StudentQuantile(Double_t p, Double_t ndf, Bool_t lower_tail=kTRUE);
          Double_t Vavilov(Double_t x, Double_t kappa, Double_t beta2);
          Double_t VavilovI(Double_t x, Double_t kappa, Double_t beta2);
}


//---- Trig and other functions ------------------------------------------------

#include <float.h>

#if defined(R__WIN32) && !defined(__CINT__)
#   ifndef finite
#      define finite _finite
#      define isnan  _isnan
#   endif
#endif
#if defined(R__AIX) || defined(R__SOLARIS_CC50) || \
    defined(R__HPUX11) || defined(R__GLIBC) || \
    (defined(R__MACOSX) && defined(__INTEL_COMPILER))
// math functions are defined inline so we have to include them here
#   include <math.h>
#   ifdef R__SOLARIS_CC50
       extern "C" { int finite(double); }
#   endif
#   if defined(R__GLIBC) && defined(__STRICT_ANSI__)
#      ifndef finite
#         define finite __finite
#      endif
#      ifndef isnan
#         define isnan  __isnan
#      endif
#   endif
#else
// don't want to include complete <math.h>
extern "C" {
   extern double sin(double);
   extern double cos(double);
   extern double tan(double);
   extern double sinh(double);
   extern double cosh(double);
   extern double tanh(double);
   extern double asin(double);
   extern double acos(double);
   extern double atan(double);
   extern double atan2(double, double);
   extern double sqrt(double);
   extern double exp(double);
   extern double pow(double, double);
   extern double log(double);
   extern double log10(double);
#ifndef R__WIN32
#   if !defined(finite)
       extern int finite(double);
#   endif
#   if !defined(isnan)
       extern int isnan(double);
#   endif
   extern double ldexp(double, int);
   extern double ceil(double);
   extern double floor(double);
#else
   _CRTIMP double ldexp(double, int);
   _CRTIMP double ceil(double);
   _CRTIMP double floor(double);
#endif
}
#endif

inline Double_t TMath::Sin(Double_t x)
   { return sin(x); }

inline Double_t TMath::Cos(Double_t x)
   { return cos(x); }

inline Double_t TMath::Tan(Double_t x)
   { return tan(x); }

inline Double_t TMath::SinH(Double_t x)
   { return sinh(x); }

inline Double_t TMath::CosH(Double_t x)
   { return cosh(x); }

inline Double_t TMath::TanH(Double_t x)
   { return tanh(x); }

inline Double_t TMath::ASin(Double_t x)
   { if (x < -1.) return -TMath::Pi()/2;
     if (x >  1.) return  TMath::Pi()/2;
     return asin(x);
   }

inline Double_t TMath::ACos(Double_t x)
   { if (x < -1.) return TMath::Pi();
     if (x >  1.) return 0;
     return acos(x);
   }

inline Double_t TMath::ATan(Double_t x)
   { return atan(x); }

inline Double_t TMath::ATan2(Double_t y, Double_t x)
   { if (x != 0) return  atan2(y, x);
     if (y == 0) return  0;
     if (y >  0) return  Pi()/2;
     else        return -Pi()/2;
   }

inline Double_t TMath::Sqrt(Double_t x)
   { return sqrt(x); }

inline Double_t TMath::Ceil(Double_t x)
   { return ceil(x); }

inline Int_t TMath::CeilNint(Double_t x)
   { return TMath::Nint(ceil(x)); }

inline Double_t TMath::Floor(Double_t x)
   { return floor(x); }

inline Int_t TMath::FloorNint(Double_t x)
   { return TMath::Nint(floor(x)); }

inline Double_t TMath::Exp(Double_t x)
   { return exp(x); }

inline Double_t TMath::Ldexp(Double_t x, Int_t exp)
   { return ldexp(x, exp); }

inline Double_t TMath::Power(Double_t x, Double_t y)
   { return pow(x, y); }

inline Double_t TMath::Log(Double_t x)
   { return log(x); }

inline Double_t TMath::Log10(Double_t x)
   { return log10(x); }

inline Int_t TMath::Finite(Double_t x)
#ifdef R__HPUX11
   { return isfinite(x); }
#else
   { return finite(x); }
#endif

inline Int_t TMath::IsNaN(Double_t x)
   { return isnan(x); }

//-------- Advanced -------------

inline Float_t TMath::NormCross(const Float_t v1[3],const Float_t v2[3],Float_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

inline Double_t TMath::NormCross(const Double_t v1[3],const Double_t v2[3],Double_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

#endif
