// @(#)root/base:$Name:  $:$Id: TMath.h,v 1.7 2001/12/03 12:47:24 rdm Exp $
// Author: Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
// Encapsulate math routines. For the time being avoid templates.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TMath {

private:
   static Double_t GamCf(Double_t a,Double_t x);
   static Double_t GamSer(Double_t a,Double_t x);

public:

   static Double_t Pi() { return 3.14159265358979323846; }
   static Double_t E()  { return 2.7182818284590452354; }

   // Trigo
   static Double_t Sin(Double_t);
   static Double_t Cos(Double_t);
   static Double_t Tan(Double_t);
   static Double_t SinH(Double_t);
   static Double_t CosH(Double_t);
   static Double_t TanH(Double_t);
   static Double_t ASin(Double_t);
   static Double_t ACos(Double_t);
   static Double_t ATan(Double_t);
   static Double_t ATan2(Double_t, Double_t);
   static Double_t ASinH(Double_t);
   static Double_t ACosH(Double_t);
   static Double_t ATanH(Double_t);
   static Double_t Hypot(Double_t x, Double_t y);

   // Misc
   static Double_t Sqrt(Double_t x);
   static Double_t Ceil(Double_t x);
   static Double_t Floor(Double_t x);
   static Double_t Exp(Double_t);
   static Double_t Power(Double_t x, Double_t y);
   static Double_t Log(Double_t x);
   static Double_t Log2(Double_t x);
   static Double_t Log10(Double_t x);
   static Int_t    Nint(Float_t x);
   static Int_t    Nint(Double_t x);
   static Int_t    Finite(Double_t x);
   static Int_t    IsNaN(Double_t x);

   // Some integer math
   static Long_t   NextPrime(Long_t x);   // Least prime number greater than x
   static Long_t   Sqrt(Long_t x);
   static Long_t   Hypot(Long_t x, Long_t y);     // sqrt(px*px + py*py)

   // Abs
   static Short_t  Abs(Short_t d);
   static Int_t    Abs(Int_t d);
   static Long_t   Abs(Long_t d);
   static Float_t  Abs(Float_t d);
   static Double_t Abs(Double_t d);

   // Even/Odd
   static Bool_t Even(Long_t a);
   static Bool_t Odd(Long_t a);

   // Sign
   static Short_t  Sign(Short_t a, Short_t b);
   static Int_t    Sign(Int_t a, Int_t b);
   static Long_t   Sign(Long_t a, Long_t b);
   static Float_t  Sign(Float_t a, Float_t b);
   static Double_t Sign(Double_t a, Double_t b);

   // Min
   static Short_t  Min(Short_t a, Short_t b);
   static UShort_t Min(UShort_t a, UShort_t b);
   static Int_t    Min(Int_t a, Int_t b);
   static UInt_t   Min(UInt_t a, UInt_t b);
   static Long_t   Min(Long_t a, Long_t b);
   static ULong_t  Min(ULong_t a, ULong_t b);
   static Float_t  Min(Float_t a, Float_t b);
   static Double_t Min(Double_t a, Double_t b);

   // Max
   static Short_t  Max(Short_t a, Short_t b);
   static UShort_t Max(UShort_t a, UShort_t b);
   static Int_t    Max(Int_t a, Int_t b);
   static UInt_t   Max(UInt_t a, UInt_t b);
   static Long_t   Max(Long_t a, Long_t b);
   static ULong_t  Max(ULong_t a, ULong_t b);
   static Float_t  Max(Float_t a, Float_t b);
   static Double_t Max(Double_t a, Double_t b);

   // Locate Min, Max
   static Int_t  LocMin(Int_t n, Short_t *a);
   static Int_t  LocMin(Int_t n, Int_t *a);
   static Int_t  LocMin(Int_t n, Float_t *a);
   static Int_t  LocMin(Int_t n, Double_t *a);
   static Int_t  LocMax(Int_t n, Short_t *a);
   static Int_t  LocMax(Int_t n, Int_t *a);
   static Int_t  LocMax(Int_t n, Float_t *a);
   static Int_t  LocMax(Int_t n, Double_t *a);

   // Range
   static Short_t  Range(Short_t lb, Short_t ub, Short_t x);
   static Int_t    Range(Int_t lb, Int_t ub, Int_t x);
   static Long_t   Range(Long_t lb, Long_t ub, Long_t x);
   static ULong_t  Range(ULong_t lb, ULong_t ub, ULong_t x);
   static Double_t Range(Double_t lb, Double_t ub, Double_t x);

   // Binary search
   static Int_t BinarySearch(Int_t n, Short_t *array, Short_t value);
   static Int_t BinarySearch(Int_t n, Short_t **array, Short_t value);
   static Int_t BinarySearch(Int_t n, Int_t *array, Int_t value);
   static Int_t BinarySearch(Int_t n, Int_t **array, Int_t value);
   static Int_t BinarySearch(Int_t n, Float_t *array, Float_t value);
   static Int_t BinarySearch(Int_t n, Float_t **array, Float_t value);
   static Int_t BinarySearch(Int_t n, Double_t *array, Double_t value);
   static Int_t BinarySearch(Int_t n, Double_t **array, Double_t value);

   // Hashing
   static ULong_t Hash(const void *txt, Int_t ntxt);
   static ULong_t Hash(const char *str);

   // Sorting
   static void Sort(Int_t n, Short_t *a,  Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, Int_t *a,    Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, Float_t *a,  Int_t *index, Bool_t down=kTRUE);
   static void Sort(Int_t n, Double_t *a, Int_t *index, Bool_t down=kTRUE);

   // Advanced
   static Float_t *Cross(Float_t v1[3],Float_t v2[3],Float_t out[3]);     // Calculate the Cross Product of two vectors
   static Float_t  Normalize(Float_t v[3]);                               // Normalize a vector
   static Float_t  NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Float_t *Normal2Plane(Float_t v1[3],Float_t v2[3],Float_t v3[3], Float_t normal[3]); // Calcualte a normal vector of a plane

   static Double_t *Cross(Double_t v1[3],Double_t v2[3],Double_t out[3]);// Calculate the Cross Product of two vectors
   static Double_t  Erf(Double_t x);
   static Double_t  Erfc(Double_t x);
   static Double_t  Freq(Double_t x);
   static Double_t  Gamma(Double_t z);
   static Double_t  Gamma(Double_t a,Double_t x);
   static Double_t  Gaus(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t  Landau(Double_t x, Double_t mean=0, Double_t sigma=1);
   static Double_t  LnGamma(Double_t z);
   static Double_t  Normalize(Double_t v[3]);                             // Normalize a vector
   static Double_t  NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3]); // Calculate the Normalized Cross Product of two vectors
   static Double_t *Normal2Plane(Double_t v1[3],Double_t v2[3],Double_t v3[3], Double_t normal[3]); // Calcualte a normal vector of a plane
   static Double_t  Prob(Double_t chi2,Int_t ndf);
   static Double_t  KolmogorovProb(Double_t z);

   // Bessel functions
   static Double_t BesselI(Int_t n,Double_t x);         // Compute integer order modified Bessel function I_n(x)
   static Double_t BesselK(Int_t n,Double_t x);         // Compute integer order modified Bessel function K_n(x)
   static Double_t BesselI0(Double_t x);                // Compute modified Bessel function I_0(x)
   static Double_t BesselK0(Double_t x);                // Compute modified Bessel function K_0(x)
   static Double_t BesselI1(Double_t x);                // Compute modified Bessel function I_1(x)
   static Double_t BesselK1(Double_t x);                // Compute modified Bessel function K_1(x)

   ClassDef(TMath,0)  //Interface to math routines
};


//---- Even/odd ----------------------------------------------------------------

inline Bool_t TMath::Even(Long_t a)
   { return ! (a & 1); }

inline Bool_t TMath::Odd(Long_t a)
   { return (a & 1); }

//---- Abs ---------------------------------------------------------------------

inline Short_t TMath::Abs(Short_t d)
   { return (d > 0) ? d : -d; }

inline Int_t TMath::Abs(Int_t d)
   { return (d > 0) ? d : -d; }

inline Long_t TMath::Abs(Long_t d)
   { return (d > 0) ? d : -d; }

inline Float_t TMath::Abs(Float_t d)
   { return (d > 0) ? d : -d; }

inline Double_t TMath::Abs(Double_t d)
   { return (d > 0) ? d : -d; }

//---- Sign --------------------------------------------------------------------

inline Short_t TMath::Sign(Short_t a, Short_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Int_t TMath::Sign(Int_t a, Int_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Long_t TMath::Sign(Long_t a, Long_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Float_t TMath::Sign(Float_t a, Float_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Double_t TMath::Sign(Double_t a, Double_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

//---- Min ---------------------------------------------------------------------

inline Short_t TMath::Min(Short_t a, Short_t b)
   { return a <= b ? a : b; }

inline UShort_t TMath::Min(UShort_t a, UShort_t b)
   { return a <= b ? a : b; }

inline Int_t TMath::Min(Int_t a, Int_t b)
   { return a <= b ? a : b; }

inline UInt_t TMath::Min(UInt_t a, UInt_t b)
   { return a <= b ? a : b; }

inline Long_t TMath::Min(Long_t a, Long_t b)
   { return a <= b ? a : b; }

inline ULong_t TMath::Min(ULong_t a, ULong_t b)
   { return a <= b ? a : b; }

inline Float_t TMath::Min(Float_t a, Float_t b)
   { return a <= b ? a : b; }

inline Double_t TMath::Min(Double_t a, Double_t b)
   { return a <= b ? a : b; }

//---- Max ---------------------------------------------------------------------

inline Short_t TMath::Max(Short_t a, Short_t b)
   { return a >= b ? a : b; }

inline UShort_t TMath::Max(UShort_t a, UShort_t b)
   { return a >= b ? a : b; }

inline Int_t TMath::Max(Int_t a, Int_t b)
   { return a >= b ? a : b; }

inline UInt_t TMath::Max(UInt_t a, UInt_t b)
   { return a >= b ? a : b; }

inline Long_t TMath::Max(Long_t a, Long_t b)
   { return a >= b ? a : b; }

inline ULong_t TMath::Max(ULong_t a, ULong_t b)
   { return a >= b ? a : b; }

inline Float_t TMath::Max(Float_t a, Float_t b)
   { return a >= b ? a : b; }

inline Double_t TMath::Max(Double_t a, Double_t b)
   { return a >= b ? a : b; }

//---- Range -------------------------------------------------------------------

inline Short_t TMath::Range(Short_t lb, Short_t ub, Short_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Int_t TMath::Range(Int_t lb, Int_t ub, Int_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Long_t TMath::Range(Long_t lb, Long_t ub, Long_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline ULong_t TMath::Range(ULong_t lb, ULong_t ub, ULong_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Double_t TMath::Range(Double_t lb, Double_t ub, Double_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

//---- Trig and other functions ------------------------------------------------


#include <float.h>

#ifdef R__WIN32
#   ifndef finite
#      define finite _finite
#      define isnan  _isnan
#   endif
#endif
#if defined(R__AIX) || defined(R__MAC) || defined(R__SOLARIS_CC50) || \
    defined(R__USESTHROW)
// math functions are defined inline so we have to include them here
#   include <math.h>
#   ifdef R__SOLARIS_CC50
       extern "C" { int finite(double); }
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
   extern int    finite(double);
   extern int    isnan(double);
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
   { return asin(x); }

inline Double_t TMath::ACos(Double_t x)
   { return acos(x); }

inline Double_t TMath::ATan(Double_t x)
   { return atan(x); }

inline Double_t TMath::ATan2(Double_t y, Double_t x)
   { return x != 0 ? atan2(y, x) : (y > 0 ? Pi()/2 : -Pi()/2); }

inline Double_t TMath::Sqrt(Double_t x)
   { return sqrt(x); }

inline Double_t TMath::Exp(Double_t x)
   { return exp(x); }

inline Double_t TMath::Power(Double_t x, Double_t y)
   { return pow(x, y); }

inline Double_t TMath::Log(Double_t x)
   { return log(x); }

inline Double_t TMath::Log10(Double_t x)
   { return log10(x); }

inline Int_t TMath::Finite(Double_t x)
   { return finite(x); }

inline Int_t TMath::IsNaN(Double_t x)
   { return isnan(x); }

//-------- Advanced -------------

inline Float_t TMath::NormCross(Float_t v1[3],Float_t v2[3],Float_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

inline Double_t TMath::NormCross(Double_t v1[3],Double_t v2[3],Double_t out[3])
{
   // Calculate the Normalized Cross Product of two vectors
   return Normalize(Cross(v1,v2,out));
}

#endif
