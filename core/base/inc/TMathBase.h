// @(#)root/base:
// Authors: Rene Brun, Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMathBase
#define ROOT_TMathBase


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath Base functions                                                 //
//                                                                      //
// Define the functions Min, Max, Abs, Sign, Range for all types.       //
// NB: These functions are unfortunately not available in a portable    //
// way in std::.                                                        //
//                                                                      //
// More functions are defined in TMath.h. TMathBase.h is designed to be //
// a stable file and used in place of TMath.h in the ROOT miniCore.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

#include <cstdlib>
#include <cmath>

namespace TMath {

   /* ************************* */
   /* * Fundamental constants * */
   /* ************************* */

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

   /* ************************** */
   /* * Mathematical Functions * */
   /* ************************** */

   // Abs
   inline Short_t  Abs(Short_t d);
   inline Int_t    Abs(Int_t d);
   inline Long_t   Abs(Long_t d);
   inline Long64_t Abs(Long64_t d);
   inline Float_t  Abs(Float_t d);
   inline Double_t Abs(Double_t d);
   inline LongDouble_t Abs(LongDouble_t d);

   // Even/Odd
   inline Bool_t   Even(Long_t a);
   inline Bool_t   Odd(Long_t a);

   // SignBit
   template<typename Integer>
   inline Bool_t SignBit(Integer a);
   inline Bool_t SignBit(Float_t a);
   inline Bool_t SignBit(Double_t a);
   inline Bool_t SignBit(LongDouble_t a);

   // Sign
   template<typename T1, typename T2>
   inline T1 Sign( T1 a, T2 b);
   inline Float_t  Sign(Float_t a, Float_t b);
   inline Double_t Sign(Double_t a, Double_t b);
   inline LongDouble_t Sign(LongDouble_t a, LongDouble_t b);

   // Min, Max of two scalars
   inline Short_t   Min(Short_t a, Short_t b);
   inline UShort_t  Min(UShort_t a, UShort_t b);
   inline Int_t     Min(Int_t a, Int_t b);
   inline UInt_t    Min(UInt_t a, UInt_t b);
   inline Long_t    Min(Long_t a, Long_t b);
   inline ULong_t   Min(ULong_t a, ULong_t b);
   inline Long64_t  Min(Long64_t a, Long64_t b);
   inline ULong64_t Min(ULong64_t a, ULong64_t b);
   inline Float_t   Min(Float_t a, Float_t b);
   inline Double_t  Min(Double_t a, Double_t b);

   inline Short_t   Max(Short_t a, Short_t b);
   inline UShort_t  Max(UShort_t a, UShort_t b);
   inline Int_t     Max(Int_t a, Int_t b);
   inline UInt_t    Max(UInt_t a, UInt_t b);
   inline Long_t    Max(Long_t a, Long_t b);
   inline ULong_t   Max(ULong_t a, ULong_t b);
   inline Long64_t  Max(Long64_t a, Long64_t b);
   inline ULong64_t Max(ULong64_t a, ULong64_t b);
   inline Float_t   Max(Float_t a, Float_t b);
   inline Double_t  Max(Double_t a, Double_t b);

   // Range
   inline Short_t   Range(Short_t lb, Short_t ub, Short_t x);
   inline Int_t     Range(Int_t lb, Int_t ub, Int_t x);
   inline Long_t    Range(Long_t lb, Long_t ub, Long_t x);
   inline ULong_t   Range(ULong_t lb, ULong_t ub, ULong_t x);
   inline Double_t  Range(Double_t lb, Double_t ub, Double_t x);

   //NextPrime is used by the Core classes.
   Long_t   NextPrime(Long_t x);   // Least prime number greater than x
}


//---- Even/odd ----------------------------------------------------------------

inline Bool_t TMath::Even(Long_t a)
   { return ! (a & 1); }

inline Bool_t TMath::Odd(Long_t a)
   { return (a & 1); }

//---- Abs ---------------------------------------------------------------------

inline Short_t TMath::Abs(Short_t d)
{ return (d >= 0) ? d : Short_t(-d);  }

inline Int_t TMath::Abs(Int_t d)
{ return std::abs(d); }

inline Long_t TMath::Abs(Long_t d)
{ return std::labs(d); }

inline Long64_t TMath::Abs(Long64_t d)
#if __cplusplus >= 201103
{ return std::llabs(d); }
#else
{ return (d >= 0) ? d : -d;  }
#endif

inline Float_t TMath::Abs(Float_t d)
{ return std::abs(d); }

inline Double_t TMath::Abs(Double_t d)
{ return std::abs(d); }

inline LongDouble_t TMath::Abs(LongDouble_t d)
{ return std::abs(d); }


//---- Sign Bit--------------------------------------------------------------------

template<typename Integer>
inline Bool_t TMath::SignBit( Integer a)
   { return (a < 0); }

inline Bool_t TMath::SignBit(Float_t a)
   { return std::signbit(a);  }

inline Bool_t TMath::SignBit(Double_t a)
   { return std::signbit(a);  }

inline Bool_t TMath::SignBit(LongDouble_t a)
   { return std::signbit(a);  }


//---- Sign --------------------------------------------------------------------

template<typename T1, typename T2>
inline T1 TMath::Sign( T1 a, T2 b)
   { return (SignBit(b)) ? - Abs(a) : Abs(a); }

inline Float_t TMath::Sign(Float_t a, Float_t b)
   { return std::copysign(a,b);  }

inline Double_t TMath::Sign(Double_t a, Double_t b)
   { return std::copysign(a,b);  }

inline LongDouble_t TMath::Sign(LongDouble_t a, LongDouble_t b)
   { return std::copysign(a,b);  }


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

inline Long64_t TMath::Min(Long64_t a, Long64_t b)
   { return a <= b ? a : b; }

inline ULong64_t TMath::Min(ULong64_t a, ULong64_t b)
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

inline Long64_t TMath::Max(Long64_t a, Long64_t b)
   { return a >= b ? a : b; }

inline ULong64_t TMath::Max(ULong64_t a, ULong64_t b)
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


#endif
